from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import logging
import torch
import torch.nn as nn
import torch.optim as optim
# ОПТИМИЗАЦИЯ: Импортируем GradScaler для смешанного вычисления (AMP)
from torch.amp import GradScaler
import numpy as np
import os
import multiprocessing
import copy

logger = logging.getLogger(__name__)

# Проверяем доступность CUDA и выбираем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelWrapper:
    """
    Базовый класс-обертка для любой модели.
    Определяет "контракт": любая модель должна иметь методы fit и predict.
    """

    def __init__(self, model_params):
        self.model = None
        self.params = model_params

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class SklearnLinearModel(ModelWrapper):
    """
    Реализация обертки для линейных моделей из Scikit-learn.
    """

    def __init__(self, model_type='linear', model_params=None, fit_intercept=True):
        super().__init__(model_params)
        self.model_type = model_type
        self.fit_intercept = fit_intercept
        self.model = self._create_model()

    def _create_model(self):
        """Фабрика для создания экземпляра модели на основе конфига."""
        model_map = {
            'ridge': Ridge,
            'lasso': Lasso,
            'elasticnet': ElasticNet,
            'linear': LinearRegression
        }
        model_class = model_map.get(self.model_type, LinearRegression)

        # Берем параметры для нужной модели из общего словаря
        specific_params = self.params.get(self.model_type, {})

        try:
            # Для всех, кроме LinearRegression, можно передать alpha и т.д.
            if model_class != LinearRegression:
                return model_class(fit_intercept=self.fit_intercept, **specific_params)
            else:  # LinearRegression не принимает alpha
                return LinearRegression(fit_intercept=self.fit_intercept)
        except Exception as e:
            logger.error(
                f"Ошибка при создании модели '{self.model_type}' с параметрами {specific_params}: {e}. Используется LinearRegression.")
            return LinearRegression(fit_intercept=self.fit_intercept)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class NeuralNetwork(nn.Module):
    """Вспомогательный класс, определяющий архитектуру нейронной сети."""

    def __init__(self, input_dim, hidden_layers, activation_fn_str='relu'):
        super(NeuralNetwork, self).__init__()

        layers = []

        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }
        activation_fn = activation_map.get(activation_fn_str, nn.ReLU())

        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation_fn)
            last_dim = hidden_dim

        # Выходной слой (1 нейрон для регрессии)
        layers.append(nn.Linear(last_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Робастная пересылка тензора на нужное устройство
        # Получаем устройство, на котором находятся параметры модели
        model_device = next(self.parameters()).device
        # Перемещаем входной тензор на то же устройство, если они отличаются
        if x.device != model_device:
            x = x.to(model_device)
        return self.network(x)


class NeuralNetworkModel(ModelWrapper):
    """Обертка для модели на основе PyTorch."""

    def __init__(self, model_params, input_dim):
        super().__init__(model_params)
        self.input_dim = input_dim
        self.device = device

        if self.device.type == 'cuda':
            logger.info("CUDA доступна. Модель нейросети будет использовать GPU.")
        else:
            logger.info("CUDA не найдена. Модель нейросети будет использовать CPU.")

        self.model = self._create_model()
        self.has_logged_training_start = False

        # Настройка оптимизатора
        optimizer_str = self.params.get('optimizer', 'adam').lower()
        lr = self.params.get('learning_rate', 0.001)
        if optimizer_str == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:  # adam по умолчанию
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Функции потерь: MSE для обучения, MAE для логирования
        self.criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()  # ИЗМЕНЕНИЕ: Добавили MAE

        # ОПТИМИЗАЦИЯ: Инициализируем GradScaler для AMP.
        # Он будет активен только если используется CUDA, в противном случае это "пустышка".
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))

    def _create_model(self):
        """Создает экземпляр нейронной сети и перемещает его на нужное устройство."""
        hidden_layers = self.params.get('hidden_layers', [64, 32])
        activation_fn = self.params.get('activation_fn', 'relu')
        # Перемещаем модель на GPU или CPU
        return NeuralNetwork(self.input_dim, hidden_layers, activation_fn).to(self.device)

    def fit(self, X, y):
        """Обучение модели."""
        self.model.train()  # Переводим модель в режим обучения

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        # --- Параметры обучения и ранней остановки ---
        epochs = self.params.get('epochs', 50)
        batch_size = self.params.get('batch_size', 32)
        early_stopping_enabled = self.params.get('early_stopping_enabled', False)
        patience = self.params.get('early_stopping_patience', 3)
        min_delta = self.params.get('early_stopping_min_delta', 0.0001)

        best_loss = np.inf
        best_mae_at_best_mse = np.inf  # ИЗМЕНЕНИЕ: Храним MAE для лучшей эпохи
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        # -------------------------------------------

        use_pin_memory = self.device.type == 'cuda'
        num_workers = 0
        if os.name != 'nt':
            try:
                cpu_count = len(os.sched_getaffinity(0))
            except AttributeError:
                cpu_count = multiprocessing.cpu_count()
            num_workers = min(cpu_count, 8) if cpu_count else 0

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_pin_memory,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True
        )

        if not self.has_logged_training_start:
            logger.info(
                f"Первый запуск обучения нейросети на устройстве: {self.device.type.upper()}. "
                f"AMP включен: {self.scaler.is_enabled()}. "
                f"Ранняя остановка: {'Вкл' if early_stopping_enabled else 'Выкл'}. "
                f"Воркеров DataLoader: {num_workers}."
            )
            self.has_logged_training_start = True

        logger.debug(f"Дообучение на {len(X_tensor)} точках (до {epochs} эпох)...")

        for epoch in range(epochs):
            total_epoch_loss = 0.0
            total_epoch_mae = 0.0  # ИЗМЕНЕНИЕ: Считаем MAE
            batch_count = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    # ИЗМЕНЕНИЕ: Считаем MAE на батче
                    mae = self.mae_criterion(outputs, batch_y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_epoch_loss += loss.item()
                total_epoch_mae += mae.item()  # ИЗМЕНЕНИЕ
                batch_count += 1

            avg_epoch_loss = total_epoch_loss / batch_count if batch_count > 0 else 0
            avg_epoch_mae = total_epoch_mae / batch_count if batch_count > 0 else 0  # ИЗМЕНЕНИЕ

            logger.debug(f"Эпоха {epoch + 1}/{epochs}, MSE: {avg_epoch_loss:.6f}, MAE: {avg_epoch_mae:.6f}")

            # --- Логика ранней остановки (основана на MSE) ---
            if early_stopping_enabled:
                if avg_epoch_loss < best_loss - min_delta:
                    best_loss = avg_epoch_loss
                    best_mae_at_best_mse = avg_epoch_mae  # ИЗМЕНЕНИЕ: Сохраняем MAE лучшей эпохи
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logger.info(f"Ранняя остановка на эпохе {epoch + 1}. Ошибка не улучшалась {patience} эпох.")
                    break

        logger.debug(f"Дообучение завершено.")
        if early_stopping_enabled:
            # ИЗМЕНЕНИЕ: Выводим в лог обе метрики
            logger.info(
                f"Загрузка весов лучшей модели. Ошибки (scaled): MSE={best_loss:.6f}, MAE={best_mae_at_best_mse:.6f}")
            self.model.load_state_dict(best_model_wts)

    def predict(self, X):
        """Предсказание с помощью модели."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                predictions = self.model(X_tensor)
        return predictions.cpu().numpy().flatten()


def model_factory(model_type, model_params, fit_intercept, input_dim=None):
    """
    Фабричная функция, которая возвращает нужную обертку для модели.
    """
    if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
        return SklearnLinearModel(model_type, model_params, fit_intercept)
    elif model_type == 'neural_network':
        if input_dim is None:
            raise ValueError("Для 'neural_network' необходимо указать 'input_dim'.")
        return NeuralNetworkModel(model_params, input_dim)
    else:
        logger.warning(f"Неизвестный тип модели: {model_type}. Будет использована LinearRegression.")
        return SklearnLinearModel('linear', model_params, fit_intercept)

