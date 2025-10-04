import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
import joblib
import os

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Класс, отвечающий за всю предварительную обработку данных для модели:
    - Нормализация признаков (X) и целевой переменной (Y).
    - Хранение скейлеров для обратного преобразования.
    """

    def __init__(self, normalization_type):
        self.normalization_type = normalization_type
        self.feature_scaler = None
        self.target_scaler = None

    def fit_transform(self, X_train, y_train):
        """Обучает скейлеры и преобразует обучающие данные."""
        if self.normalization_type == 'min_max':
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        elif self.normalization_type == 'z_score':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:  # 'none'
            return X_train, y_train

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        y_train_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        return X_train_scaled, y_train_scaled

    def transform_features(self, X):
        """Преобразует новые признаки, используя обученный скейлер."""
        if self.feature_scaler:
            return self.feature_scaler.transform(X)
        return X

    def inverse_transform_target(self, y_scaled):
        """Выполняет обратное преобразование для предсказанных значений."""
        if self.target_scaler:
            return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        return y_scaled

    def save(self, filepath):
        """Сохраняет объект препроцессора в файл."""
        try:
            joblib.dump(self, filepath)
            logger.info(f"Препроцессор успешно сохранен в {filepath}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении препроцессора в {filepath}: {e}")

    @staticmethod
    def load(filepath):
        """Загружает объект препроцессора из файла."""
        if not os.path.exists(filepath):
            logger.warning(f"Файл препроцессора не найден: {filepath}. Будет создан новый.")
            return None
        try:
            preprocessor = joblib.load(filepath)
            logger.info(f"Препроцессор успешно загружен из {filepath}")
            return preprocessor
        except Exception as e:
            logger.error(f"Ошибка при загрузке препроцессора из {filepath}: {e}", exc_info=True)
            return None


def balance_training_data(df, balancing_col, max_stationary_percent):
    """
    Выполняет балансировку обучающей выборки, уменьшая долю статичных точек.
    """
    moving_data = df[df[balancing_col] != 0]
    stationary_data = df[df[balancing_col] == 0]
    n_moving = len(moving_data)
    n_stationary = len(stationary_data)
    n_total = n_moving + n_stationary

    if n_total == 0 or n_moving == 0:
        return df  # Нечего балансировать

    current_stationary_percent = (n_stationary / n_total) * 100
    if current_stationary_percent <= max_stationary_percent:
        return df  # Балансировка не требуется

    # Рассчитываем, сколько статичных точек нужно оставить
    if (100 - max_stationary_percent) > 0:
        target_n_stationary = int((max_stationary_percent * n_moving) / (100 - max_stationary_percent))
        target_n_stationary = min(target_n_stationary, n_stationary)  # Не можем взять больше, чем есть

        if target_n_stationary > 0:
            stationary_data_sampled = stationary_data.sample(n=target_n_stationary, random_state=42)
            balanced_df = pd.concat([moving_data, stationary_data_sampled])
            logger.debug(f"Балансировка: {n_total} -> {len(balanced_df)} точек.")
            return balanced_df
        else:  # Если нужно оставить 0 статичных точек
            return moving_data

    return df  # Возвращаем исходные данные, если расчет не удался
