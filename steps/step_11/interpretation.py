import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from .models import SklearnLinearModel, NeuralNetworkModel


class ModelInterpreter:
    """
    Базовый класс для интерпретаторов.
    Определяет "контракт": должен быть метод для расчета вкладов.
    """

    def calculate_contributions(self, model_wrapper, preprocessor, X_scaled):
        raise NotImplementedError


class LinearModelInterpreter(ModelInterpreter):
    """
    Класс для расчета вклада признаков для линейных моделей.
    Полностью инкапсулирует сложную логику денормализации.
    """

    def calculate_contributions(self, model_wrapper, preprocessor, X_scaled_df):
        """
        Рассчитывает вклад каждого признака в итоговое предсказание.
        - model_wrapper: Обученная обертка модели (SklearnLinearModel).
        - preprocessor: Обученный препроцессор (DataPreprocessor).
        - X_scaled_df: Нормализованные данные для предсказания (Pandas DataFrame).
        """
        model = model_wrapper.model
        y_scaler = preprocessor.target_scaler
        fit_intercept = model_wrapper.fit_intercept

        contributions = pd.DataFrame(index=X_scaled_df.index)

        # Если Y не был нормализован
        if not y_scaler:
            feature_contributions = X_scaled_df.values * model.coef_
            intercept_value = model.intercept_ if fit_intercept else 0.0

        # Если Y был нормализован через StandardScaler
        elif isinstance(y_scaler, StandardScaler):
            denormalized_coef = model.coef_ * y_scaler.scale_
            feature_contributions = X_scaled_df.values * denormalized_coef

            if fit_intercept:
                intercept_value = model.intercept_ * y_scaler.scale_ + y_scaler.mean_
            else:
                intercept_value = y_scaler.mean_

        # Если Y был нормализован через MinMaxScaler
        elif isinstance(y_scaler, MinMaxScaler):
            data_range = y_scaler.data_max_ - y_scaler.data_min_
            denormalized_coef = model.coef_ * data_range
            feature_contributions = X_scaled_df.values * denormalized_coef

            if fit_intercept:
                intercept_value = model.intercept_ * data_range + y_scaler.data_min_
            else:
                intercept_value = y_scaler.data_min_
        else:
            return None  # Неизвестный тип скейлера

        # Приводим intercept к скаляру
        intercept_scalar = intercept_value[0] if isinstance(intercept_value, np.ndarray) else intercept_value

        for i, col in enumerate(X_scaled_df.columns):
            contributions[f'contribution_{col}'] = feature_contributions[:, i]

        contributions['contribution_intercept'] = intercept_scalar

        return contributions


class NeuralNetworkInterpreter(ModelInterpreter):
    """
    Класс для расчета вклада признаков для нейросетевых моделей.
    Использует метод градиентов на входе, чтобы аппроксимировать вклад.
    """

    def calculate_contributions(self, model_wrapper, preprocessor, X_scaled_df):
        """
        Рассчитывает вклад каждого признака в итоговое предсказание.
        """
        model = model_wrapper.model
        y_scaler = preprocessor.target_scaler

        model.eval()

        X_tensor = torch.FloatTensor(X_scaled_df.values)
        X_tensor.requires_grad_(True)

        predictions_scaled_tensor = model(X_tensor)
        predictions_scaled_tensor.sum().backward()

        input_gradients = X_tensor.grad.detach().numpy()

        contributions = pd.DataFrame(index=X_scaled_df.index)

        # Денормализация градиентов
        y_scale = getattr(y_scaler, 'scale_', [1.0])[0] if y_scaler else 1.0
        denormalized_gradients = input_gradients * y_scale

        # Вклад признака ~ X_scaled * d(Y_denorm)/d(X_scaled)
        feature_contributions = X_scaled_df.values * denormalized_gradients

        for i, col in enumerate(X_scaled_df.columns):
            contributions[f'contribution_{col}'] = feature_contributions[:, i]

        # Сумма вкладов признаков
        sum_feature_contributions = contributions.sum(axis=1)

        # Предсказанное значение (денормализованное)
        predicted_y_denorm = model_wrapper.predict(X_scaled_df.values)
        if y_scaler:
            predicted_y_denorm = preprocessor.inverse_transform_target(predicted_y_denorm)

        # Интерсепт вычисляется как разница, чтобы гарантировать
        # что sum(всех_вкладов) == prediction
        contributions['contribution_intercept'] = predicted_y_denorm - sum_feature_contributions

        return contributions


def interpreter_factory(model_wrapper):
    """Фабрика для выбора нужного интерпретатора."""
    if isinstance(model_wrapper, SklearnLinearModel):
        return LinearModelInterpreter()
    elif isinstance(model_wrapper, NeuralNetworkModel):
        return NeuralNetworkInterpreter()
    else:
        return None
