import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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


def interpreter_factory(model_wrapper):
    """Фабрика для выбора нужного интерпретатора."""
    # Проверяем, что это наш класс-обертка для sklearn моделей
    if isinstance(model_wrapper, SklearnLinearModel):
        return LinearModelInterpreter()
    # elif isinstance(model_wrapper, XGBoostModel):
    #     return SHAPInterpreter()
    else:
        # Возвращаем "пустой" интерпретатор, если модель не поддерживается
        return None
