from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import logging

logger = logging.getLogger(__name__)


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


# В будущем здесь можно добавить другие обертки, например:
# class XGBoostModel(ModelWrapper): ...
# class NeuralNetworkModel(ModelWrapper): ...

def model_factory(model_type, model_params, fit_intercept):
    """
    Фабричная функция, которая возвращает нужную обертку для модели.
    """
    if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
        return SklearnLinearModel(model_type, model_params, fit_intercept)
    # elif model_type == 'xgboost':
    #     return XGBoostModel(...)
    else:
        logger.warning(f"Неизвестный тип модели: {model_type}. Будет использована LinearRegression.")
        return SklearnLinearModel('linear', model_params, fit_intercept)
