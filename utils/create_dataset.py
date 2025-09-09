import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.data_utils import load_and_prepare_data
import logging

logger = logging.getLogger(__name__)


def create_datasets(config: dict) -> tuple:
    logger.info("Создание датасетов с конфигурацией: %s", config)
    # Загрузка данных для обучающего набора
    (X_train, y_train), _, _, scaler, num_features = load_and_prepare_data(
        config["TRAIN_CSV"], config, dataset_type="train"
    )
    logger.info("Обучающие данные: X shape %s, y shape %s", X_train.shape, y_train.shape)

    # Загрузка данных для валидации и теста
    _, (X_val, y_val), (X_test, y_test), scaler, _ = load_and_prepare_data(
        config["VAL_TEST_CSV"], config, scaler=scaler, dataset_type="val"
    )
    logger.info("Валидационные данные: X shape %s, y shape %s", X_val.shape, y_val.shape)
    logger.info("Тестовые данные: X shape %s, y shape %s", X_test.shape, y_test.shape)

    # Приведение данных к нужным типам
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t = torch.LongTensor(y_val)
    y_test_t = torch.LongTensor(y_test)

    # Создание TensorDataset'ов
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Создание DataLoader'ов
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

    logger.info("Датасеты и DataLoader'ы созданы")
    return train_loader, val_loader, test_loader, scaler, num_features, train_dataset
