import torch
from torch.utils.data import TensorDataset, DataLoader
from data_utils import load_and_prepare_data


def create_datasets(config):
    """
    Загружает данные, подготавливает датасеты и DataLoader'ы,
    а также возвращает скейлер, число признаков и обучающий датасет.
    """
    # Загрузка данных (обучающая выборка)
    (X_train, y_train), _, _, scaler, num_features = load_and_prepare_data(
        config["TRAIN_CSV"], config, dataset_type="train"
    )
    # Загрузка данных (валидация и тест)
    _, (X_val, y_val), (X_test, y_test), _, _ = load_and_prepare_data(
        config["VAL_TEST_CSV"], config, scaler=scaler, dataset_type="val"
    )

    # Приведение данных к нужным типам (в данном случае всегда FloatTensor для X и LongTensor для y)
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

    return train_loader, val_loader, test_loader, scaler, num_features, train_dataset
