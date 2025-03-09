import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import logging
from typing import Tuple, List, Any, Optional, Dict

logger = logging.getLogger(__name__)


def load_csv_data(csv_path: str, sep: str, decimal: str) -> np.ndarray:
    """
    Загружает данные из CSV-файла и возвращает массив float32.
    """
    logger.info("Загрузка данных из файла: %s", csv_path)
    try:
        df = pd.read_csv(csv_path, skiprows=1, header=None, sep=sep, decimal=decimal)
        data = df.values.astype(np.float32)
        logger.info("Данные успешно загружены, shape: %s", data.shape)
    except Exception as e:
        logger.error("Ошибка при загрузке или преобразовании данных из %s: %s", csv_path, e)
        raise ValueError(f"Ошибка преобразования данных: {e}")
    return data


def create_scaler(features: np.ndarray, feature_range: Tuple[float, float] = (-1, 1),
                  force_scaling: bool = False, forced_bounds: Optional[Dict[Any, Dict[str, float]]] = None
                  ) -> MinMaxScaler:
    """
    Инициализирует и настраивает MinMaxScaler для заданных признаков.
    Если force_scaling=True и forced_bounds заданы, масштабирует признаки согласно фиксированным границам.
    """
    logger.info("Инициализация нового MinMaxScaler с feature_range=%s", feature_range)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(features)

    if force_scaling and forced_bounds:
        for i, bounds in forced_bounds.items():
            forced_min = bounds.get("min")
            forced_max = bounds.get("max")
            if forced_min is not None and forced_max is not None and i < features.shape[1]:
                scaler.data_min_[i] = forced_min
                scaler.data_max_[i] = forced_max
                frange = scaler.feature_range[1] - scaler.feature_range[0]
                scaler.scale_[i] = frange / (forced_max - forced_min)
                scaler.min_[i] = scaler.feature_range[0] - forced_min * scaler.scale_[i]
                logger.info("Для фичи %d установлены фиксированные границы: min=%s, max=%s", i, forced_min, forced_max)
    return scaler


def scale_data(data: np.ndarray, scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Масштабирует данные с помощью MinMaxScaler.
    Если scaler не предоставлен, создается новый.
    """
    if scaler is None:
        logger.info("Scaler не предоставлен, создание нового.")
        scaler = create_scaler(data)
        data_scaled = scaler.fit_transform(data)
        logger.info("Данные масштабированы, новая шкала создана.")
    else:
        data_scaled = scaler.transform(data)
        logger.info("Данные масштабированы с использованием предоставленного scaler.")
    return data_scaled, scaler


def create_segments(data: np.ndarray, segment_length: int, step: int) -> Tuple[np.ndarray, List[int]]:
    """
    Разбивает данные на сегменты указанной длины с заданным шагом.
    Возвращает массив сегментов и список начальных индексов сегментов.
    """
    n_rows = data.shape[0]
    logger.info("Создание сегментов: общее количество строк: %d", n_rows)
    segments = []
    indices = []
    for start in range(0, n_rows - segment_length + 1, step):
        segments.append(data[start:start + segment_length, :])
        indices.append(start)
    # Если последний сегмент не охватывает конец данных, добавляем его отдельно.
    if indices and indices[-1] != n_rows - segment_length:
        segments.append(data[n_rows - segment_length:n_rows, :])
        indices.append(n_rows - segment_length)
    logger.info("Создано %d сегментов.", len(segments))
    return np.array(segments), indices


def predict_segments(model: torch.jit.ScriptModule, segments: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Выполняет предсказание для каждого сегмента.
    """
    logger.info("Предсказание для %d сегментов.", segments.shape[0])
    model.eval()
    segments_tensor = torch.from_numpy(segments).to(device)
    with torch.no_grad():
        logits = model(segments_tensor)
        preds = torch.argmax(logits, dim=2)
    logger.info("Предсказания выполнены.")
    return preds.cpu().numpy()


def create_sequences(features: np.ndarray, labels: np.ndarray, input_length: int, step: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Создает последовательности из признаков и меток с заданной длиной окна и шагом.
    """
    logger.info("Создание последовательностей с длиной окна: %d и шагом: %d", input_length, step)
    X_seq, y_seq = [], []
    for i in range(0, len(features) - input_length + 1, step):
        window = features[i:i + input_length]
        X_seq.append(window)
        y_seq.append(labels[i:i + input_length])
    X_seq_arr = np.array(X_seq)
    y_seq_arr = np.array(y_seq)
    logger.info("Созданы последовательности, X shape: %s, y shape: %s", X_seq_arr.shape, y_seq_arr.shape)
    return X_seq_arr, y_seq_arr


def load_and_prepare_data(csv_path: str, CONFIG: Dict[str, Any],
                          scaler: Optional[MinMaxScaler] = None,
                          dataset_type: str = "train") -> Tuple[Any, Any, Any, MinMaxScaler, int]:
    """
    Загружает данные из CSV, фильтрует отрицательные метки, масштабирует признаки и создает последовательности.

    Возвращает кортеж с данными для train/val/test, scaler и числом признаков.
    """
    logger.info("Загрузка и подготовка данных для '%s' набора из файла: %s", dataset_type, csv_path)
    csv_settings = CONFIG["CSV_SETTINGS"]
    try:
        df = pd.read_csv(csv_path, skiprows=1, header=None, sep=csv_settings["sep"], decimal=csv_settings["decimal"])
        data = df.values.astype(np.float32)
        logger.info("Данные загружены, shape: %s", data.shape)
    except Exception as e:
        logger.error("Ошибка при загрузке данных: %s", e)
        raise ValueError(f"Ошибка преобразования данных: {e}")

    features = data[:, :-1]
    labels = data[:, -1].astype(np.int64)
    logger.info("Уникальные метки до фильтрации: %s", np.unique(labels))
    valid = labels >= 0
    features, labels = features[valid], labels[valid]
    logger.info("Уникальные метки после фильтрации: %s", np.unique(labels))

    # Масштабирование признаков
    if scaler is None:
        logger.info("Инициализация scaler для признаков")
        scaler = create_scaler(features, feature_range=(-1, 1),
                               force_scaling=CONFIG.get("FORCE_FEATURE_SCALING", False),
                               forced_bounds=CONFIG.get("FORCED_FEATURE_BOUNDS", {}))
        features_final = scaler.transform(features)
    else:
        features_final = scaler.transform(features)
        logger.info("Использован предоставленный scaler для масштабирования признаков")

    # Создание последовательностей
    win_len = CONFIG["INPUT_LENGTH"]
    step = CONFIG["WINDOW_STEP"] if dataset_type.lower() == "train" else CONFIG["INPUT_LENGTH"]
    X_all, y_all = create_sequences(features_final, labels, win_len, step)

    if dataset_type.lower() == "train":
        return (X_all, y_all), (None, None), (None, None), scaler, features.shape[1]
    elif dataset_type.lower() in ["val", "test", "validation"]:
        return (None, None), (X_all, y_all), (X_all, y_all), scaler, features.shape[1]
    else:
        logger.error("Неизвестный тип набора данных: %s", dataset_type)
        raise ValueError(f"Неизвестный тип набора данных: {dataset_type}")
