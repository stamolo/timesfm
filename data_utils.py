import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import logging

logger = logging.getLogger(__name__)


def load_csv_data(csv_path: str, sep: str, decimal: str) -> np.ndarray:
    logger.info("Загрузка данных из файла: %s", csv_path)
    df = pd.read_csv(csv_path, skiprows=1, header=None, sep=sep, decimal=decimal)
    try:
        data = df.values.astype(np.float32)
        logger.info("Данные успешно загружены, shape: %s", data.shape)
    except Exception as e:
        logger.error("Ошибка преобразования данных из %s: %s", csv_path, e)
        raise ValueError(f"Ошибка преобразования данных: {e}")
    return data


def scale_data(data: np.ndarray, scaler: MinMaxScaler = None) -> tuple[np.ndarray, MinMaxScaler]:
    if scaler is None:
        logger.info("Инициализация нового MinMaxScaler")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)
        logger.info("Данные масштабированы, новая шкала создана")
    else:
        data_scaled = scaler.transform(data)
        logger.info("Данные масштабированы с использованием предоставленного scaler")
    return data_scaled, scaler


def create_segments(data: np.ndarray, segment_length: int, step: int) -> tuple[np.ndarray, list[int]]:
    n_rows = data.shape[0]
    logger.info("Создание сегментов из данных с количеством строк: %d", n_rows)
    segments = []
    indices = []
    for start in range(0, n_rows - segment_length + 1, step):
        segments.append(data[start:start + segment_length, :])
        indices.append(start)
    if indices and indices[-1] != n_rows - segment_length:
        segments.append(data[n_rows - segment_length:n_rows, :])
        indices.append(n_rows - segment_length)
    logger.info("Создано %d сегментов", len(segments))
    return np.array(segments), indices


def predict_segments(model: torch.jit.ScriptModule, segments: np.ndarray, device: torch.device) -> np.ndarray:
    logger.info("Предсказание сегментов, количество сегментов: %d", segments.shape[0])
    model.eval()
    segments_tensor = torch.from_numpy(segments).to(device)
    with torch.no_grad():
        logits = model(segments_tensor)
        preds = torch.argmax(logits, dim=2)
    logger.info("Предсказания выполнены")
    return preds.cpu().numpy()


def load_and_prepare_data(csv_path: str, CONFIG: dict, scaler: MinMaxScaler = None,
                          dataset_type: str = "train") -> tuple:
    logger.info("Загрузка и подготовка данных для '%s' набора из файла: %s", dataset_type, csv_path)
    csv_settings = CONFIG["CSV_SETTINGS"]
    df = pd.read_csv(csv_path, skiprows=1, header=None, sep=csv_settings["sep"], decimal=csv_settings["decimal"])
    try:
        data = df.values.astype(np.float32)
        logger.info("Данные загружены, shape: %s", data.shape)
    except Exception as e:
        logger.error("Ошибка преобразования данных: %s", e)
        raise ValueError(f"Ошибка преобразования данных: {e}")
    features = data[:, :-1]
    labels = data[:, -1].astype(np.int64)
    logger.info("Уникальные метки до фильтрации: %s", np.unique(labels))
    valid = labels >= 0
    features, labels = features[valid], labels[valid]
    logger.info("Уникальные метки после фильтрации: %s", np.unique(labels))

    if CONFIG.get("USE_DISCRETE_EMBEDDING", False):
        forced_bounds = CONFIG.get("FORCED_FEATURE_BOUNDS", None)
        if forced_bounds is None:
            logger.error("Не заданы фиксированные границы для квантования!")
            raise ValueError("Не заданы фиксированные границы для квантования!")
        quantizer = FixedBoundsQuantizer(forced_bounds, CONFIG.get("N_BINS_PER_FEATURE"))
        features_quantized = quantizer.transform(features)
        features_final = features_quantized
    else:
        if scaler is None:
            logger.info("Инициализация нового MinMaxScaler для признаков")
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(features)
            if CONFIG.get("FORCE_FEATURE_SCALING", False):
                forced_bounds = CONFIG.get("FORCED_FEATURE_BOUNDS", {})
                for i, bounds in forced_bounds.items():
                    forced_min = bounds.get("min")
                    forced_max = bounds.get("max")
                    if forced_min is not None and forced_max is not None and i < features.shape[1]:
                        scaler.data_min_[i] = forced_min
                        scaler.data_max_[i] = forced_max
                        feature_range = scaler.feature_range[1] - scaler.feature_range[0]
                        scaler.scale_[i] = feature_range / (forced_max - forced_min)
                        scaler.min_[i] = scaler.feature_range[0] - forced_min * scaler.scale_[i]
                        logger.info("Для фичи %d установлены min=%s и max=%s.", i, forced_min, forced_max)
            features_final = scaler.transform(features)
        else:
            features_final = scaler.transform(features)

    def create_sequences(f: np.ndarray, l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        win_len = CONFIG["INPUT_LENGTH"]
        step = CONFIG["WINDOW_STEP"] if dataset_type == "train" else CONFIG["INPUT_LENGTH"]
        for i in range(0, len(f) - win_len + 1, step):
            window = f[i:i + win_len]
            X_seq.append(window)
            y_seq.append(l[i:i + win_len])
        return np.array(X_seq), np.array(y_seq)

    X_all, y_all = create_sequences(features_final, labels)
    logger.info("Созданы последовательности, X shape: %s, y shape: %s", X_all.shape, y_all.shape)

    if dataset_type.lower() == "train":
        return (X_all, y_all), (None, None), (None, None), scaler, features.shape[1]
    elif dataset_type.lower() in ["val", "test", "validation"]:
        return (None, None), (X_all, y_all), (X_all, y_all), scaler, features.shape[1]
    else:
        logger.error("Неизвестный тип набора данных: %s", dataset_type)
        raise ValueError(f"Неизвестный тип набора данных: {dataset_type}")


class FixedBoundsQuantizer:
    def __init__(self, forced_bounds: dict, n_bins_per_feature: dict):
        self.forced_bounds = forced_bounds
        self.n_bins_per_feature = n_bins_per_feature
        self.bin_edges = {}
        for i, bounds in forced_bounds.items():
            n_bins = n_bins_per_feature.get(i, 256)
            self.bin_edges[i] = np.linspace(bounds['min'], bounds['max'], n_bins + 1)
        logger.info("Инициализирован FixedBoundsQuantizer с %d фичами", len(self.bin_edges))

    def transform(self, X: np.ndarray) -> np.ndarray:
        logger.info("Квантование признаков, X shape: %s", X.shape)
        quantized = []
        num_features = X.shape[1]
        for i in range(num_features):
            if i in self.bin_edges:
                edges = self.bin_edges[i]
                tokens = np.digitize(X[:, i], edges) - 1
                n_bins = self.n_bins_per_feature.get(i, 256)
                tokens = np.clip(tokens, 0, n_bins - 1)
            else:
                tokens = np.zeros(X.shape[0], dtype=np.int64)
            quantized.append(tokens.reshape(-1, 1))
        result = np.concatenate(quantized, axis=1)
        logger.info("Квантование завершено, result shape: %s", result.shape)
        return result
