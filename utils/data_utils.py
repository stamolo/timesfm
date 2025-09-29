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
                denominator = (forced_max - forced_min)
                if denominator == 0:
                    logger.warning(
                        f"ДИАГНОСТИКА: Деление на ноль для фичи {i}! min=max={forced_max}. Использую epsilon.")
                    denominator += 1e-8
                scaler.scale_[i] = frange / denominator
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
        output = model(segments_tensor)
    logger.info("Предсказания выполнены.")
    return output.squeeze().cpu().numpy()


def create_sequences(features: np.ndarray, labels: np.ndarray, input_length: int, step: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Создает последовательности из признаков и меток с заданной длиной окна и шагом.
    """
    logger.info("Создание последовательностей с длиной окна: %d и шагом: %d", input_length, step)
    X_seq, y_seq = [], []
    for i in range(0, len(features) - input_length + 1, step):
        window_x = features[i:i + input_length]
        X_seq.append(window_x)
        window_y = labels[i:i + input_length]
        y_seq.append(window_y)
    X_seq_arr = np.array(X_seq)
    y_seq_arr = np.array(y_seq)
    logger.info("Созданы последовательности, X shape: %s, y shape: %s", X_seq_arr.shape, y_seq_arr.shape)
    return X_seq_arr, y_seq_arr


def load_and_prepare_data(csv_path: str, CONFIG: Dict[str, Any],
                          scaler: Optional[MinMaxScaler] = None,
                          dataset_type: str = "train") -> Tuple[Any, Any, Any, MinMaxScaler, int]:
    """
    Загружает данные из CSV, разделяет на признаки и метки, масштабирует и создает последовательности.
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

    if np.isnan(data).any():
        logger.error("!!! ДИАГНОСТИКА: Обнаружены NaN в данных СРАЗУ ПОСЛЕ загрузки из CSV-файла '%s'.", csv_path)
        raise ValueError(f"NaN в исходном файле {csv_path}")

    # Разделение на признаки и метки
    target_col_idx = CONFIG.get("TARGET_COLUMN", -1)
    if target_col_idx < 0:
        target_col_idx = data.shape[1] + target_col_idx

    if not 0 <= target_col_idx < data.shape[1]:
        raise ValueError(
            f"TARGET_COLUMN индекс {target_col_idx} вне допустимого диапазона для данных с {data.shape[1]} столбцами.")

    logger.info("Целевой столбец (TARGET_COLUMN) имеет индекс: %d", target_col_idx)
    labels = data[:, target_col_idx].astype(np.float32)

    # === ИЗМЕНЕНИЕ: Выбор признаков на основе FEATURE_COLUMNS ===
    feature_columns = CONFIG.get("FEATURE_COLUMNS")
    if feature_columns:  # Если список не пустой и не None
        logger.info(f"Используются указанные столбцы для признаков: {feature_columns}")
        # Проверяем, не пересекается ли выбор с целевым столбцом
        if target_col_idx in feature_columns:
            logger.warning(
                f"Целевой столбец ({target_col_idx}) также указан в FEATURE_COLUMNS. Он будет исключен из признаков.")
            feature_columns = [col for col in feature_columns if col != target_col_idx]

        # Проверяем, что все индексы в пределах допустимого
        max_col_index = data.shape[1] - 1
        if any(col > max_col_index for col in feature_columns):
            raise ValueError(
                f"Один из индексов в FEATURE_COLUMNS превышает максимальный индекс столбца ({max_col_index})")

        features = data[:, feature_columns]
    else:  # Если FEATURE_COLUMNS не задан, используем все, кроме целевого
        logger.info("FEATURE_COLUMNS не указан. Используются все столбцы, кроме целевого.")
        feature_indices = [i for i in range(data.shape[1]) if i != target_col_idx]
        features = data[:, feature_indices]
    # ========================================================

    # Фильтрация по меткам (может быть полезна для удаления аномальных/пропущенных таргетных значений)
    logger.info("Фильтрация по меткам (значения >= 0). Если это не требуется, закомментируйте.")
    valid = labels >= 0
    features, labels = features[valid], labels[valid]

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

    if np.isnan(features_final).any():
        logger.error("!!! ДИАГНОСТИКА: Обнаружены NaN в данных ПОСЛЕ масштабирования.")
        raise ValueError("NaN появились после масштабирования данных.")

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

