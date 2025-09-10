#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import pickle
import logging
from utils.data_utils import load_csv_data, scale_data, create_segments, predict_segments

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------
# Конфигурация по умолчанию
# ------------------------------
DEFAULT_CSV_SEP = ";"
DEFAULT_CSV_DECIMAL = ","

DEFAULT_INPUT_CSV = r"C:\Users\Александр\PycharmProjects\timesfm\dataset\4\v4_t_512c5869be1c40d28a83c4a0a2a5e416.csv"
#DEFAULT_INPUT_CSV = r"H:\Py\fm\test_sps\test2sps_1200_1.csv"

DEFAULT_MODEL_PATH = r"D:\models\checkpoints_k\best_model.pt"
#DEFAULT_MODEL_PATH = r"G:\models\epoch_0762_scripted.pt"

CONFIG = {
    "INPUT_LENGTH": 1200,
    "OUTPUT_LENGTH": 1100,
    "NUM_CLASSES": 4,
    "CSV_SETTINGS": {"sep": DEFAULT_CSV_SEP, "decimal": DEFAULT_CSV_DECIMAL},
}


def process_csv(model_path: str, csv_path: str, output_csv: str,
                segment_length: int, output_length: int,
                sep: str, decimal: str,
                scaler_path: str = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Используемое устройство: %s", device)
    if scaler_path is None:
        scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pkl")
    data = load_csv_data(csv_path, sep, decimal)
    logger.info("Загружено строк: %d, признаков: %d", data.shape[0], data.shape[1])

    # Всегда используем масштабирование через scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Скейлер загружен из: %s", scaler_path)
    else:
        scaler = None
    data_prepared, scaler = scale_data(data, scaler)
    logger.info("Данные масштабированы.")

    segments, seg_start_indices = create_segments(data_prepared, segment_length, step=output_length)
    logger.info("Сформировано %d сегментов.", segments.shape[0])

    if not os.path.exists(model_path):
        logger.error("Модель не найдена: %s", model_path)
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    model = torch.jit.load(model_path, map_location=device)
    logger.info("Модель загружена: %s", model_path)

    preds_segments = predict_segments(model, segments, device)
    center_offset = (segment_length - output_length) // 2
    pred_column = np.full((data.shape[0],), np.nan)
    for idx, start in enumerate(seg_start_indices):
        pred_column[start + center_offset: start + center_offset + output_length] = preds_segments[idx]
    logger.info("Предсказания объединены.")

    n_features = data.shape[1]
    import pandas as pd
    col_names = [f"feature_{i}" for i in range(n_features)]
    df_out = pd.DataFrame(data, columns=col_names)
    df_out["predicted_label"] = pred_column
    df_out.to_csv(output_csv, index=False, sep=sep, decimal=decimal)
    logger.info("Результаты сохранены в %s", output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для разметки данных с помощью предобученной модели.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Путь к модели (TorchScript)")
    parser.add_argument("--csv", type=str, default=DEFAULT_INPUT_CSV, help="Путь к входному CSV")
    parser.add_argument("--output", type=str, required=True, help="Путь к выходному файлу или директории")
    parser.add_argument("--segment_length", type=int, default=CONFIG["INPUT_LENGTH"], help="Длина сегмента")
    parser.add_argument("--output_length", type=int, default=CONFIG["OUTPUT_LENGTH"], help="Длина центрального участка")
    parser.add_argument("--sep", type=str, default=DEFAULT_CSV_SEP, help="Разделитель CSV")
    parser.add_argument("--decimal", type=str, default=DEFAULT_CSV_DECIMAL, help="Десятичный разделитель")
    parser.add_argument("--scaler", type=str, default=None, help="Путь к scaler (pickle)")
    args = parser.parse_args()

    # --- FIX START ---
    # Проверяем, является ли указанный путь --output директорией.
    # Если да, то создаём имя файла внутри этой директории.
    output_path = args.output
    if os.path.isdir(output_path):
        # Берём имя входного файла (например, "test1sps.csv")
        base_name = os.path.basename(args.csv)
        # Создаём новое имя (например, "test1sps_predicted.csv")
        file_name, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_name}_predicted{file_ext}"
        # Соединяем путь к директории и новое имя файла
        output_path = os.path.join(output_path, output_filename)
        logger.info(f"Выходной путь является директорией. Результат будет сохранен в: {output_path}")
    # --- FIX END ---

    process_csv(model_path=args.model,
                csv_path=args.csv,
                output_csv=output_path,  # Используем исправленный путь
                segment_length=args.segment_length,
                output_length=args.output_length,
                sep=args.sep,
                decimal=args.decimal,
                scaler_path=args.scaler)
