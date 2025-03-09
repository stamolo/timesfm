#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
from data_utils import load_csv_data, scale_data, create_segments, predict_segments, FixedBoundsQuantizer

# ------------------------------
# Конфигурация по умолчанию
# ------------------------------
DEFAULT_CSV_SEP = ";"
DEFAULT_CSV_DECIMAL = ","
DEFAULT_INPUT_CSV = r"H:\Py\fm\test2sps.csv"
DEFAULT_MODEL_PATH = r"G:\models\checkpoints_k\0120\epoch_0003_scripted.pt"
#DEFAULT_MODEL_PATH = r"G:\models\checkpoints_k\0104\epoch_0030_scripted.pt"
#DEFAULT_MODEL_PATH = r"G:\models\checkpoints\0549\epoch_0763_scripted.pt"
#DEFAULT_MODEL_PATH = r"H:\Py\fm\checkpoints\0547\epoch_0006_scripted.pt"
#DEFAULT_MODEL_PATH = r"H:\Py\fm\checkpoints\0546\epoch_0835_scripted.pt"
#DEFAULT_MODEL_PATH = r"H:\Py\fm\checkpoints\0533\epoch_0898_scripted.pt"

# Конфигурация для инференса
CONFIG = {
    "INPUT_LENGTH": 1200,
    "OUTPUT_LENGTH": 1100,
    "NUM_CLASSES": 4,
    "USE_DISCRETE_EMBEDDING": False,
    "N_BINS_PER_FEATURE": {0: 1024, 1: 1024, 2: 1024, 3: 1024},
    "FORCED_FEATURE_BOUNDS": {
        0: {"min": -25, "max": 300},
        1: {"min": -25, "max": 40},
        2: {"min": -25, "max": 300},
        3: {"min": -25, "max": 280},
    },
    "CSV_SETTINGS": {"sep": DEFAULT_CSV_SEP, "decimal": DEFAULT_CSV_DECIMAL},
}

def process_csv(model_path: str, csv_path: str, output_csv: str,
                segment_length: int, output_length: int,
                sep: str, decimal: str,
                scaler_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    if scaler_path is None:
        scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pkl")
    data = load_csv_data(csv_path, sep, decimal)
    print(f"Загружено строк: {data.shape[0]}, признаков: {data.shape[1]}")
    if CONFIG.get("USE_DISCRETE_EMBEDDING", False):
        quantizer = FixedBoundsQuantizer(CONFIG["FORCED_FEATURE_BOUNDS"], CONFIG["N_BINS_PER_FEATURE"])
        data_prepared = quantizer.transform(data)
        print("Данные квантованы.")
    else:
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print("Скейлер загружен.")
        else:
            scaler = None
        data_prepared, scaler = scale_data(data, scaler)
        print("Данные масштабированы.")
    segments, seg_start_indices = create_segments(data_prepared, segment_length, step=output_length)
    print(f"Сформировано {segments.shape[0]} сегментов.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    model = torch.jit.load(model_path, map_location=device)
    print("Модель загружена.")
    preds_segments = predict_segments(model, segments, device)
    center_offset = (segment_length - output_length) // 2
    pred_column = np.full((data.shape[0],), np.nan)
    for idx, start in enumerate(seg_start_indices):
        pred_column[start + center_offset: start + center_offset + output_length] = preds_segments[idx]
    print("Предсказания объединены.")
    n_features = data.shape[1]
    import pandas as pd
    col_names = [f"feature_{i}" for i in range(n_features)]
    df_out = pd.DataFrame(data, columns=col_names)
    df_out["predicted_label"] = pred_column
    df_out.to_csv(output_csv, index=False, sep=sep, decimal=decimal)
    print(f"Результаты сохранены в {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для разметки данных с помощью предобученной модели.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Путь к модели (TorchScript)")
    parser.add_argument("--csv", type=str, default=DEFAULT_INPUT_CSV, help="Путь к входному CSV")
    parser.add_argument("--output", type=str, required=True, help="Путь к выходному CSV")
    parser.add_argument("--segment_length", type=int, default=CONFIG["INPUT_LENGTH"], help="Длина сегмента")
    parser.add_argument("--output_length", type=int, default=CONFIG["OUTPUT_LENGTH"], help="Длина центрального участка")
    parser.add_argument("--sep", type=str, default=DEFAULT_CSV_SEP, help="Разделитель CSV")
    parser.add_argument("--decimal", type=str, default=DEFAULT_CSV_DECIMAL, help="Десятичный разделитель")
    parser.add_argument("--scaler", type=str, default=None, help="Путь к scaler (pickle)")
    args = parser.parse_args()
    process_csv(model_path=args.model,
                csv_path=args.csv,
                output_csv=args.output,
                segment_length=args.segment_length,
                output_length=args.output_length,
                sep=args.sep,
                decimal=args.decimal,
                scaler_path=args.scaler)
