import pandas as pd
import numpy as np
import os

# Пример конфигурации
config = {
    "INPUT_CSV": "train_kl_swapped.csv",
    "CSV_SETTINGS": {"sep": ";", "decimal": ","},
    # Логарифмирование: ключ – исходный номер столбца, значение True - применять ln
    "LOG_COLUMNS": {0: False, 1: False, 2: False, 3: False, 4: False},
    # Дифференциал: ключ – исходный номер столбца, значение True - рассчитывать diff
    "DIFF_COLUMNS": {0: False, 1: True, 2: False, 3: False, 4: False},
    # Список номеров исходных столбцов для удаления (те номера, что были в исходном файле)
    "DELETE_COLUMNS": [],
    #     # Значение для замены недопустимых значенийпри логарифмировании
    "REPLACE_INVALID": 0,
    # Значение для замены недопустимых результатов при вычислении дифференциала
    "DIFF_REPLACE_INVALID": 0
}

def read_data(config):
    """Читает CSV-файл с пропуском первой строки (заголовок)"""
    csv_settings = config.get("CSV_SETTINGS", {}).copy()
    csv_settings["skiprows"] = 1
    try:
        df = pd.read_csv(config["INPUT_CSV"], header=None, **csv_settings)
        return df
    except Exception as e:
        print(f"Ошибка при чтении файла {config['INPUT_CSV']}: {e}")
        raise

def transform_ln_value(value, replacement):
    """
    Преобразует значение с использованием натурального логарифма.
    Если значение <= 0 или не числовое, возвращается replacement.
    """
    try:
        num = float(value)
        if num > 0:
            return np.log(num)
        else:
            return replacement
    except Exception:
        return replacement

def transform_ln(df, config):
    """Применяет логарифмирование к указанным столбцам"""
    for col_index, apply_log in config.get("LOG_COLUMNS", {}).items():
        if apply_log:
            if col_index in df.columns:
                df[col_index] = df[col_index].apply(lambda x: transform_ln_value(x, config["REPLACE_INVALID"]))
            else:
                print(f"Предупреждение: столбец {col_index} не найден для логарифмирования.")
    return df

def add_diff_columns(df, config):
    """
    Вычисляет дифференциал для указанных столбцов и вставляет новый столбец сразу после исходного.
    Новый столбец получает имя вида 'номер_diff'.
    Обрабатываем столбцы в порядке убывания исходного номера, чтобы не нарушалась нумерация.
    """
    for col_index in sorted(config.get("DIFF_COLUMNS", {}), reverse=True):
        if config["DIFF_COLUMNS"].get(col_index, False):
            if col_index in df.columns:
                diff_replacement = config.get("DIFF_REPLACE_INVALID")
                diff_series = df[col_index].diff().fillna(diff_replacement)
                new_col_name = f"{col_index}_diff"
                df.insert(col_index + 1, new_col_name, diff_series)
            else:
                print(f"Предупреждение: столбец {col_index} не найден для вычисления дифференциала.")
    return df

def delete_original_columns(df, config):
    """
    Удаляет исходные столбцы по их номерам из исходного файла.
    Новые столбцы с дифференциалом (с именами вида 'номер_diff') остаются.
    """
    delete_cols = config.get("DELETE_COLUMNS", [])
    df.drop(columns=delete_cols, inplace=True, errors='ignore')
    return df

def write_data(df, config):
    """Сохраняет итоговый DataFrame в новый CSV-файл с добавлением суффикса '_ln' к имени файла."""
    base, ext = os.path.splitext(config["INPUT_CSV"])
    output_filename = base + "_ln" + ext
    try:
        save_settings = config.get("CSV_SETTINGS", {}).copy()
        save_settings.pop("skiprows", None)
        df.to_csv(output_filename, header=False, index=False, **save_settings)
        print(f"Файл успешно сохранен как {output_filename}")
    except Exception as e:
        print(f"Ошибка при записи файла {output_filename}: {e}")
        raise

def pipeline(config):
    """Пайплайн обработки данных: последовательное выполнение всех этапов"""
    df = read_data(config)
    df = transform_ln(df, config)
    df = add_diff_columns(df, config)
    df = delete_original_columns(df, config)
    write_data(df, config)

if __name__ == "__main__":
    pipeline(config)
