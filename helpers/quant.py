import pandas as pd
import numpy as np
import os

# Конфигурация с параметрами и настройками
CONFIG = {
    "NUM_QUANT": 16,  # Используется только если файл с границами не задан.
    "COLUMN_INDEX": -1,  # Индекс столбца для квантования (например, -1 для последнего, 0 для первого)
    "CSV_SETTINGS": {
        "sep": ";",
        "decimal": ","
    },
    "INPUT_FILE": "test_kl_swapped_ln.csv",                # Входной CSV файл
    "OUTPUT_FILE": "test_kl_swapped_quantized.csv",      # Выходной CSV файл с квантованными данными
    "BOUNDARIES_FILE": "quant_boundaries.csv",  # Файл для сохранения границ квантования (если вычисляем)
    "BOUNDARIES_INPUT_FILE": "quant_boundaries.csv"                 # Файл с границами квантования для идентичного квантования (если указан)
}

def main():
    # Извлечение настроек из конфигурационного словаря
    col_index = CONFIG["COLUMN_INDEX"]
    csv_settings = CONFIG["CSV_SETTINGS"]
    input_file = CONFIG["INPUT_FILE"]
    output_file = CONFIG["OUTPUT_FILE"]
    boundaries_file = CONFIG["BOUNDARIES_FILE"]
    boundaries_input_file = CONFIG["BOUNDARIES_INPUT_FILE"]

    # Чтение исходного файла
    df = pd.read_csv(input_file, **csv_settings)

    # Определение столбца для квантования по индексу
    try:
        col_to_quant = df.columns[col_index]
    except IndexError:
        raise ValueError(f"Некорректный индекс столбца: {col_index}")

    values = df[col_to_quant]

    # Если задан входной файл с границами, загрузим их
    if boundaries_input_file and os.path.exists(boundaries_input_file):
        boundaries_df = pd.read_csv(boundaries_input_file, **csv_settings)
        # Предполагается, что в файле есть столбец 'boundary'
        boundaries = boundaries_df["boundary"].to_numpy()
        num_quant = len(boundaries) - 1  # Автоматическое определение количества квантов
    else:
        # Если файла с границами нет, используем значение из CONFIG
        num_quant = CONFIG["NUM_QUANT"]
        # Вычисление квантильных границ: рассчитываем значения на процентилях от 0 до 100 с шагом 100/num_quant
        quantile_values = [np.percentile(values, i * 100 / num_quant) for i in range(num_quant + 1)]
        boundaries = np.array(quantile_values)
        # Сохраним вычисленные границы в файл для последующего использования
        boundaries_df = pd.DataFrame({
            "quant": list(range(num_quant + 1)),
            "boundary": quantile_values
        })
        boundaries_df.to_csv(boundaries_file, index=False, **csv_settings)

    # Квантование значений: для каждого значения определяем номер интервала, используя np.searchsorted
    quantized = np.searchsorted(boundaries, values, side='right') - 1

    # Корректировка для значений, равных максимальной границе
    quantized[quantized == num_quant] = num_quant - 1

    # Замена значений в выбранном столбце на квантованные
    df[col_to_quant] = quantized

    # Запись результирующего файла с квантованными данными
    df.to_csv(output_file, index=False, **csv_settings)

if __name__ == "__main__":
    main()
