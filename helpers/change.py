import csv
import os

# Конфигурация: индексы столбцов для обмена и параметры CSV.
CONFIG = {
    "swap_columns": [0, 4],  # меняем столбцы с индексами 0 и 4
    "csv_settings": {
        "delimiter": ";",  # разделитель столбцов — точка с запятой
        "quotechar": '"',  # символ кавычки
        "lineterminator": "\n",  # терминатор строки
        "doublequote": True,  # удвоение кавычек внутри поля
        "skipinitialspace": False,  # не пропускать пробелы после разделителя
        "quoting": csv.QUOTE_MINIMAL  # минимальное цитирование
    }
}


def swap_columns_in_csv(input_csv, output_csv, col1, col2, csv_settings):
    """
    Читает CSV файл с указанными настройками, меняет местами столбцы с индексами col1 и col2,
    и записывает результат в выходной файл.
    """
    # Чтение CSV файла с явными настройками
    with open(input_csv, 'r', encoding='utf-8', newline='') as csvfile:
        reader = csv.reader(csvfile, **csv_settings)
        rows = list(reader)

    if not rows:
        print("CSV файл пустой.")
        return

    # Обмен столбцов в каждой строке
    for i, row in enumerate(rows):
        if len(row) <= max(col1, col2):
            raise IndexError(f"В строке {i} отсутствует один из указанных столбцов")
        row[col1], row[col2] = row[col2], row[col1]

    # Запись результата с теми же настройками CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, **csv_settings)
        writer.writerows(rows)


def main():
    input_file = "train_kl.csv"
    # Формирование имени выходного файла (добавляется суффикс _swapped)
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_swapped{ext}"

    try:
        col1, col2 = CONFIG["swap_columns"]
        csv_settings = CONFIG["csv_settings"]
        swap_columns_in_csv(input_file, output_file, col1, col2, csv_settings)
        print(f"Обработка завершена успешно. Результат сохранён в файле: {output_file}")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
