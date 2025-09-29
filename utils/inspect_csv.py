import pandas as pd
import argparse
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_problematic_rows(csv_path: str, sep: str, decimal: str):
    """
    Читает CSV-файл, находит строки, которые приводят к значениям NaN, и выводит их.
    """
    if not os.path.exists(csv_path):
        logger.error(f"Файл не найден по пути: {csv_path}")
        return

    logger.info(f"Начинаю проверку файла: {csv_path}")

    try:
        # Шаг 1: Читаем файл как "сырой" текст, чтобы сохранить оригинальное содержимое для вывода.
        df = pd.read_csv(
            csv_path,
            skiprows=1,
            header=None,
            sep=sep,
            dtype=str
        )
        # Заполняем полностью пустые ячейки (например из-за ';;') пустой строкой, чтобы избежать ошибок
        df.fillna('', inplace=True)

        # === ИСПРАВЛЕНИЕ v2: Более надежный метод ===
        # Создаем копию для преобразований
        df_to_convert = df.copy()

        # Явно заменяем десятичный разделитель на точку во всех столбцах
        for col in df_to_convert.columns:
            if decimal != '.':
                df_to_convert[col] = df_to_convert[col].str.replace(decimal, '.', regex=False)

        # Пытаемся преобразовать все в числа. Ошибки станут NaN.
        df_numeric = df_to_convert.apply(pd.to_numeric, errors='coerce')

        # Находим маску для строк, где есть хотя бы одно значение NaN
        problematic_rows_mask = df_numeric.isnull().any(axis=1)

        if problematic_rows_mask.any():
            # Получаем индексы этих строк
            problematic_indices = problematic_rows_mask[problematic_rows_mask].index
            logger.warning(f"Найдено {len(problematic_indices)} проблемных строк.")

            print("\n" + "=" * 25)
            print(" Найдены проблемные строки ")
            print("=" * 25)
            # Выводим информацию по каждой найденной строке, используя оригинальный df
            for idx in problematic_indices:
                # Номер строки в файле = индекс pandas + 2
                # (+1, так как мы пропустили заголовок, и +1, так как индекс начинается с 0)
                line_number = idx + 2
                raw_content = df.iloc[idx].values
                print(f"\n[!] Строка в файле: {line_number}")
                print(f"    Содержимое: '{sep.join(map(str, raw_content))}'")
            print("\n" + "=" * 25 + "\n")
            logger.info(
                "Проверьте эти строки в вашем CSV файле. Возможные причины: пропущенные значения (например, два разделителя подряд ';;'), текстовые символы вместо чисел.")

        else:
            logger.info("Проблемных строк (NaN) не найдено. Файл выглядит корректным.")

    except Exception as e:
        logger.error(f"Произошла ошибка при чтении или анализе файла: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инструмент для поиска и отображения строк с NaN в CSV файлах.")
    parser.add_argument(
        "--file",
        type=str,
        default='C:\\Users\\Александр\\PycharmProjects\\timesfm\\dataset\\test_ns_e1_16_t.csv',
        help="Путь к CSV файлу для проверки."
    )
    parser.add_argument("--sep", type=str, default=';', help="Разделитель в CSV файле.")
    parser.add_argument("--decimal", type=str, default=',', help="Десятичный разделитель в CSV файле.")

    args = parser.parse_args()

    find_problematic_rows(args.file, args.sep, args.decimal)

