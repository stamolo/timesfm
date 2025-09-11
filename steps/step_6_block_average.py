import os
import pandas as pd
import logging
from config import PIPELINE_CONFIG

# Получаем настроенный логгер
logger = logging.getLogger(__name__)


def run_step_6():
    """
    Шаг 6 (Новая логика): Расчет среднего веса на крюке для
    каждого непрерывного блока состояний клиньев (0 или 1).
    """
    logger.info("---[ Шаг 6: Расчет среднего веса по блокам состояний клиньев ]---")
    try:
        # --- 1. Загрузка данных и параметров из конфига ---
        step_5_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_5_OUTPUT_FILE'])
        if not os.path.exists(step_5_path):
            logger.error(f"Файл {step_5_path} не найден. Запустите Шаг 5.")
            return False

        df = pd.read_csv(step_5_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_5_path}, содержащий {len(df)} строк.")

        # Параметры из конфига
        slips_col = PIPELINE_CONFIG['STEP_6_INPUT_SLIPS_COLUMN']
        weight_col = PIPELINE_CONFIG['STEP_6_INPUT_WEIGHT_COLUMN']
        output_col = PIPELINE_CONFIG['STEP_6_OUTPUT_AVG_WEIGHT_COLUMN']

        # Проверка наличия столбцов
        if slips_col not in df.columns or weight_col not in df.columns:
            logger.error(f"Необходимые столбцы '{slips_col}' или '{weight_col}' не найдены в файле.")
            return False

        logger.info(f"Создание 'ступенчатого' столбца '{output_col}' на основе '{weight_col}' и '{slips_col}'.")

        # --- 2. Расчет среднего значения по блокам ---
        # Создаем идентификатор для каждого непрерывного блока (где значение slips_col не меняется)
        # (df[slips_col].diff() != 0) создает True в начале каждого нового блока
        # .cumsum() превращает это в уникальный номер для каждого блока: 1, 2, 3, ...
        block_id = (df[slips_col].diff() != 0).cumsum()

        # Группируем по этому ID, для каждой группы считаем среднее значение веса
        # и с помощью transform "растягиваем" это среднее на все строки в группе.
        # Это эффективный способ создать "ступенчатый" столбец.
        df[output_col] = df.groupby(block_id)[weight_col].transform('mean')

        logger.info("Расчет среднего веса по блокам завершен.")

        # --- 3. Сохранение результата ---
        output_filename = PIPELINE_CONFIG['STEP_6_OUTPUT_FILE']
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], output_filename)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Итоговый датасет со средним весом по блокам сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 6: {e}", exc_info=True)
        return False
