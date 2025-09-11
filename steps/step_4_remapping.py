import os
import pandas as pd
import logging
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def run_step_4():
    """
    Шаг 4: Перекодирование состояний в бинарный признак.
    Возвращает True в случае успеха и False в случае ошибки.
    """
    logger.info("---[ Шаг 4: Перекодирование состояний в бинарный признак ]---")
    try:
        step_3_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_3_OUTPUT_FILE'])
        if not os.path.exists(step_3_path):
            logger.error(f"Файл {step_3_path} не найден. Запустите Шаг 3.")
            return False

        df = pd.read_csv(step_3_path, sep=';', decimal=',', encoding='utf-8')

        input_col = PIPELINE_CONFIG['STEP_4_INPUT_COLUMN']
        output_col = PIPELINE_CONFIG['STEP_4_OUTPUT_COLUMN']

        if input_col not in df.columns:
            logger.error(f"Входной столбец '{input_col}' не найден в файле {step_3_path}.")
            return False

        # Этап 1: Объединение переходных состояний (1->0, 3->2)
        initial_mapping = PIPELINE_CONFIG['STEP_4_MAPPING_INITIAL']
        logger.info(f"Применение начального маппинга: {initial_mapping}")
        temp_col = df[input_col].replace(initial_mapping)

        # Этап 2: Создание финального бинарного состояния (0->0, 2->1)
        final_mapping = PIPELINE_CONFIG['STEP_4_MAPPING_FINAL']
        logger.info(f"Применение финального маппинга: {final_mapping}")
        df[output_col] = temp_col.replace(final_mapping)

        # Проверка, что все значения в новом столбце - 0 или 1
        unique_values = df[output_col].unique()
        logger.info(f"Уникальные значения в новом столбце '{output_col}': {unique_values}")

        # Сохранение результата
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_4_OUTPUT_FILE'])
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Новый бинарный признак '{output_col}' добавлен. Результат сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 4: {e}", exc_info=True)
        return False
