import os
import pandas as pd
import logging
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def run_step_8():
    """
    Шаг 8: Расчет глубины забоя как кумулятивного максимума от
    финальной глубины инструмента, рассчитанной на Шаге 7.
    """
    logger.info("---[ Шаг 8: Расчет глубины забоя ]---")
    try:
        # --- 1. Загрузка данных и параметров ---
        step_7_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_7_OUTPUT_FILE'])
        if not os.path.exists(step_7_path):
            logger.error(f"Файл {step_7_path} не найден. Запустите Шаг 7.")
            return False

        df = pd.read_csv(step_7_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_7_path}, содержащий {len(df)} строк.")

        # Параметры из конфига
        tool_depth_col = PIPELINE_CONFIG['STEP_8_INPUT_TOOL_DEPTH_COLUMN']
        output_col = PIPELINE_CONFIG['STEP_8_OUTPUT_BHD_COLUMN']

        if tool_depth_col not in df.columns:
            logger.error(f"Входной столбец '{tool_depth_col}' не найден в файле.")
            return False

        # --- 2. Расчет глубины забоя ---
        logger.info(f"Расчет '{output_col}' как кумулятивного максимума от '{tool_depth_col}'...")

        # Используем встроенную и очень быструю функцию pandas cummax()
        df[output_col] = df[tool_depth_col].cummax()

        logger.info("Расчет глубины забоя успешно завершен.")

        # --- 3. Сохранение результата ---
        output_filename = PIPELINE_CONFIG['STEP_8_OUTPUT_FILE']
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], output_filename)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Финальный датасет с расчетной глубиной забоя сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 8: {e}", exc_info=True)
        return False
