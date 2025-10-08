import os
import pandas as pd
import logging
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def run_step_2():
    """
    Шаг 2: Добавляет отступы (padding) из средних значений в начало и конец данных.
    Возвращает True в случае успеха и False в случае ошибки.
    """
    logger.info("---[ Шаг 2: Добавление отступов (padding) ]---")
    try:
        step_1_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_1_OUTPUT_FILE'])
        if not os.path.exists(step_1_path):
            logger.error(f"Файл {step_1_path} не найден. Запустите Шаг 1.")
            return False

        df_step_1 = pd.read_csv(step_1_path, sep=';', decimal=',', encoding='utf-8')

        padding_size = PIPELINE_CONFIG['PADDING_SIZE']
        logger.info(f"Будет добавлено по {padding_size} строк в начало и конец файла.")

        prediction_cols = PIPELINE_CONFIG['PREDICTION_INPUT_COLUMNS']
        mean_values = df_step_1[prediction_cols].mean()

        padding_row = pd.DataFrame([mean_values.to_dict()])
        df_padding = pd.concat([padding_row] * padding_size, ignore_index=True)

        df_padded = pd.concat([df_padding, df_step_1, df_padding], ignore_index=True)

        # Используем ffill/bfill для заполнения пропусков в неколичественных столбцах
        df_padded.ffill(inplace=True)
        df_padded.bfill(inplace=True)

        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_2_OUTPUT_FILE'])
        df_padded.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Данные с отступами сохранены в: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка на Шаге 2: {e}", exc_info=True)
        return False

