import os
import pandas as pd
import numpy as np
import logging
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def run_step_9():
    """
    Шаг 9: Расчет производных и квадратов значений.
    """
    logger.info("---[ Шаг 9: Расчет производных и квадратов значений ]---")
    try:
        # 1. Загрузка данных из предыдущего шага
        step_8_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_8_OUTPUT_FILE'])
        if not os.path.exists(step_8_path):
            logger.error(f"Файл {step_8_path} не найден. Запустите Шаг 8.")
            return False

        df = pd.read_csv(step_8_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_8_path}, содержащий {len(df)} строк.")

        # 2. Получение имен столбцов из конфига
        hook_height_col = PIPELINE_CONFIG['STEP_9_HOOK_HEIGHT_COLUMN']
        weight_col = PIPELINE_CONFIG['STEP_9_WEIGHT_COLUMN']
        bit_depth_col = PIPELINE_CONFIG['STEP_9_BIT_DEPTH_COLUMN']
        speed_col = PIPELINE_CONFIG['STEP_9_OUTPUT_SPEED_COLUMN']
        weight_delta_col = PIPELINE_CONFIG['STEP_9_OUTPUT_WEIGHT_DELTA_COLUMN']
        bit_depth_sq_col = PIPELINE_CONFIG['STEP_9_OUTPUT_BIT_DEPTH_SQUARED_COLUMN']
        speed_sq_col = PIPELINE_CONFIG['STEP_9_OUTPUT_SPEED_SQUARED_COLUMN']
        # --- НОВЫЙ СТОЛБЕЦ ---
        signed_speed_sq_col = PIPELINE_CONFIG['STEP_9_OUTPUT_SIGNED_SPEED_SQUARED_COLUMN']

        # Проверка наличия исходных столбцов
        required_cols = [hook_height_col, weight_col, bit_depth_col]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Необходимый столбец '{col}' не найден в файле.")
                return False

        # 3. Расчет производных (разностных значений)
        logger.info(f"Расчет скорости инструмента ('{speed_col}') на основе '{hook_height_col}'.")
        # Скорость считается как изменение высоты. Знак инвертируется,
        # чтобы движение инструмента вниз (уменьшение высоты крюка) давало положительную скорость.
        df[speed_col] = -df[hook_height_col].diff()

        logger.info(f"Расчет изменения веса на крюке ('{weight_delta_col}') на основе '{weight_col}'.")
        df[weight_delta_col] = df[weight_col].diff()

        # Первые значения после .diff() будут NaN. Заполняем их нулями.
        df.fillna({speed_col: 0, weight_delta_col: 0}, inplace=True)
        logger.info("Значения NaN в первых строках производных столбцов заменены на 0.")

        # 4. Расчет квадратов значений
        logger.info(f"Расчет квадрата глубины долота ('{bit_depth_sq_col}').")
        df[bit_depth_sq_col] = df[bit_depth_col] ** 2

        logger.info(f"Расчет квадрата скорости инструмента ('{speed_sq_col}').")
        df[speed_sq_col] = df[speed_col] ** 2

        # --- НОВАЯ ЛОГИКА ---
        logger.info(f"Расчет квадрата скорости со знаком направления ('{signed_speed_sq_col}').")
        # Умножаем абсолютное значение (квадрат) на знак исходной скорости.
        # Это сохраняет магнитуду квадрата, но добавляет контекст направления.
        df[signed_speed_sq_col] = df[speed_sq_col] * np.sign(df[speed_col])

        # 5. Сохранение результата
        output_filename = PIPELINE_CONFIG['STEP_9_OUTPUT_FILE']
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], output_filename)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Итоговый датасет с производными и квадратами сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 9: {e}", exc_info=True)
        return False
