# ./steps/step_5_tool_depth.py
import os
import pandas as pd
import logging
from config import PIPELINE_CONFIG
from tqdm import tqdm

# Получаем настроенный логгер
logger = logging.getLogger(__name__)


def run_step_5():
    """
    Шаг 5: Расчет глубины инструмента по перемещению крюка с учетом состояния клиньев.
    Возвращает True в случае успеха и False в случае ошибки.
    """
    logger.info("---[ Шаг 5: Расчет глубины инструмента ]---")
    try:
        # --- 1. Проверка и загрузка входных данных ---
        step_4_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_4_OUTPUT_FILE'])
        if not os.path.exists(step_4_path):
            logger.error(f"Файл {step_4_path} не найден. Запустите Шаг 4.")
            return False

        df = pd.read_csv(step_4_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_4_path}, содержащий {len(df)} строк.")

        # --- 2. Получение имен столбцов из конфига ---
        hook_height_col = PIPELINE_CONFIG.get('STEP_5_HOOK_HEIGHT_COLUMN', 'Высота_крюка_103')
        slips_col = PIPELINE_CONFIG.get('STEP_5_SLIPS_COLUMN', 'клинья_binary')
        output_col = PIPELINE_CONFIG.get('STEP_5_OUTPUT_COLUMN', 'Глубина_инструмента')

        # Проверка наличия необходимых столбцов
        required_cols = [hook_height_col, slips_col]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Необходимый столбец '{col}' не найден в файле.")
                return False

        # --- 3. Расчет глубины инструмента ---
        logger.info(f"Расчет столбца '{output_col}' на основе '{hook_height_col}' и '{slips_col}'.")

        # Инициализация нового столбца и начальных значений
        # Создаем список для хранения результатов, так как это быстрее, чем .loc
        instrument_depth = [0.0] * len(df)

        # Используем .values для ускорения доступа к данным в цикле
        hook_height_values = df[hook_height_col].values
        slips_values = df[slips_col].values

        # Итерация со второй строки, так как для первой нужен предыдущий шаг
        for i in tqdm(range(1, len(df)), desc="Расчет глубины инструмента"):
            # Разница высоты крюка между текущим и предыдущим шагом
            delta_hook_height = hook_height_values[i] - hook_height_values[i - 1]

            # Если клинья открыты (0), то глубина инструмента меняется
            if slips_values[i] == 0:
                # Если крюк идет вниз (delta < 0), глубина увеличивается.
                # Если крюк идет вверх (delta > 0), глубина уменьшается.
                # Поэтому мы вычитаем дельту.
                instrument_depth[i] = instrument_depth[i - 1] - delta_hook_height
            # Если буровая колонна в клиньях (1), глубина инструмента не меняется
            else:
                instrument_depth[i] = instrument_depth[i - 1]

        df[output_col] = instrument_depth
        logger.info(f"Расчет завершен. Первые 5 значений в '{output_col}': {df[output_col].head().tolist()}")

        # --- 4. Сохранение результата ---
        output_filename = PIPELINE_CONFIG.get('STEP_5_OUTPUT_FILE', 'step_5_with_tool_depth.csv')
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], output_filename)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Итоговый датасет с глубиной инструмента сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 5: {e}", exc_info=True)
        return False