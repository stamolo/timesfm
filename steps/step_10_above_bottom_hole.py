import os
import pandas as pd
import logging
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def run_step_10():
    """
    Шаг 10: Расчет параметра 'Над забоем, м' и его бинарной версии.
    'Над забоем, м' = 'Глубина_забоя_36' - 'Глубина_долота_35'.
    Бинарная версия = 1 если 'Над забоем, м' > порога, иначе 0.
    """
    logger.info("---[ Шаг 10: Расчет параметра 'Над забоем, м' и его бинарной версии ]---")
    try:
        # 1. Загрузка данных из предыдущего шага
        step_9_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_9_OUTPUT_FILE'])
        if not os.path.exists(step_9_path):
            logger.error(f"Файл {step_9_path} не найден. Запустите Шаг 9.")
            return False

        df = pd.read_csv(step_9_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_9_path}, содержащий {len(df)} строк.")

        # 2. Получение имен столбцов из конфига
        bhd_col = PIPELINE_CONFIG['STEP_10_BHD_COLUMN']
        bit_depth_col = PIPELINE_CONFIG['STEP_10_BIT_DEPTH_COLUMN']
        output_col = PIPELINE_CONFIG['STEP_10_OUTPUT_COLUMN']

        # Проверка наличия исходных столбцов
        if bhd_col not in df.columns:
            logger.error(f"Необходимый столбец '{bhd_col}' не найден в файле.")
            return False
        if bit_depth_col not in df.columns:
            logger.error(f"Необходимый столбец '{bit_depth_col}' не найден в файле.")
            return False

        # 3. Расчет основного параметра
        logger.info(f"Расчет '{output_col}' как разницы между '{bhd_col}' и '{bit_depth_col}'.")
        df[output_col] = df[bhd_col] - df[bit_depth_col]

        # 4. Расчет бинарной версии параметра, если опция включена
        calculate_binary = PIPELINE_CONFIG.get('STEP_10_CALCULATE_BINARY', False)
        if calculate_binary:
            threshold = PIPELINE_CONFIG.get('STEP_10_BINARY_THRESHOLD', 35.0)
            binary_output_col = PIPELINE_CONFIG.get('STEP_10_BINARY_OUTPUT_COLUMN', 'Над забоем, м (бинарный)')

            logger.info(f"Расчет бинарного параметра '{binary_output_col}' с порогом > {threshold}.")

            # (df[output_col] > threshold) вернет True/False. .astype(int) преобразует в 1/0.
            df[binary_output_col] = (df[output_col] > threshold).astype(int)
            logger.info(f"Бинарный параметр успешно рассчитан.")
        else:
            logger.info("Расчет бинарной версии параметра пропущен (отключен в конфиге).")

        # 5. Сохранение результата
        output_filename = PIPELINE_CONFIG['STEP_10_OUTPUT_FILE']
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], output_filename)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Итоговый датасет с новыми параметрами сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 10: {e}", exc_info=True)
        return False

