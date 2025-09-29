import os
import pandas as pd
import numpy as np
import logging
from config import PIPELINE_CONFIG
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_step_7():
    """
    Шаг 7: Обнуление глубины с использованием двойной проверки:
    1. Падение веса между рабочими блоками (N vs Z).
    2. Близость нового рабочего веса к весу на клиньях.
    Генерирует отчет с детальным обоснованием по обоим условиям.
    """
    logger.info("---[ Шаг 7: Продвинутый сброс (двойная проверка) и детальный отчет ]---")
    try:
        # --- 1. Загрузка данных и параметров ---
        step_6_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_6_OUTPUT_FILE'])
        if not os.path.exists(step_6_path):
            logger.error(f"Файл {step_6_path} не найден. Запустите Шаг 6.")
            return False

        df = pd.read_csv(step_6_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_6_path}, содержащий {len(df)} строк.")

        # Параметры из конфига
        n = PIPELINE_CONFIG['STEP_7_PREVIOUS_BLOCKS_N']
        z = PIPELINE_CONFIG['STEP_7_CURRENT_BLOCKS_Z']
        y = PIPELINE_CONFIG['STEP_7_MIN_PREV_BLOCK_LENGTH_Y']
        m = PIPELINE_CONFIG['STEP_7_MIN_BLOCK_LENGTH_M']
        x = PIPELINE_CONFIG['STEP_7_WEIGHT_DROP_THRESHOLD_X']
        n_slips = PIPELINE_CONFIG['STEP_7_SLIPS_BLOCKS_N']
        r = PIPELINE_CONFIG['STEP_7_MAX_WEIGHT_ABOVE_SLIPS_R']

        # Имена столбцов
        slips_binary_col = PIPELINE_CONFIG['STEP_5_SLIPS_COLUMN']
        slips_0123_col = PIPELINE_CONFIG['STEP_4_INPUT_COLUMN']
        weight_col = PIPELINE_CONFIG['STEP_6_INPUT_WEIGHT_COLUMN']
        time_col = PIPELINE_CONFIG.get('SORT_COLUMN', 'Время_204')
        hook_height_col = PIPELINE_CONFIG['STEP_5_HOOK_HEIGHT_COLUMN']
        output_col = PIPELINE_CONFIG['STEP_7_OUTPUT_COLUMN']

        # --- 2. Идентификация блоков и сбор статистики ---
        logger.info("Идентификация блоков и сбор статистики...")
        df['block_id'] = (df[slips_binary_col].diff() != 0).cumsum()

        df_with_index = df.reset_index()

        block_stats = df_with_index.groupby('block_id').agg(
            # ИСПРАВЛЕНО: 'idxmin'/'idxmax' по 'block_id' заменены на 'min'/'max' по 'index'.
            start_row=('index', 'min'),
            end_row=('index', 'max'),
            start_time=(time_col, 'first'),
            end_time=(time_col, 'last'),
            block_length=('block_id', 'size'),
            avg_weight=(weight_col, 'mean'),
            slips_state_binary=(slips_binary_col, 'first'),
            slips_state_0123=(slips_0123_col, 'first')
        ).reset_index(drop=True)

        # --- 3. Поиск точек сброса с детальным логированием в отчет ---
        logger.info(f"Анализ блоков (N={n}, Z={z}, R={r}) для поиска точек сброса...")
        block_stats['Точка сброса'] = False
        block_stats['Обоснование сброса'] = ''

        working_blocks = block_stats[block_stats['slips_state_binary'] == 0].copy()
        slips_blocks = block_stats[block_stats['slips_state_binary'] == 1].copy()
        reset_indices = []

        wb_indices = working_blocks.index.tolist()

        if len(wb_indices) < n + z:
            logger.warning("Недостаточно рабочих блоков для анализа.")
        else:
            for i in range(n, len(wb_indices) - z + 1):
                # Окна блоков
                prev_indices = wb_indices[i - n:i]
                current_indices = wb_indices[i:i + z]

                prev_blocks = working_blocks.loc[prev_indices]
                current_blocks = working_blocks.loc[current_indices]

                valid_prev_blocks = prev_blocks[prev_blocks['block_length'] >= y]
                valid_current_blocks = current_blocks[current_blocks['block_length'] >= m]

                if valid_prev_blocks.empty or valid_current_blocks.empty: continue

                # --- Условие 1: Сравнение рабочих блоков ---
                avg_weight_previous = valid_prev_blocks['avg_weight'].mean()
                avg_weight_current = valid_current_blocks['avg_weight'].mean()
                weight_diff = avg_weight_previous - avg_weight_current
                condition1_met = weight_diff > x

                # --- Условие 2: Сравнение с весом на клиньях ---
                first_current_block_start_row = working_blocks.loc[current_indices[0], 'start_row']
                relevant_slips_blocks = slips_blocks[slips_blocks['end_row'] < first_current_block_start_row].tail(
                    n_slips)

                if relevant_slips_blocks.empty: continue

                avg_weight_slips = relevant_slips_blocks['avg_weight'].mean()
                weight_above_slips = avg_weight_current - avg_weight_slips
                condition2_met = weight_above_slips <= r

                # --- Формирование детального обоснования ---
                reasoning_parts = []
                # Формируем текст для Условия 1
                cond1_status = "ОК" if condition1_met else "НЕ ОК"
                reasoning_parts.append(
                    f"Условие 1 ({cond1_status}): падение веса на {weight_diff:.2f}т "
                    f"({' > ' if condition1_met else ' <= '}{x:.2f}т)."
                )
                # Формируем текст для Условия 2
                cond2_status = "ОК" if condition2_met else "НЕ ОК"
                reasoning_parts.append(
                    f"Условие 2 ({cond2_status}): новый вес {avg_weight_current:.2f}т "
                    f"({' <= ' if condition2_met else ' > '}вес на клиньях {avg_weight_slips:.2f}т + допуск {r:.2f}т)."
                )

                final_reasoning = " | ".join(reasoning_parts)
                first_current_block_idx = current_indices[0]

                # --- Принятие решения и запись в отчет ---
                if condition1_met and condition2_met:
                    reset_point = int(working_blocks.loc[first_current_block_idx, 'start_row'])
                    reset_indices.append(reset_point)

                    block_stats.at[first_current_block_idx, 'Точка сброса'] = True
                    # Добавляем префикс "Сброс" к финальному тексту
                    block_stats.at[first_current_block_idx, 'Обоснование сброса'] = f"Сброс. {final_reasoning}"
                    logger.info(f"Найдена точка сброса на строке {reset_point}: {final_reasoning}")
                else:
                    # Даже если сброса нет, записываем в отчет детальную причину
                    block_stats.at[first_current_block_idx, 'Обоснование сброса'] = f"Нет сброса. {final_reasoning}"

        # --- 4. Генерация и сохранение отчета --- (без изменений)
        logger.info("Генерация расширенного сводного отчета по блокам...")
        report_df = block_stats[[
            'start_time', 'end_time', 'start_row', 'end_row',
            'slips_state_0123', 'slips_state_binary', 'avg_weight', 'block_length',
            'Точка сброса', 'Обоснование сброса'
        ]].copy()
        report_df.columns = [
            'Время начала', 'Время окончания', 'Начальная строка', 'Конечная строка',
            'Режим (клинья_0123)', 'Режим (клинья_binary)', 'Средний вес', 'Длина блока (точек)',
            'Точка сброса', 'Обоснование сброса'
        ]
        report_filename = PIPELINE_CONFIG['STEP_7_BLOCKS_REPORT_FILE']
        report_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], report_filename)
        report_df.to_csv(report_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Расширенный сводный отчет по блокам успешно сохранен в: {report_path}")

        # --- 5. Финальный пересчет глубины --- (без изменений)
        logger.info(f"Найдено {len(reset_indices)} точек сброса. Выполняется финальный пересчет глубины...")
        reset_points_set = set(reset_indices)
        final_depth = [0.0] * len(df)
        hook_height_values = df[hook_height_col].values
        slips_values = df[slips_binary_col].values
        for i in tqdm(range(1, len(df)), desc="Финальный расчет глубины"):
            if i in reset_points_set:
                final_depth[i] = 0.0
            else:
                delta_hook_height = hook_height_values[i] - hook_height_values[i - 1]
                if slips_values[i] == 0:
                    final_depth[i] = final_depth[i - 1] - delta_hook_height
                else:
                    final_depth[i] = final_depth[i - 1]
        df[output_col] = final_depth

        # --- 6. Сохранение результата --- (без изменений)
        output_filename = PIPELINE_CONFIG['STEP_7_OUTPUT_FILE']
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], output_filename)
        df.drop(columns=['block_id'], inplace=True)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Финальный датасет с продвинутой корректировкой глубины сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 7: {e}", exc_info=True)
        return False

