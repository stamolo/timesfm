import os
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def find_max_continuous_travel(travel_diffs):
    """
    Находит максимальный непрерывный путь вверх и вниз, итеративно проходя по данным.
    """
    max_up_travel = 0
    current_up_travel = 0
    max_down_travel = 0
    current_down_travel = 0

    for diff in travel_diffs.dropna():
        if diff > 0:
            current_up_travel += diff
            # При смене направления сбрасываем счетчик другого направления
            max_down_travel = max(max_down_travel, current_down_travel)
            current_down_travel = 0
        elif diff < 0:
            current_down_travel += abs(diff)
            # При смене направления сбрасываем счетчик другого направления
            max_up_travel = max(max_up_travel, current_up_travel)
            current_up_travel = 0
        else:  # diff == 0
            # Если движение остановилось, фиксируем и сбрасываем оба счетчика
            max_up_travel = max(max_up_travel, current_up_travel)
            current_up_travel = 0
            max_down_travel = max(max_down_travel, current_down_travel)
            current_down_travel = 0

    # Финальная проверка после окончания цикла на случай, если движение не прерывалось до конца окна
    max_up_travel = max(max_up_travel, current_up_travel)
    max_down_travel = max(max_down_travel, current_down_travel)

    return max_up_travel, max_down_travel


def run_step_11():
    """
    Шаг 11: Построение регрессионной модели в скользящем окне для поиска аномалий.
    - Учитывает разрывы в данных для сброса модели.
    - Обучается только на участках с достаточной динамикой движения крюка.
    - Использует динамический размер окна для обучения.
    """
    logger.info("---[ Шаг 11: Поиск аномалий (динамическое окно, проверка динамики и разрывов) ]---")
    try:
        # 1. Загрузка данных и параметров
        step_10_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_10_OUTPUT_FILE'])
        df = pd.read_csv(step_10_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_10_path}, содержащий {len(df)} строк.")

        target_col = PIPELINE_CONFIG['STEP_11_TARGET_COLUMN']
        feature_cols = PIPELINE_CONFIG['STEP_11_FEATURE_COLUMNS']
        slips_col = PIPELINE_CONFIG['STEP_11_SLIPS_COLUMN']
        above_bhd_col = PIPELINE_CONFIG['STEP_11_ABOVE_BHD_COLUMN']

        min_window_size = PIPELINE_CONFIG['STEP_11_MIN_WINDOW_SIZE']
        max_window_size = PIPELINE_CONFIG['STEP_11_MAX_WINDOW_SIZE']
        window_step = PIPELINE_CONFIG.get('STEP_11_WINDOW_STEP', 1)

        anomaly_threshold = PIPELINE_CONFIG['STEP_11_ANOMALY_THRESHOLD']
        min_consecutive = PIPELINE_CONFIG.get('STEP_11_CONSECUTIVE_ANOMALIES_MIN', 1)
        time_col = PIPELINE_CONFIG.get('SORT_COLUMN')
        df[time_col] = pd.to_datetime(df[time_col].str.replace(',', '.'))
        time_gap_minutes = PIPELINE_CONFIG.get('STEP_11_TIME_GAP_THRESHOLD_MINUTES', 15)
        hook_height_col = PIPELINE_CONFIG.get('STEP_9_HOOK_HEIGHT_COLUMN')
        min_travel = PIPELINE_CONFIG.get('STEP_11_MIN_CONTINUOUS_TRAVEL_EACH_DIRECTION', 10.0)
        training_flag_col = PIPELINE_CONFIG.get('STEP_11_TRAINING_FLAG_COLUMN', 'Модель_обучалась_флаг')

        # --- ИСПРАВЛЕНИЕ: Загрузка параметров обрезки ---
        use_clip = PIPELINE_CONFIG.get('STEP_11_USE_PREDICTION_CLIP', False)
        min_clip = PIPELINE_CONFIG.get('STEP_11_PREDICTION_MIN_CLIP', 0)
        max_clip = PIPELINE_CONFIG.get('STEP_11_PREDICTION_MAX_CLIP', 150)

        # 2. Фильтрация данных и разделение на блоки по времени
        df_filtered = df[(df[slips_col] == 0) & (df[above_bhd_col] == 1)].copy()
        logger.info(f"Найдено {len(df_filtered)} точек данных, удовлетворяющих условиям для анализа.")

        data_blocks = []
        if time_gap_minutes > 0:
            logger.info(f"Включена функция сброса модели при разрыве данных более {time_gap_minutes} минут.")
            time_diffs = df_filtered[time_col].diff().dt.total_seconds() / 60
            break_indices = df_filtered.index[time_diffs > time_gap_minutes].tolist()
            last_index = 0
            for end_index in break_indices:
                data_blocks.append(df_filtered.loc[last_index:end_index - 1].copy())
                last_index = end_index
            data_blocks.append(df_filtered.loc[last_index:].copy())
        else:
            data_blocks.append(df_filtered.copy())

        # 3. Инициализация столбцов в основном DataFrame
        df['predicted_weight'] = np.nan
        df['residual'] = np.nan
        df['is_anomaly'] = 0
        df[training_flag_col] = 0

        # 4. Основной цикл по блокам и внутри блоков
        total_blocks = len(data_blocks)
        for block_num, block_df in enumerate(data_blocks, 1):
            if len(block_df) < min_window_size:
                logger.warning(
                    f"Блок данных (ID {block_num - 1}) после разрыва слишком мал ({len(block_df)} точек) и будет пропущен.")
                continue

            logger.info(f"Обработка блока данных #{block_num} из {total_blocks}, размер: {len(block_df)} точек.")

            model = LinearRegression()
            is_model_fitted = False

            block_df.loc[:, 'is_anomaly'] = 0
            block_df.loc[:, 'potential_anomaly_flag'] = 0

            for i in tqdm(range(min_window_size, len(block_df), window_step),
                          desc=f"Анализ блока {block_num}/{total_blocks}"):

                start_index = max(0, i - max_window_size)
                train_window_full = block_df.iloc[start_index:i]

                clean_train_window = train_window_full[train_window_full['is_anomaly'] == 0]

                # Проверка условий для переобучения
                should_retrain = False
                if len(clean_train_window) >= min_window_size:
                    travel_diffs = clean_train_window[hook_height_col].diff()
                    max_upward, max_downward = find_max_continuous_travel(travel_diffs)

                    if max_upward >= min_travel and max_downward >= min_travel:
                        should_retrain = True

                if should_retrain:
                    model.fit(clean_train_window[feature_cols], clean_train_window[target_col])
                    is_model_fitted = True
                    df.loc[clean_train_window.index, training_flag_col] = 1

                if not is_model_fitted:
                    continue

                predict_block = block_df.iloc[i: i + window_step]
                predictions = model.predict(predict_block[feature_cols])

                # --- ИСПРАВЛЕНИЕ: Применение обрезки, если опция включена ---
                if use_clip:
                    predictions = np.clip(predictions, min_clip, max_clip)

                df.loc[predict_block.index, 'predicted_weight'] = predictions
                df.loc[predict_block.index, 'residual'] = predict_block[target_col] - predictions

                potential_anomalies_mask = abs(df.loc[predict_block.index, 'residual']) > anomaly_threshold

                if potential_anomalies_mask.any():
                    potential_anomaly_indices = predict_block.index[potential_anomalies_mask]
                    block_df.loc[potential_anomaly_indices, 'potential_anomaly_flag'] = 1

                    grouper = (abs(df['residual']) > anomaly_threshold).diff().ne(0).cumsum()
                    block_sizes = df.loc[potential_anomaly_indices].groupby(grouper.loc[potential_anomaly_indices])[
                        'residual'].transform('size')

                    actual_anomaly_indices = block_sizes[block_sizes >= min_consecutive].index

                    if not actual_anomaly_indices.empty:
                        df.loc[actual_anomaly_indices, 'is_anomaly'] = 1
                        block_df.loc[actual_anomaly_indices, 'is_anomaly'] = 1

        logger.info(f"Найдено {df['is_anomaly'].sum()} итоговых аномальных точек.")

        # 5. Сохранение результата
        output_filename = PIPELINE_CONFIG['STEP_11_OUTPUT_FILE']
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], output_filename)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Итоговый датасет с предсказаниями и аномалиями сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 11: {e}", exc_info=True)
        return False

