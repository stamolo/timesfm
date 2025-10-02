import os
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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
    - Добавлена проверка минимальной глубины долота с противоусловием по весу.
    - Добавлена фильтрация по давлению для обучения и поиска аномалий.
    - Добавлена проверка на "устаревание" модели: если модель не переобучалась дольше X минут, она сбрасывается.
    """
    logger.info("---[ Шаг 11: Поиск аномалий (динамическое окно, проверка динамики и разрывов) ]---")
    try:
        # 1. Загрузка данных и параметров
        step_10_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_10_OUTPUT_FILE'])
        df = pd.read_csv(step_10_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_10_path}, содержащий {len(df)} строк.")

        # --- НОВЫЕ ПАРАМЕТРЫ ДЛЯ ВЫБОРА МОДЕЛИ ---
        model_type = PIPELINE_CONFIG.get('STEP_11_MODEL_TYPE', 'linear').lower()
        model_params = PIPELINE_CONFIG.get('STEP_11_MODEL_PARAMS', {})
        logger.info(f"Выбрана модель для анализа: {model_type}")
        # ---------------------------------------------

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

        # --- ПАРАМЕТРЫ ДЛЯ ПРОВЕРКИ ГЛУБИНЫ И ВЕСА ---
        bit_depth_col = PIPELINE_CONFIG.get('STEP_11_BIT_DEPTH_COLUMN')
        enable_depth_check = PIPELINE_CONFIG.get('STEP_11_ENABLE_MIN_DEPTH_CHECK', False)
        min_depth_threshold = PIPELINE_CONFIG.get('STEP_11_MIN_DEPTH_THRESHOLD', 100.0)
        enable_weight_override = PIPELINE_CONFIG.get('STEP_11_ENABLE_WEIGHT_OVERRIDE', False)
        min_weight_override = PIPELINE_CONFIG.get('STEP_11_MIN_WEIGHT_OVERRIDE', 35.0)

        # --- ПАРАМЕТРЫ ДЛЯ ФИЛЬТРАЦИИ ПО ДАВЛЕНИЮ ---
        pressure_col = PIPELINE_CONFIG.get('STEP_11_PRESSURE_COLUMN')
        enable_pressure_filter = PIPELINE_CONFIG.get('STEP_11_ENABLE_PRESSURE_FILTER', False)
        pressure_threshold = PIPELINE_CONFIG.get('STEP_11_PRESSURE_THRESHOLD', 200.0)

        # --- НОВЫЕ ПАРАМЕТРЫ ДЛЯ СБРОСА УСТАРЕВШЕЙ МОДЕЛИ ---
        enable_stale_check = PIPELINE_CONFIG.get('STEP_11_ENABLE_MODEL_STALE_CHECK', False)
        stale_threshold_minutes = PIPELINE_CONFIG.get('STEP_11_MODEL_STALE_THRESHOLD_MINUTES', 60)

        if enable_stale_check:
            logger.info(f"Включена проверка на устаревание модели. Порог: {stale_threshold_minutes} минут.")

        if enable_pressure_filter:
            logger.info(
                f"Включен фильтр по давлению. Данные, где '{pressure_col}' > {pressure_threshold} атм, будут игнорироваться.")
            if pressure_col not in df.columns:
                logger.error(f"Столбец для фильтрации по давлению '{pressure_col}' не найден. Фильтр будет отключен.")
                enable_pressure_filter = False

        if enable_depth_check:
            logger.info(
                f"Включена проверка минимальной глубины. Аномалии не будут распознаваться, если '{bit_depth_col}' < {min_depth_threshold} м.")
            if bit_depth_col not in df.columns:
                logger.error(f"Столбец для проверки глубины '{bit_depth_col}' не найден. Проверка будет отключена.")
                enable_depth_check = False
            if enable_weight_override:
                logger.info(f"...ИСКЛЮЧЕНИЕ: анализ будет выполнен, если '{target_col}' > {min_weight_override} т.")
                if target_col not in df.columns:
                    logger.error(f"Столбец для проверки веса '{target_col}' не найден. Противоусловие будет отключено.")
                    enable_weight_override = False

        # Загрузка параметров обрезки
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
            last_index = df_filtered.index[0] if not df_filtered.empty else 0
            for end_index in break_indices:
                # Находим реальный индекс строки в df_filtered, который соответствует end_index из df
                loc_end = df_filtered.index.get_loc(end_index)
                data_blocks.append(df_filtered.loc[last_index:df_filtered.index[loc_end - 1]].copy())
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

            # --- Фабрика моделей ---
            try:
                if model_type == 'ridge':
                    params = model_params.get('ridge', {})
                    model = Ridge(**params)
                    logger.info(f"Создана модель Ridge с параметрами: {params}")
                elif model_type == 'lasso':
                    params = model_params.get('lasso', {})
                    model = Lasso(**params)
                    logger.info(f"Создана модель Lasso с параметрами: {params}")
                elif model_type == 'elasticnet':
                    params = model_params.get('elasticnet', {})
                    model = ElasticNet(**params)
                    logger.info(f"Создана модель ElasticNet с параметрами: {params}")
                elif model_type == 'linear':
                    model = LinearRegression()
                    logger.info("Создана стандартная модель LinearRegression.")
                else:
                    logger.warning(f"Неизвестный тип модели '{model_type}'. Будет использована LinearRegression по умолчанию.")
                    model = LinearRegression()
            except Exception as e:
                logger.error(f"Ошибка при создании модели типа '{model_type}': {e}. Будет использована LinearRegression по умолчанию.")
                model = LinearRegression()
            # --- Конец фабрики моделей ---

            is_model_fitted = False
            last_training_time = None  # Для отслеживания времени последнего обучения

            # *** ОПТИМИЗАЦИЯ: Инициализируем столбцы в локальном блоке для ускорения расчетов ***
            block_df.loc[:, 'predicted_weight'] = np.nan
            block_df.loc[:, 'residual'] = np.nan
            block_df.loc[:, 'is_anomaly'] = 0

            for i in tqdm(range(min_window_size, len(block_df), window_step),
                          desc=f"Анализ блока {block_num}/{total_blocks}"):

                # --- НОВАЯ ЛОГИКА: ПРОВЕРКА УСТАРЕВАНИЯ МОДЕЛИ ---
                if is_model_fitted and enable_stale_check and last_training_time:
                    current_time = block_df[time_col].iloc[i]
                    time_diff_minutes = (current_time - last_training_time).total_seconds() / 60
                    if time_diff_minutes > stale_threshold_minutes:
                        logger.info(
                            f"Модель устарела (не обучалась {time_diff_minutes:.1f} мин > "
                            f"порога {stale_threshold_minutes} мин). Сброс модели."
                        )
                        is_model_fitted = False
                        last_training_time = None
                # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

                start_index = max(0, i - max_window_size)
                train_window_full = block_df.iloc[start_index:i]

                clean_train_window = train_window_full[train_window_full['is_anomaly'] == 0]

                # --- Фильтрация обучающего окна по давлению ---
                if enable_pressure_filter:
                    clean_train_window = clean_train_window[clean_train_window[pressure_col] <= pressure_threshold]
                # --- КОНЕЦ ---

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
                    # --- НОВАЯ ЛОГИКА: Запоминаем время последнего обучения ---
                    last_training_time = clean_train_window[time_col].iloc[-1]
                    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

                if not is_model_fitted:
                    continue

                predict_block = block_df.iloc[i: i + window_step]

                # --- ОБНОВЛЕННАЯ ЛОГИКА: ФИЛЬТРАЦИЯ БЛОКА ДЛЯ АНАЛИЗА ---
                analysis_block = predict_block

                # 1. Фильтр по давлению
                if enable_pressure_filter:
                    analysis_block = analysis_block[analysis_block[pressure_col] <= pressure_threshold]

                # 2. Фильтр по глубине и весу (если блок еще не пуст)
                if not analysis_block.empty and enable_depth_check:
                    depth_mask = analysis_block[bit_depth_col] >= min_depth_threshold
                    if enable_weight_override:
                        weight_mask = analysis_block[target_col] > min_weight_override
                        final_mask = depth_mask | weight_mask
                    else:
                        final_mask = depth_mask
                    analysis_block = analysis_block[final_mask]

                if analysis_block.empty:
                    continue
                # --- КОНЕЦ ОБНОВЛЕННОЙ ЛОГИКИ ---

                predictions = model.predict(analysis_block[feature_cols])

                # Применение обрезки, если опция включена
                if use_clip:
                    predictions = np.clip(predictions, min_clip, max_clip)

                # *** ОПТИМИЗАЦИЯ: Обновляем данные в обоих DataFrame ***
                residuals = analysis_block[target_col] - predictions
                df.loc[analysis_block.index, 'predicted_weight'] = predictions
                df.loc[analysis_block.index, 'residual'] = residuals
                block_df.loc[analysis_block.index, 'predicted_weight'] = predictions
                block_df.loc[analysis_block.index, 'residual'] = residuals

                potential_anomalies_mask = abs(residuals) > anomaly_threshold

                if potential_anomalies_mask.any():
                    # *** ОПТИМИЗАЦИЯ: Вся логика поиска последовательностей теперь работает
                    # с `block_df`, который имеет фиксированный и относительно небольшой размер.
                    # Это предотвращает замедление по мере обработки всего файла. ***

                    # 1. Создаем маску потенциальных аномалий для всего блока
                    potential_mask_in_block = abs(block_df['residual']) > anomaly_threshold

                    # 2. Создаем группы из смежных одинаковых значений в этой маске
                    grouper = potential_mask_in_block.diff().ne(0).cumsum()

                    # 3. Получаем ID групп, которые содержат аномалии
                    anomaly_group_ids = grouper[potential_mask_in_block]

                    if not anomaly_group_ids.empty:
                        # 4. Рассчитываем размеры только для этих аномальных групп
                        block_sizes = anomaly_group_ids.groupby(anomaly_group_ids).transform('size')

                        # 5. Находим индексы, где размер группы соответствует минимальному порогу
                        actual_anomaly_indices = block_sizes[block_sizes >= min_consecutive].index

                        if not actual_anomaly_indices.empty:
                            # 6. Обновляем флаги в обоих DataFrame
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
