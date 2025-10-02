import os
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    - Добавлена возможность нормализации данных перед обучением.
    - Учитывает разрывы в данных для сброса модели.
    - Обучается только на участках с достаточной динамикой движения крюка.
    - Использует динамический размер окна для обучения.
    - Добавлена проверка минимальной глубины долота с противоусловием по весу.
    - Добавлена фильтрация по давлению для обучения и поиска аномалий.
    - Добавлена проверка на "устаревание" модели: если модель не переобучалась дольше X минут, она сбрасывается.
    - Добавлена балансировка обучающей выборки для устранения перекоса в сторону статичных данных.
    - ИЗМЕНЕНИЕ: Добавлена возможность включать/исключать аномалии из обучающей выборки.
    - ИЗМЕНЕНИЕ: Добавлен расчет вклада каждого признака в предсказание для линейных моделей.
    """
    logger.info("---[ Шаг 11: Поиск аномалий (динамическое окно, проверка динамики и разрывов) ]---")
    try:
        # 1. Загрузка данных и параметров
        step_10_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_10_OUTPUT_FILE'])
        df = pd.read_csv(step_10_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_10_path}, содержащий {len(df)} строк.")

        # --- ПАРАМЕТРЫ ВЫБОРА МОДЕЛИ И НОРМАЛИЗАЦИИ ---
        model_type = PIPELINE_CONFIG.get('STEP_11_MODEL_TYPE', 'linear').lower()
        model_params = PIPELINE_CONFIG.get('STEP_11_MODEL_PARAMS', {})
        normalization_type = PIPELINE_CONFIG.get('STEP_11_NORMALIZATION_TYPE', 'none').lower()
        logger.info(f"Выбрана модель для анализа: {model_type}")
        logger.info(f"Выбран тип нормализации данных: {normalization_type}")
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
        min_travel = PIPELINE_CONFIG.get('STEP_11_MIN_CONTINUOUS_TRAVEL', 3.0)
        training_flag_col = PIPELINE_CONFIG.get('STEP_11_TRAINING_FLAG_COLUMN', 'Модель_обучалась_флаг')

        # --- ИЗМЕНЕНИЕ: Загрузка параметра для управления обучением на аномалиях ---
        exclude_anomalies_from_training = PIPELINE_CONFIG.get('STEP_11_EXCLUDE_ANOMALIES_FROM_TRAINING', True)
        if exclude_anomalies_from_training:
            logger.info("Режим обучения: ранее найденные аномалии будут ИСКЛЮЧАТЬСЯ из обучающей выборки.")
        else:
            logger.info("Режим обучения: модель будет обучаться на ВСЕХ точках, включая ранее найденные аномалии.")
        # -------------------------------------------------------------------------

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

        # --- ПАРАМЕТРЫ БАЛАНСИРОВКИ ДАННЫХ ---
        enable_balancing = PIPELINE_CONFIG.get('STEP_11_ENABLE_DATA_BALANCING', False)
        balancing_col = PIPELINE_CONFIG.get('STEP_11_BALANCING_COLUMN')
        max_stationary_percent = PIPELINE_CONFIG.get('STEP_11_BALANCING_MAX_STATIONARY_PERCENT', 50)
        if enable_balancing:
            logger.info(
                f"Включена балансировка данных по '{balancing_col}'. Макс. % статичных точек: {max_stationary_percent}%.")
            if balancing_col not in df.columns:
                logger.error(f"Столбец для балансировки '{balancing_col}' не найден. Балансировка будет отключена.")
                enable_balancing = False
        # ---------------------------------------------

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

        # --- ИЗМЕНЕНИЕ: Инициализация столбцов для вклада признаков ---
        is_linear_model = model_type in ['linear', 'ridge', 'lasso', 'elasticnet']
        contribution_cols = []
        if is_linear_model:
            logger.info("Инициализация столбцов для хранения вклада признаков в предсказание.")
            contribution_cols = [f'contribution_{f}' for f in feature_cols]
            intercept_col_name = 'contribution_intercept'
            for col in contribution_cols:
                df[col] = np.nan
            df[intercept_col_name] = np.nan
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

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
                    logger.warning(
                        f"Неизвестный тип модели '{model_type}'. Будет использована LinearRegression по умолчанию.")
                    model = LinearRegression()
            except Exception as e:
                logger.error(
                    f"Ошибка при создании модели типа '{model_type}': {e}. Будет использована LinearRegression по умолчанию.")
                model = LinearRegression()
            # --- Конец фабрики моделей ---

            is_model_fitted = False
            scaler = None
            last_training_time = None

            block_df.loc[:, 'predicted_weight'] = np.nan
            block_df.loc[:, 'residual'] = np.nan
            block_df.loc[:, 'is_anomaly'] = 0

            for i in tqdm(range(min_window_size, len(block_df), window_step),
                          desc=f"Анализ блока {block_num}/{total_blocks}"):

                # --- ПРОВЕРКА УСТАРЕВАНИЯ МОДЕЛИ ---
                if is_model_fitted and enable_stale_check and last_training_time:
                    current_time = block_df[time_col].iloc[i]
                    time_diff_minutes = (current_time - last_training_time).total_seconds() / 60
                    if time_diff_minutes > stale_threshold_minutes:
                        logger.info(
                            f"Модель устарела (не обучалась {time_diff_minutes:.1f} мин > "
                            f"порога {stale_threshold_minutes} мин). Сброс модели."
                        )
                        is_model_fitted = False
                        scaler = None
                        last_training_time = None

                start_index = max(0, i - max_window_size)
                train_window_full = block_df.iloc[start_index:i]

                if exclude_anomalies_from_training:
                    clean_train_window = train_window_full[train_window_full['is_anomaly'] == 0]
                else:
                    clean_train_window = train_window_full

                if enable_pressure_filter:
                    clean_train_window = clean_train_window[clean_train_window[pressure_col] <= pressure_threshold]

                should_retrain = False
                if len(clean_train_window) >= min_window_size:
                    travel_diffs = clean_train_window[hook_height_col].diff()
                    max_upward, max_downward = find_max_continuous_travel(travel_diffs)
                    if max_upward >= min_travel or max_downward >= min_travel:
                        should_retrain = True

                if should_retrain:
                    training_data = clean_train_window
                    if enable_balancing:
                        moving_data = training_data[training_data[balancing_col] != 0]
                        stationary_data = training_data[training_data[balancing_col] == 0]
                        n_moving = len(moving_data)
                        n_stationary = len(stationary_data)
                        n_total = n_moving + n_stationary
                        if n_total > 0 and n_moving > 0:
                            current_stationary_percent = (n_stationary / n_total) * 100
                            if current_stationary_percent > max_stationary_percent:
                                if (100 - max_stationary_percent) > 0:
                                    target_n_stationary = int(
                                        (max_stationary_percent * n_moving) / (100 - max_stationary_percent))
                                    target_n_stationary = min(target_n_stationary, n_stationary)
                                    stationary_data_sampled = stationary_data.sample(n=target_n_stationary,
                                                                                     random_state=42)
                                    training_data = pd.concat([moving_data, stationary_data_sampled])

                    X_train = training_data[feature_cols]
                    y_train = training_data[target_col]

                    if normalization_type == 'min_max':
                        scaler = MinMaxScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                    elif normalization_type == 'z_score':
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                    else:  # 'none'
                        scaler = None
                        X_train_scaled = X_train

                    model.fit(X_train_scaled, y_train)
                    is_model_fitted = True
                    df.loc[clean_train_window.index, training_flag_col] = 1
                    last_training_time = clean_train_window[time_col].iloc[-1]

                if not is_model_fitted:
                    continue

                predict_block = block_df.iloc[i: i + window_step]
                analysis_block = predict_block

                if enable_pressure_filter:
                    analysis_block = analysis_block[analysis_block[pressure_col] <= pressure_threshold]

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

                X_predict = analysis_block[feature_cols]
                if scaler:
                    X_predict_scaled = scaler.transform(X_predict)
                else:
                    X_predict_scaled = X_predict

                predictions = model.predict(X_predict_scaled)

                if use_clip:
                    predictions = np.clip(predictions, min_clip, max_clip)

                # --- ИЗМЕНЕНИЕ: Расчет и сохранение вклада каждого признака ---
                if is_linear_model:
                    feature_contributions = X_predict_scaled * model.coef_
                    df.loc[analysis_block.index, contribution_cols] = feature_contributions
                    df.loc[analysis_block.index, intercept_col_name] = model.intercept_
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

                residuals = analysis_block[target_col] - predictions
                df.loc[analysis_block.index, 'predicted_weight'] = predictions
                df.loc[analysis_block.index, 'residual'] = residuals
                block_df.loc[analysis_block.index, 'predicted_weight'] = predictions
                block_df.loc[analysis_block.index, 'residual'] = residuals

                potential_anomalies_mask = abs(residuals) > anomaly_threshold
                if potential_anomalies_mask.any():
                    potential_mask_in_block = abs(block_df['residual']) > anomaly_threshold
                    grouper = potential_mask_in_block.diff().ne(0).cumsum()
                    anomaly_group_ids = grouper[potential_mask_in_block]
                    if not anomaly_group_ids.empty:
                        block_sizes = anomaly_group_ids.groupby(anomaly_group_ids).transform('size')
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

