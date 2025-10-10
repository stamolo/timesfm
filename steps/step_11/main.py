import os
import pandas as pd
import numpy as np
import logging
import time
import json
from collections import defaultdict
from tqdm import tqdm
from config import PIPELINE_CONFIG
# ИЗМЕНЕНИЕ: Импортируем наш обработчик БД
from utils import db_handler

from .data_preprocessor import DataPreprocessor, balance_training_data
from .models import model_factory
from .interpretation import interpreter_factory

logger = logging.getLogger(__name__)


class TimeTracker:
    """Вспомогательный класс для замера и отчета по времени выполнения."""

    def __init__(self):
        self.totals = defaultdict(float)
        self.starts = {}

    def start(self, name):
        """Начинает замер времени для задачи с именем 'name'."""
        self.starts[name] = time.time()

    def stop(self, name):
        """Останавливает замер времени для задачи с именем 'name'."""
        if name in self.starts:
            self.totals[name] += time.time() - self.starts[name]
            del self.starts[name]

    def report(self):
        """Выводит итоговый отчет в лог."""
        logger.info("---[ Отчет по времени выполнения Шага 11 ]---")
        total_time = sum(self.totals.values())
        if total_time == 0:
            logger.info("Нет данных для отчета.")
            return

        for name, duration in sorted(self.totals.items()):
            percentage = (duration / total_time) * 100
            logger.info(f"- {name:<40}: {duration:>7.2f} сек ({percentage:>5.1f}%)")
        logger.info("-" * 60)
        logger.info(f"- {'Общее время':<40}: {total_time:>7.2f} сек (100.0%)")
        logger.info("-------------------------------------------------")


def find_max_continuous_travel(travel_diffs):
    """
    Находит максимальное непрерывное движение вверх и вниз в серии данных.
    """
    max_up_travel = 0
    max_down_travel = 0
    current_up_travel = 0
    current_down_travel = 0

    for diff in travel_diffs.dropna():
        if diff > 0:
            current_up_travel += diff
            current_down_travel = 0
        elif diff < 0:
            current_down_travel += abs(diff)
            current_up_travel = 0

        max_up_travel = max(max_up_travel, current_up_travel)
        max_down_travel = max(max_down_travel, current_down_travel)

    return max_up_travel, max_down_travel


def run_step_11():
    """
    Шаг 11: Построение регрессионной модели в скользящем окне для поиска аномалий.
    (ОРКЕСТРАТОР)
    """
    logger.info("---[ Шаг 11: Поиск аномалий (декомпозированная версия) ]---")
    tracker = TimeTracker()

    try:
        # --- ИЗМЕНЕНИЕ: Логика очистки артефактов этого шага ---
        start_step = PIPELINE_CONFIG.get("START_PIPELINE_FROM_STEP", 1)
        if start_step == 11:
            db_handler.cleanup_step_artifacts(step_number=11)
        # ----------------------------------------------------

        # --- ИЗМЕНЕНИЕ: Загрузка данных из БД ---
        tracker.start('1. Загрузка и фильтрация данных из БД')
        object_name = db_handler.get_target_object_name()
        df = db_handler.read_data_from_db(table_or_view_name=object_name)
        logger.info(f"Загружены данные для объекта '{object_name}', содержащие {len(df)} строк.")
        # ----------------------------------------

        # --- ПАРАМЕТРЫ (без изменений) ---
        model_type = PIPELINE_CONFIG.get('STEP_11_MODEL_TYPE', 'linear').lower()
        if model_type == 'neural_network':
            model_params = PIPELINE_CONFIG.get('STEP_11_NN_PARAMS', {})
        else:
            model_params = PIPELINE_CONFIG.get('STEP_11_MODEL_PARAMS', {})
        normalization_type = PIPELINE_CONFIG.get('STEP_11_NORMALIZATION_TYPE', 'none').lower()
        fit_intercept_option = PIPELINE_CONFIG.get('STEP_11_MODEL_FIT_INTERCEPT', True)
        continuous_training = PIPELINE_CONFIG.get('STEP_11_CONTINUOUS_TRAINING', False)
        if continuous_training:
            logger.info("Включен режим непрерывного обучения (warm start).")
        target_col = PIPELINE_CONFIG['STEP_11_TARGET_COLUMN']
        feature_cols = PIPELINE_CONFIG['STEP_11_FEATURE_COLUMNS']
        slips_col = PIPELINE_CONFIG['STEP_11_SLIPS_COLUMN']
        above_bhd_col = PIPELINE_CONFIG['STEP_11_ABOVE_BHD_COLUMN']
        min_window_size = PIPELINE_CONFIG['STEP_11_MIN_WINDOW_SIZE']
        max_window_size = PIPELINE_CONFIG['STEP_11_MAX_WINDOW_SIZE']
        window_step = PIPELINE_CONFIG.get('STEP_11_WINDOW_STEP', 1)
        anomaly_thresholds = PIPELINE_CONFIG.get('STEP_11_ANOMALY_THRESHOLDS', {'low': 4.0})
        logger.info(f"Используются пороги для аномалий: {anomaly_thresholds}")
        min_consecutive = PIPELINE_CONFIG.get('STEP_11_CONSECUTIVE_ANOMALIES_MIN', 1)
        time_col = PIPELINE_CONFIG.get('SORT_COLUMN')
        # Преобразование времени происходит при чтении из БД, но проверим
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col].astype(str).str.replace(',', '.'))

        time_gap_minutes = PIPELINE_CONFIG.get('STEP_11_TIME_GAP_THRESHOLD_MINUTES', 15)
        hook_height_col = PIPELINE_CONFIG.get('STEP_9_HOOK_HEIGHT_COLUMN')
        min_travel = PIPELINE_CONFIG.get('STEP_11_MIN_CONTINUOUS_TRAVEL', 3.0)
        training_flag_col = PIPELINE_CONFIG.get('STEP_11_TRAINING_FLAG_COLUMN', 'Модель_обучалась_флаг')
        exclude_anomalies_from_training = PIPELINE_CONFIG.get('STEP_11_EXCLUDE_ANOMALIES_FROM_TRAINING', True)
        bit_depth_col = PIPELINE_CONFIG.get('STEP_11_BIT_DEPTH_COLUMN')
        enable_depth_check = PIPELINE_CONFIG.get('STEP_11_ENABLE_MIN_DEPTH_CHECK', False)
        min_depth_threshold = PIPELINE_CONFIG.get('STEP_11_MIN_DEPTH_THRESHOLD', 100.0)
        enable_weight_override = PIPELINE_CONFIG.get('STEP_11_ENABLE_WEIGHT_OVERRIDE', False)
        min_weight_override = PIPELINE_CONFIG.get('STEP_11_MIN_WEIGHT_OVERRIDE', 35.0)
        pressure_col = PIPELINE_CONFIG.get('STEP_11_PRESSURE_COLUMN')
        enable_pressure_filter = PIPELINE_CONFIG.get('STEP_11_ENABLE_PRESSURE_FILTER', False)
        pressure_threshold = PIPELINE_CONFIG.get('STEP_11_PRESSURE_THRESHOLD', 200.0)
        enable_stale_check = PIPELINE_CONFIG.get('STEP_11_ENABLE_MODEL_STALE_CHECK', False)
        stale_threshold_minutes = PIPELINE_CONFIG.get('STEP_11_MODEL_STALE_THRESHOLD_MINUTES', 60)
        enable_balancing = PIPELINE_CONFIG.get('STEP_11_ENABLE_DATA_BALANCING', False)
        balancing_col = PIPELINE_CONFIG.get('STEP_11_BALANCING_COLUMN')
        max_stationary_percent = PIPELINE_CONFIG.get('STEP_11_BALANCING_MAX_STATIONARY_PERCENT', 50)
        use_clip = PIPELINE_CONFIG.get('STEP_11_USE_PREDICTION_CLIP', False)
        min_clip = PIPELINE_CONFIG.get('STEP_11_PREDICTION_MIN_CLIP', 0)
        max_clip = PIPELINE_CONFIG.get('STEP_11_PREDICTION_MAX_CLIP', 150)
        enable_error_based_retraining = PIPELINE_CONFIG.get('STEP_11_ENABLE_ERROR_BASED_RETRAINING', False)
        retraining_error_threshold = PIPELINE_CONFIG.get('STEP_11_RETRAINING_ERROR_THRESHOLD', 3.0)
        if enable_error_based_retraining:
            logger.info(
                f"Включена оптимизация: модель будет переобучаться только при MAE > {retraining_error_threshold}")
        # --- КОНЕЦ ПАРАМЕТРОВ ---

        # 2. Фильтрация и разделение на блоки
        df_filtered = df[(df[slips_col] == 0) & (df[above_bhd_col] == 1)].copy()
        logger.info(f"Найдено {len(df_filtered)} точек для анализа.")
        data_blocks = []
        if time_gap_minutes > 0:
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

        # 3. Инициализация столбцов
        df['predicted_weight'] = np.nan
        df['residual'] = np.nan
        df['is_anomaly'] = 0
        df[training_flag_col] = 0
        contribution_cols = [f'contribution_{f}' for f in feature_cols]
        intercept_col_name = 'contribution_intercept'
        for col in contribution_cols + [intercept_col_name]:
            df[col] = np.nan
        tracker.stop('1. Загрузка и фильтрация данных из БД')

        # 4. Инициализация или загрузка модели (без изменений)
        # ... (код инициализации модели оставлен без изменений)
        model_wrapper = None
        preprocessor = None
        interpreter = None
        is_model_fitted = False

        load_enabled = PIPELINE_CONFIG.get('STEP_11_LOAD_MODEL_ENABLED', False)
        if load_enabled:
            logger.info("Попытка загрузить существующую модель...")
            save_dir = PIPELINE_CONFIG.get('STEP_11_MODEL_SAVE_DIR', 'saved_model')
            model_filename = PIPELINE_CONFIG.get('STEP_11_MODEL_FILENAME', 'model.dat')
            preprocessor_filename = PIPELINE_CONFIG.get('STEP_11_PREPROCESSOR_FILENAME', 'preprocessor.dat')
            model_path = os.path.join(save_dir, model_filename)
            preprocessor_path = os.path.join(save_dir, preprocessor_filename)

            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                model_wrapper = model_factory(model_type, model_params, fit_intercept_option,
                                              input_dim=len(feature_cols))
                model_wrapper.load(model_path)
                preprocessor = DataPreprocessor.load(preprocessor_path)
                if preprocessor:
                    is_model_fitted = True
                    interpreter = interpreter_factory(model_wrapper)
                    logger.info("Модель и препроцессор успешно загружены.")
                else:
                    is_model_fitted = False
                    logger.warning("Не удалось загрузить препроцессор, модель будет обучаться с нуля.")
            else:
                logger.warning("Файлы модели и/или препроцессора не найдены. Модель будет обучаться с нуля.")

        if not is_model_fitted:
            model_wrapper = model_factory(model_type, model_params, fit_intercept_option, input_dim=len(feature_cols))
            interpreter = interpreter_factory(model_wrapper)
            preprocessor = DataPreprocessor(normalization_type)

        last_training_time = None

        # 5. Основной цикл (без изменений в логике)
        # ... (весь цикл `for block_num, block_df...` и `for i in tqdm...` остается без изменений)
        total_blocks = len(data_blocks)
        for block_num, block_df in enumerate(data_blocks, 1):
            if len(block_df) < min_window_size:
                continue
            logger.info(f"Обработка блока #{block_num}/{total_blocks}, размер: {len(block_df)} точек.")

            block_df.loc[:, 'predicted_weight'] = np.nan
            block_df.loc[:, 'residual'] = np.nan
            block_df.loc[:, 'is_anomaly'] = 0

            for i in tqdm(range(min_window_size, len(block_df), window_step),
                          desc=f"Анализ блока {block_num}/{total_blocks}"):

                if is_model_fitted and enable_stale_check and last_training_time:
                    current_time = block_df[time_col].iloc[i]
                    time_diff_minutes = (current_time - last_training_time).total_seconds() / 60
                    if time_diff_minutes > stale_threshold_minutes:
                        is_model_fitted = False
                        last_training_time = None
                        if continuous_training:
                            logger.info("Модель устарела, создается новый экземпляр для дообучения.")
                            model_wrapper = model_factory(model_type, model_params, fit_intercept_option,
                                                          input_dim=len(feature_cols))
                            interpreter = interpreter_factory(model_wrapper)

                train_window_full = block_df.iloc[max(0, i - max_window_size):i]

                clean_train_window = train_window_full[
                    train_window_full['is_anomaly'] == 0] if exclude_anomalies_from_training else train_window_full
                if enable_pressure_filter:
                    clean_train_window = clean_train_window[clean_train_window[pressure_col] <= pressure_threshold]

                is_training_data_valid = False
                if len(clean_train_window) >= min_window_size:
                    travel_diffs = clean_train_window[hook_height_col].diff()
                    max_upward, max_downward = find_max_continuous_travel(travel_diffs)
                    if max_upward >= min_travel or max_downward >= min_travel:
                        is_training_data_valid = True

                should_retrain = False
                if is_training_data_valid:
                    if not is_model_fitted:
                        should_retrain = True
                    elif enable_error_based_retraining:
                        start_check_idx = max(0, i - window_step)
                        recent_check_window = block_df.iloc[start_check_idx:i]
                        clean_check_window = recent_check_window[
                            recent_check_window[
                                'is_anomaly'] == 0] if exclude_anomalies_from_training else recent_check_window
                        if enable_pressure_filter:
                            clean_check_window = clean_check_window[
                                clean_check_window[pressure_col] <= pressure_threshold]
                        if not clean_check_window.empty:
                            X_check = clean_check_window[feature_cols]
                            y_check = clean_check_window[target_col]
                            X_check_scaled = preprocessor.transform_features(X_check)
                            predictions_check = model_wrapper.predict(X_check_scaled)
                            predictions_check_denorm = preprocessor.inverse_transform_target(predictions_check)
                            error = np.mean(np.abs(y_check.values - predictions_check_denorm))
                            if error > retraining_error_threshold:
                                logger.info(
                                    f"Ошибка MAE ({error:.2f}) на последнем блоке > порога ({retraining_error_threshold}). Запуск переобучения.")
                                should_retrain = True
                    else:
                        should_retrain = True

                if should_retrain:
                    if not continuous_training:
                        model_wrapper = model_factory(model_type, model_params, fit_intercept_option,
                                                      input_dim=len(feature_cols))
                        interpreter = interpreter_factory(model_wrapper)

                    tracker.start('2.1. Подготовка данных к обучению')
                    training_data = clean_train_window
                    if enable_balancing:
                        training_data = balance_training_data(training_data, balancing_col, max_stationary_percent)
                    X_train = training_data[feature_cols]
                    y_train = training_data[target_col]
                    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
                    tracker.stop('2.1. Подготовка данных к обучению')

                    tracker.start('2.2. Обучение модели (fit)')
                    model_wrapper.fit(X_train_scaled, y_train_scaled)
                    tracker.stop('2.2. Обучение модели (fit)')

                    is_model_fitted = True
                    df.loc[training_data.index, training_flag_col] = 1
                    last_training_time = training_data[time_col].iloc[-1]

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

                tracker.start('3.1. Подготовка данных к предсказанию')
                X_predict = analysis_block[feature_cols]
                X_predict_scaled = preprocessor.transform_features(X_predict)
                tracker.stop('3.1. Подготовка данных к предсказанию')

                tracker.start('3.2. Предсказание (predict)')
                predictions_scaled = model_wrapper.predict(X_predict_scaled)
                predictions = preprocessor.inverse_transform_target(predictions_scaled)
                tracker.stop('3.2. Предсказание (predict)')

                if use_clip:
                    predictions = np.clip(predictions, min_clip, max_clip)

                tracker.start('4. Интерпретация (вклады)')
                if interpreter:
                    X_predict_scaled_df = pd.DataFrame(X_predict_scaled, index=X_predict.index, columns=feature_cols)
                    contributions_df = interpreter.calculate_contributions(model_wrapper, preprocessor,
                                                                           X_predict_scaled_df)
                    if contributions_df is not None:
                        df.loc[analysis_block.index, contributions_df.columns] = contributions_df
                tracker.stop('4. Интерпретация (вклады)')

                tracker.start('5. Пост-обработка и поиск аномалий')
                residuals = analysis_block[target_col] - predictions

                df.loc[analysis_block.index, 'predicted_weight'] = predictions
                df.loc[analysis_block.index, 'residual'] = residuals
                block_df.loc[analysis_block.index, 'predicted_weight'] = predictions
                block_df.loc[analysis_block.index, 'residual'] = residuals

                thresh_low = anomaly_thresholds.get('low', 4.0)
                potential_anomalies_mask = abs(residuals) > thresh_low

                if potential_anomalies_mask.any():
                    potential_mask_in_block = abs(block_df['residual']) > thresh_low
                    grouper = potential_mask_in_block.diff().ne(0).cumsum()
                    anomaly_group_ids = grouper[potential_mask_in_block]

                    if not anomaly_group_ids.empty:
                        block_sizes = anomaly_group_ids.groupby(anomaly_group_ids).transform('size')
                        actual_anomaly_indices = block_sizes[block_sizes >= min_consecutive].index

                        if not actual_anomaly_indices.empty:
                            thresh_med = anomaly_thresholds.get('medium', 8.0)
                            thresh_high = anomaly_thresholds.get('high', 12.0)
                            confirmed_residuals = abs(df.loc[actual_anomaly_indices, 'residual'])

                            anomaly_levels = pd.Series(0, index=actual_anomaly_indices, dtype=int)
                            anomaly_levels[confirmed_residuals >= thresh_high] = 3
                            anomaly_levels[
                                (confirmed_residuals >= thresh_med) & (confirmed_residuals < thresh_high)] = 2
                            anomaly_levels[(confirmed_residuals >= thresh_low) & (confirmed_residuals < thresh_med)] = 1

                            df.loc[actual_anomaly_indices, 'is_anomaly'] = anomaly_levels
                            block_df.loc[actual_anomaly_indices, 'is_anomaly'] = anomaly_levels
                tracker.stop('5. Пост-обработка и поиск аномалий')

        # 6. Сохранение модели (без изменений)
        # ... (код сохранения модели оставлен без изменений)
        save_enabled = PIPELINE_CONFIG.get('STEP_11_SAVE_MODEL_ENABLED', False)
        if save_enabled and is_model_fitted:
            logger.info("Сохранение итоговой модели, препроцессора и конфигурации...")
            save_dir = PIPELINE_CONFIG.get('STEP_11_MODEL_SAVE_DIR', 'saved_model')
            model_filename = PIPELINE_CONFIG.get('STEP_11_MODEL_FILENAME', 'model.dat')
            preprocessor_filename = PIPELINE_CONFIG.get('STEP_11_PREPROCESSOR_FILENAME', 'preprocessor.dat')
            config_filename = PIPELINE_CONFIG.get('STEP_11_CONFIG_FILENAME', 'pipeline_config.json')

            os.makedirs(save_dir, exist_ok=True)

            model_path = os.path.join(save_dir, model_filename)
            preprocessor_path = os.path.join(save_dir, preprocessor_filename)
            config_path = os.path.join(save_dir, config_filename)

            model_wrapper.save(model_path)
            preprocessor.save(preprocessor_path)

            try:
                config_to_save = PIPELINE_CONFIG.copy()
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_to_save, f, ensure_ascii=False, indent=4)
                logger.info(f"Конфигурация пайплайна сохранена в: {config_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении файла конфигурации: {e}")

        elif save_enabled and not is_model_fitted:
            logger.warning("Сохранение модели пропущено, так как модель не была обучена.")

        # --- ИЗМЕНЕНИЕ: Сохранение результатов в отдельную таблицу и создание VIEW ---
        total_anomalies = (df['is_anomaly'] > 0).sum()
        logger.info(f"Найдено {total_anomalies} итоговых аномальных точек.")

        # 7.1. Выделяем только новые, рассчитанные на этом шаге, столбцы
        calculated_cols = [
                              'predicted_weight', 'residual', 'is_anomaly', training_flag_col
                          ] + contribution_cols + [intercept_col_name]

        # Убедимся, что временная колонка - это индекс для сохранения
        df.set_index(time_col, inplace=True)

        # Выбираем только те строки, где есть хоть какие-то расчеты
        results_df = df[calculated_cols].dropna(how='all')

        if not results_df.empty:
            # 7.2. Сохраняем эти столбцы в свою таблицу
            db_handler.write_df_to_table(results_df, table_name="step11")

            # 7.3. Создаем/обновляем VIEW для объединения
            db_handler.create_or_replace_results_view()
        else:
            logger.warning("Нет данных для записи в таблицу результатов Шага 11.")
        # --------------------------------------------------------------------------

        tracker.report()
        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 11: {e}", exc_info=True)
        tracker.report()
        return False
