import os
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def run_step_11():
    """
    Шаг 11: Построение регрессионной модели в скользящем окне для
    поиска аномалий в весе на крюке.
    Работает в один проход с настраиваемым шагом окна.
    """
    logger.info("---[ Шаг 11: Построение модели и поиск аномалий (оптимизированный) ]---")
    try:
        # 1. Загрузка данных и параметров
        step_10_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_10_OUTPUT_FILE'])
        if not os.path.exists(step_10_path):
            logger.error(f"Файл {step_10_path} не найден. Запустите Шаг 10.")
            return False

        df = pd.read_csv(step_10_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_10_path}, содержащий {len(df)} строк.")

        target_col = PIPELINE_CONFIG['STEP_11_TARGET_COLUMN']
        feature_cols = PIPELINE_CONFIG['STEP_11_FEATURE_COLUMNS']
        slips_col = PIPELINE_CONFIG['STEP_11_SLIPS_COLUMN']
        above_bhd_col = PIPELINE_CONFIG['STEP_11_ABOVE_BHD_COLUMN']
        window_size = PIPELINE_CONFIG['STEP_11_WINDOW_SIZE_N']
        window_step = PIPELINE_CONFIG.get('STEP_11_WINDOW_STEP', 1)
        anomaly_threshold = PIPELINE_CONFIG['STEP_11_ANOMALY_THRESHOLD']
        min_consecutive = PIPELINE_CONFIG.get('STEP_11_CONSECUTIVE_ANOMALIES_MIN', 1)

        exclude_all_potential = PIPELINE_CONFIG.get('STEP_11_EXCLUDE_ALL_POTENTIAL_ANOMALIES', False)

        use_clip = PIPELINE_CONFIG.get('STEP_11_USE_PREDICTION_CLIP', False)
        min_clip = PIPELINE_CONFIG.get('STEP_11_PREDICTION_MIN_CLIP', 0)
        max_clip = PIPELINE_CONFIG.get('STEP_11_PREDICTION_MAX_CLIP', 150)

        # 2. Фильтрация данных
        df_filtered = df[(df[slips_col] == 0) & (df[above_bhd_col] == 1)].copy()
        logger.info(f"Найдено {len(df_filtered)} точек данных, удовлетворяющих условиям.")
        if len(df_filtered) <= window_size:
            logger.warning("Недостаточно данных для анализа в скользящем окне.")
            return True

        model = LinearRegression()

        # 3. Однопроходный анализ с динамическим исключением аномалий
        logger.info(f"Запуск анализа со скользящим окном (размер: {window_size}, шаг: {window_step})...")
        if exclude_all_potential:
            logger.info("Режим обучения: Исключаются ВСЕ потенциальные аномалии (шум).")
        else:
            logger.info("Режим обучения: Исключаются только ПОДТВЕРЖДЕННЫЕ аномалии.")

        df['predicted_weight'] = np.nan
        df['is_anomaly'] = 0
        df_filtered['is_anomaly'] = 0
        df_filtered['potential_anomaly_flag'] = 0

        for i in tqdm(range(window_size, len(df_filtered), window_step), desc="Анализ аномалий"):
            train_window_full = df_filtered.iloc[i - window_size:i]

            if exclude_all_potential:
                clean_train_window = train_window_full[train_window_full['potential_anomaly_flag'] == 0]
            else:
                clean_train_window = train_window_full[train_window_full['is_anomaly'] == 0]

            if len(clean_train_window) < len(feature_cols) + 1:
                pass
            else:
                model.fit(clean_train_window[feature_cols], clean_train_window[target_col])

            predict_block = df_filtered.iloc[i: i + window_step]
            X_predict = predict_block[feature_cols]

            predictions = model.predict(X_predict)

            if use_clip:
                predictions = np.clip(predictions, min_clip, max_clip)

            df.loc[predict_block.index, 'predicted_weight'] = predictions
            df.loc[predict_block.index, 'residual'] = predict_block[target_col] - predictions

            potential_anomalies_mask = abs(df.loc[predict_block.index, 'residual']) > anomaly_threshold

            if potential_anomalies_mask.any():
                potential_anomaly_indices = predict_block.index[potential_anomalies_mask]
                df_filtered.loc[potential_anomaly_indices, 'potential_anomaly_flag'] = 1

                pa_indices = potential_anomaly_indices
                grouper = (df['residual'].notna() & (abs(df['residual']) > anomaly_threshold)).diff().ne(0).cumsum()
                block_sizes = df.loc[pa_indices].groupby(grouper.loc[pa_indices])['residual'].transform('size')
                actual_anomaly_indices = block_sizes[block_sizes >= min_consecutive].index

                if not actual_anomaly_indices.empty:
                    df.loc[actual_anomaly_indices, 'is_anomaly'] = 1
                    df_filtered.loc[actual_anomaly_indices, 'is_anomaly'] = 1

        logger.info(f"Найдено {df['is_anomaly'].sum()} итоговых аномальных точек.")

        # 4. Сохранение результата (визуализация перенесена в Шаг 12)
        output_filename = PIPELINE_CONFIG['STEP_11_OUTPUT_FILE']
        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], output_filename)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Итоговый датасет с предсказаниями и аномалиями сохранен в: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 11: {e}", exc_info=True)
        return False

