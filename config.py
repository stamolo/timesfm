import os
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env в корне проекта
load_dotenv()

# --- Настройки подключения к базе данных (загружаются из .env) ---
DB_CONFIG = {
    "DRIVER": os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
    "SERVER": os.getenv("DB_SERVER"),
    "DATABASE": os.getenv("DB_DATABASE"),
    "VIEW_NAME": "[dbo].[v_kharyaginskoe_kha_e1_16_1]",
    "USERNAME": os.getenv("DB_USERNAME"),
    "PASSWORD": os.getenv("DB_PASSWORD")
}

# --- Настройки пайплайна ---
PIPELINE_CONFIG = {
    # Общие
    "OUTPUT_DIR": "output",
    "MODEL_PATH": r"D:\models\checkpoints_k\best_model_kl.pt",
    "START_PIPELINE_FROM_STEP": 11,  # С какого шага начинать пайплайн (1-12)

    # Шаг 1: Выгрузка
    "USE_EXISTING_STEP_1_OUTPUT": False,
    "TOP_N": 1700000,
    "SORT_COLUMN": "Время_204",
    "ALL_COLUMNS": [
        "Время_204", "Вес_на_крюке_28", "Высота_крюка_103",
        "Давление_на_входе_18", "Обороты_ротора_72",
        "Глубина_долота_35", "Глубина_забоя_36"
    ],
    "STEP_1_OUTPUT_FILE": "step_1_extracted.csv",

    # Шаг 2: Добавление отступов (padding)
    "PADDING_SIZE": 50,
    "STEP_2_OUTPUT_FILE": "step_2_padded.csv",

    # Шаг 3: Предсказание
    "PREDICTION_INPUT_COLUMNS": [
        "Вес_на_крюке_28", "Высота_крюка_103",
        "Давление_на_входе_18", "Обороты_ротора_72"
    ],
    "STEP_3_OUTPUT_FILE": "step_3_with_prediction.csv",

    # Шаг 4: Перекодирование состояний
    "STEP_4_INPUT_COLUMN": "клинья_0123",
    "STEP_4_OUTPUT_COLUMN": "клинья_binary",
    "STEP_4_MAPPING_INITIAL": {1: 2, 3: 2},
    "STEP_4_MAPPING_FINAL": {0: 0, 2: 1},
    "STEP_4_OUTPUT_FILE": "step_4_final_dataset.csv",

    # Шаг 5: Расчет глубины инструмента
    "STEP_5_HOOK_HEIGHT_COLUMN": "Высота_крюка_103",
    "STEP_5_SLIPS_COLUMN": "клинья_binary",
    "STEP_5_OUTPUT_COLUMN": "Глубина_инструмента",
    "STEP_5_OUTPUT_FILE": "step_5_with_tool_depth.csv",

    # Шаг 6: Расчет среднего веса по блокам состояний клиньев
    "STEP_6_INPUT_SLIPS_COLUMN": "клинья_binary",
    "STEP_6_INPUT_WEIGHT_COLUMN": "Вес_на_крюке_28",
    "STEP_6_OUTPUT_AVG_WEIGHT_COLUMN": "средний_вес_по_блоку",
    "STEP_6_OUTPUT_FILE": "step_6_block_average_weight.csv",

    # Шаг 7: Продвинутый сброс глубины по анализу блоков
    "STEP_7_PREVIOUS_BLOCKS_N": 30,
    "STEP_7_CURRENT_BLOCKS_Z": 30,
    "STEP_7_MIN_PREV_BLOCK_LENGTH_Y": 60,
    "STEP_7_MIN_BLOCK_LENGTH_M": 60,
    "STEP_7_WEIGHT_DROP_THRESHOLD_X": 10.0,
    "STEP_7_SLIPS_BLOCKS_N": 15,
    "STEP_7_MAX_WEIGHT_ABOVE_SLIPS_R": 10,
    "STEP_7_OUTPUT_COLUMN": "Глубина_инструмента_финал",
    "STEP_7_OUTPUT_FILE": "step_7_final_dataset.csv",
    "STEP_7_BLOCKS_REPORT_FILE": "step_7_blocks_report.csv",

    # Шаг 8: Расчет глубины забоя
    "STEP_8_INPUT_TOOL_DEPTH_COLUMN": "Глубина_инструмента_финал",
    "STEP_8_OUTPUT_BHD_COLUMN": "Глубина_забоя_расчетная",
    "STEP_8_OUTPUT_FILE": "step_8_final_dataset.csv",

    # Шаг 9: Расчет производных и квадратов
    "STEP_9_HOOK_HEIGHT_COLUMN": "Высота_крюка_103",
    "STEP_9_WEIGHT_COLUMN": "Вес_на_крюке_28",
    "STEP_9_BIT_DEPTH_COLUMN": "Глубина_долота_35",
    "STEP_9_OUTPUT_SPEED_COLUMN": "Скорость_инструмента",
    "STEP_9_OUTPUT_WEIGHT_DELTA_COLUMN": "Изменение_веса_на_крюке",
    "STEP_9_OUTPUT_BIT_DEPTH_SQUARED_COLUMN": "Глубина_долота_кв",
    "STEP_9_OUTPUT_SPEED_SQUARED_COLUMN": "Скорость_инструмента_кв",
    "STEP_9_OUTPUT_SIGNED_SPEED_SQUARED_COLUMN": "Скорость_инструмента_кв_знак",
    "STEP_9_OUTPUT_FILE": "step_9_final_with_derivatives.csv",

    # Шаг 10: Расчет параметра "Над забоем"
    "STEP_10_BHD_COLUMN": "Глубина_забоя_36",
    "STEP_10_BIT_DEPTH_COLUMN": "Глубина_долота_35",
    "STEP_10_OUTPUT_COLUMN": "Над забоем, м",
    "STEP_10_CALCULATE_BINARY": True,
    "STEP_10_BINARY_THRESHOLD": 35.0,
    "STEP_10_BINARY_OUTPUT_COLUMN": "Над забоем, м (бинарный)",
    "STEP_10_OUTPUT_FILE": "step_10_final_dataset.csv",

    # Шаг 11: Поиск аномалий веса
    # --- НОВЫЙ ПАРАМЕТР ДЛЯ УПРАВЛЕНИЯ INTERCEPT ---
    # True = модель сама подбирает свободный член (стандартное поведение).
    # False = модель работает без свободного члена, проходя через начало координат.
    "STEP_11_MODEL_FIT_INTERCEPT": False,
    # ------------------------------------------------
    "STEP_11_NORMALIZATION_TYPE": "z_score",  # Варианты: "none", "min_max", "z_score"
    "STEP_11_MODEL_TYPE": "lasso",  # Варианты: "linear", "ridge", "lasso", "elasticnet"
    "STEP_11_MODEL_PARAMS": {
        "ridge": {"alpha": 2.0},      # Параметр регуляризации для Ridge (L2)
        "lasso": {"alpha": 0.1},      # Параметр регуляризации для Lasso (L1)
        "elasticnet": {"alpha": 2.0, "l1_ratio": 0.5} # Параметры для ElasticNet (L1+L2)
    },
    "STEP_11_TARGET_COLUMN": "Вес_на_крюке_28",
    "STEP_11_FEATURE_COLUMNS": [
        "Скорость_инструмента",
        #"Скорость_инструмента_кв_знак",
        "Глубина_долота_35",
        "Давление_на_входе_18",
        "Обороты_ротора_72"
        #"Глубина_долота_кв"
    ],
    "STEP_11_SLIPS_COLUMN": "клинья_0123",
    "STEP_11_ABOVE_BHD_COLUMN": "Над забоем, м (бинарный)",
    "STEP_11_BIT_DEPTH_COLUMN": "Глубина_долота_35",  # Имя столбца для проверки глубины
    "STEP_11_ENABLE_MIN_DEPTH_CHECK": True,  # Включить/выключить проверку
    "STEP_11_MIN_DEPTH_THRESHOLD": 80.0,  # Порог глубины в метрах для начала анализа
    "STEP_11_MIN_WINDOW_SIZE": 1200,  # Мин. кол-во точек для обучения
    "STEP_11_MAX_WINDOW_SIZE": 30000,  # Макс. размер окна, которое смотрит назад
    "STEP_11_WINDOW_STEP": 10,
    "STEP_11_ANOMALY_THRESHOLD": 5.0,
    "STEP_11_CONSECUTIVE_ANOMALIES_MIN": 1,
    "STEP_11_EXCLUDE_ANOMALIES_FROM_TRAINING": False,
    "STEP_11_USE_PREDICTION_CLIP": True,
    "STEP_11_PREDICTION_MIN_CLIP": 0,
    "STEP_11_PREDICTION_MAX_CLIP": 150,
    "STEP_11_TIME_GAP_THRESHOLD_MINUTES": 10,
    "STEP_11_MIN_CONTINUOUS_TRAVEL": 15.0,  # Мин. суммарный ход крюка вверх Или вниз для переобучения
    "STEP_11_ENABLE_WEIGHT_OVERRIDE": True,
    "STEP_11_MIN_WEIGHT_OVERRIDE": 35.0,

    # --- СБРОС УСТАРЕВШЕЙ МОДЕЛИ ---
    "STEP_11_ENABLE_MODEL_STALE_CHECK": False,        # Включить/выключить проверку
    "STEP_11_MODEL_STALE_THRESHOLD_MINUTES": 24000,     # Порог "устаревания" в минутах
    # ---------------------------------------------------

    # --- ПАРАМЕТРЫ ДЛЯ ФИЛЬТРАЦИИ ПО ДАВЛЕНИЮ ---
    "STEP_11_PRESSURE_COLUMN": "Давление_на_входе_18",  # Имя столбца с давлением
    "STEP_11_ENABLE_PRESSURE_FILTER": False,  # Включить/выключить фильтр
    "STEP_11_PRESSURE_THRESHOLD": 20.0,  # Порог давления в атмосферах
    # ---------------------------------------------------

    # --- НОВЫЕ ПАРАМЕТРЫ ДЛЯ БАЛАНСИРОВКИ ДАННЫХ ---
    "STEP_11_ENABLE_DATA_BALANCING": True,  # Включить/выключить балансировку
    "STEP_11_BALANCING_COLUMN": "Скорость_инструмента", # Столбец для проверки баланса
    "STEP_11_BALANCING_MAX_STATIONARY_PERCENT": 15, # Макс. допустимый % точек без движения в обучающей выборке
    # ---------------------------------------------------

    "STEP_11_TRAINING_FLAG_COLUMN": "Модель_обучалась_флаг",
    "STEP_11_OUTPUT_FILE": "step_11_anomaly_detection.csv",

    # Шаг 12: Построение итоговых графиков
    "STEP_12_INPUT_FILE": "step_11_anomaly_detection.csv",
    "STEP_12_PLOT_FILE_PREFIX": "step_12_anomaly_plot",
    "STEP_12_CHUNK_MINUTES": 60,
    "STEP_12_SHOW_CONTRIBUTION_PLOT": True,
    "STEP_12_SKIPPED_CHUNKS_REPORT_FILE": "step_12_skipped_chunks_report.csv",
    "STEP_12_PLOT_SETTINGS": {
        "bit_depth_col": "Глубина_долота_35",
        "bhd_col": "Глубина_забоя_36",
        "hookload_col": "Вес_на_крюке_28",
        "predicted_hookload_col": "predicted_weight",
        "avg_hookload_col": "средний_вес_по_блоку",
        "slips_col": "клинья_0123",
        "pressure_col": "Давление_на_входе_18",
        "rpm_col": "Обороты_ротора_72",
        "hook_height_col": "Высота_крюка_103",
        "training_flag_col": "Модель_обучалась_флаг",
        # --- ОБНОВЛЕННЫЕ ПАРАМЕТРЫ ---
        "fixed_pressure_rpm_scale": True,  # Вкл/выкл фиксированный масштаб для давления/оборотов
        "scale_percentile": 95.0,  # Уровень перцентиля для расчета верхней границы
        "fixed_hook_height_scale": True,  # Вкл/выкл фиксированный масштаб для высоты крюка
        "hook_height_min": 0,  # Нижняя граница для высоты крюка
        "hook_height_max": 35,  # Верхняя граница для высоты крюка
        # --------------------------------
        "colors": {
            "bit_depth": "green",
            "bhd": "black",
            "hookload": "royalblue",
            "predicted_hookload": "orange",
            "avg_hookload": "gold",
            "anomaly": "red",
            "slips": "black",
            "pressure": "maroon",
            "rpm": "purple",
            "hook_height_trained": "black",
            "hook_height_not_trained": "grey"
        },
        "anomaly_marker_s": 15,
        "anomaly_marker_alpha": 0.6
    }
}
