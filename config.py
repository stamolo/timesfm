import os
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env в корне проекта
load_dotenv()

# --- Настройки подключения к базе данных (загружаются из .env) ---
DB_CONFIG = {
    "DRIVER": os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
    "SERVER": os.getenv("DB_SERVER"),
    "DATABASE": os.getenv("DB_DATABASE"),
    "VIEW_NAME": "[dbo].[v4_t_9bf1109f4ff141ac94b95b082947f387]",  # Имя view возвращено в конфиг
    "USERNAME": os.getenv("DB_USERNAME"),
    "PASSWORD": os.getenv("DB_PASSWORD")
}

# --- Настройки пайплайна ---
PIPELINE_CONFIG = {
    # Общие
    "OUTPUT_DIR": "output",
    "MODEL_PATH": r"D:\models\checkpoints_k\best_model.pt",

    # Шаг 1: Выгрузка
    "TOP_N": 1000000,
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
    "STEP_4_MAPPING_INITIAL": {1: 0, 3: 0},
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
    # --- Условие 1: Сравнение рабочих блоков между собой ---
    "STEP_7_PREVIOUS_BLOCKS_N": 5,
    "STEP_7_CURRENT_BLOCKS_Z": 3,
    "STEP_7_MIN_PREV_BLOCK_LENGTH_Y": 60,
    "STEP_7_MIN_BLOCK_LENGTH_M": 60,
    "STEP_7_WEIGHT_DROP_THRESHOLD_X": 10.0,

    # --- Условие 2: Сравнение рабочих блоков с блоками на клиньях ---
    # N_slips: количество ПОСЛЕДНИХ блоков на клиньях для расчета "опорного" веса
    "STEP_7_SLIPS_BLOCKS_N": 5,
    # R: макс. допустимое превышение текущего рабочего веса над "опорным" (в тоннах)
    "STEP_7_MAX_WEIGHT_ABOVE_SLIPS_R": 10,

    # --- Общие настройки Шага 7 ---
    "STEP_7_OUTPUT_COLUMN": "Глубина_инструмента_финал",
    "STEP_7_OUTPUT_FILE": "step_7_final_dataset.csv",
    "STEP_7_BLOCKS_REPORT_FILE": "step_7_blocks_report.csv",

    # Шаг 8: Расчет глубины забоя
    # Входной столбец с финальной глубиной инструмента из Шага 7
    "STEP_8_INPUT_TOOL_DEPTH_COLUMN": "Глубина_инструмента_финал",
    # Имя нового столбца с расчетной глубиной забоя
    "STEP_8_OUTPUT_BHD_COLUMN": "Глубина_забоя_расчетная",
    # Имя итогового файла пайплайна
    "STEP_8_OUTPUT_FILE": "step_8_final_dataset.csv"
}

