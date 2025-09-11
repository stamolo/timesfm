import os
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env в корне проекта
load_dotenv()

# --- Настройки подключения к базе данных (загружаются из .env) ---
DB_CONFIG = {
    "DRIVER": os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
    "SERVER": os.getenv("DB_SERVER"),
    "DATABASE": os.getenv("DB_DATABASE"),
    "VIEW_NAME": "[dbo].[v4_t_00c42aaff5f54cdeb352ae32e58260fb]",  # Имя view возвращено в конфиг
    "USERNAME": os.getenv("DB_USERNAME"),
    "PASSWORD": os.getenv("DB_PASSWORD")
}

# --- Настройки пайплайна ---
PIPELINE_CONFIG = {
    # Общие
    "OUTPUT_DIR": "output",
    "MODEL_PATH": r"D:\models\checkpoints_k\best_model.pt",

    # Шаг 1: Выгрузка
    "TOP_N": 500000,  # Количество записей для выгрузки (None для всех)
    "SORT_COLUMN": "Время_204",
    "ALL_COLUMNS": [
        "Время_204", "Вес_на_крюке_28", "Высота_крюка_103",
        "Давление_на_входе_18", "Обороты_ротора_72",
        "Глубина_долота_35", "Глубина_забоя_36"
    ],
    "STEP_1_OUTPUT_FILE": "step_1_extracted.csv",

    # Шаг 2: Добавление отступов (padding)
    "PADDING_SIZE": 50,  # Количество строк для добавления в начале и конце
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
    "STEP_6_INPUT_SLIPS_COLUMN": "клинья_binary",  # Входной столбец с состоянием клиньев
    "STEP_6_INPUT_WEIGHT_COLUMN": "Вес_на_крюке_28",  # Входной столбец с весом
    "STEP_6_OUTPUT_AVG_WEIGHT_COLUMN": "средний_вес_по_блоку",  # Имя нового столбца
    "STEP_6_OUTPUT_FILE": "step_6_block_average_weight.csv"  # Имя итогового файла
}

