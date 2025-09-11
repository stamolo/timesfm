import os
import pandas as pd
import pyodbc
import logging
from config import DB_CONFIG, PIPELINE_CONFIG

# Получаем настроенный логгер из основного файла
logger = logging.getLogger(__name__)


def run_step_1():
    """
    Шаг 1: Подключение к базе данных и выгрузка данных из view.
    Возвращает True в случае успеха и False в случае ошибки.
    """
    logger.info("---[ Шаг 1: Выгрузка данных из БД ]---")
    try:
        conn_str = (
            f"DRIVER={{{DB_CONFIG['DRIVER']}}};"
            f"SERVER={DB_CONFIG['SERVER']};DATABASE={DB_CONFIG['DATABASE']};"
            f"UID={DB_CONFIG['USERNAME']};PWD={DB_CONFIG['PASSWORD']};"
            "Trusted_Connection=yes;"
        )

        select_clause = "SELECT"
        top_n = PIPELINE_CONFIG.get("TOP_N")
        if isinstance(top_n, int) and top_n > 0:
            select_clause += f" TOP ({top_n})"
            logger.info(f"Будет выгружено TOP {top_n} записей.")

        columns_list = PIPELINE_CONFIG['ALL_COLUMNS']
        columns_str = f" {', '.join([f'[{col}]' for col in columns_list])}"
        from_clause = f" FROM {DB_CONFIG['VIEW_NAME']}"

        order_by_clause = ""
        sort_column = PIPELINE_CONFIG.get("SORT_COLUMN")
        if sort_column and sort_column in columns_list:
            order_by_clause = f" ORDER BY [{sort_column}] ASC"
            logger.info(f"Данные будут отсортированы по '{sort_column}' ASC (хронологический порядок).")

        query = f"{select_clause}{columns_str}{from_clause}{order_by_clause}"

        logger.info(f"Подключение к {DB_CONFIG['SERVER']}...")
        with pyodbc.connect(conn_str) as conn:
            logger.info(f"Выполнение запроса: {query}")
            df = pd.read_sql(query, conn)
            logger.info(f"Загружено {len(df)} строк.")

        output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_1_OUTPUT_FILE'])
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Данные успешно сохранены в: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 1: {e}", exc_info=True)
        return False
