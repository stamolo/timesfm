import os
import pandas as pd
import logging
from config import DB_CONFIG, PIPELINE_CONFIG

# Импортируем драйверы для баз данных
try:
    import pyodbc
except ImportError:
    pyodbc = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None

# Получаем настроенный логгер из основного файла
logger = logging.getLogger(__name__)


def run_step_1():
    """
    Шаг 1: Подключение к базе данных и выгрузка данных.
    Поддерживает SQL Server и PostgreSQL, а также опциональную фильтрацию.
    Либо использование существующего файла, если включена опция.
    Возвращает True в случае успеха и False в случае ошибки.
    """
    logger.info("---[ Шаг 1: Выгрузка данных ]---")

    use_existing = PIPELINE_CONFIG.get("USE_EXISTING_STEP_1_OUTPUT", False)
    output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_1_OUTPUT_FILE'])

    if use_existing:
        logger.info("Опция 'USE_EXISTING_STEP_1_OUTPUT' включена.")
        if os.path.exists(output_path):
            logger.info(f"Используется существующий файл: {output_path}")
            return True
        else:
            logger.error(
                f"Файл {output_path} не найден, но опция 'USE_EXISTING_STEP_1_OUTPUT' включена. Пайплайн остановлен.")
            return False

    try:
        db_type = DB_CONFIG.get("TYPE")
        df = None
        params = None
        query = ""

        # --- Общие параметры для запроса ---
        columns_list = PIPELINE_CONFIG['ALL_COLUMNS']
        top_n = PIPELINE_CONFIG.get("TOP_N")
        sort_column = PIPELINE_CONFIG.get("SORT_COLUMN")
        filter_col = DB_CONFIG.get("FILTER_COLUMN_NAME")
        filter_val = DB_CONFIG.get("FILTER_COLUMN_VALUE")

        # --- Логика для SQL Server ---
        if db_type == "SQL_SERVER":
            if not pyodbc:
                logger.error("Библиотека 'pyodbc' не установлена: pip install pyodbc")
                return False

            conn_str = (
                f"DRIVER={{{DB_CONFIG['DRIVER']}}};"
                f"SERVER={DB_CONFIG['SERVER']};DATABASE={DB_CONFIG['DATABASE']};"
                f"UID={DB_CONFIG['USERNAME']};PWD={DB_CONFIG['PASSWORD']};"
                "Trusted_Connection=yes;"
            )

            select_clause = "SELECT"
            if isinstance(top_n, int) and top_n > 0:
                select_clause += f" TOP ({top_n})"

            columns_str = ", ".join([f'[{col}]' for col in columns_list])
            from_clause = f"FROM {DB_CONFIG['VIEW_NAME']}"
            where_clause = ""
            order_by_clause = ""

            if filter_col and filter_val:
                where_clause = f'WHERE [{filter_col}] = ?'
                params = (filter_val,)
                logger.info(f'Применен фильтр: WHERE [{filter_col}] = \'{filter_val}\'')

            if sort_column and sort_column in columns_list:
                order_by_clause = f"ORDER BY [{sort_column}] ASC"

            query = f"{select_clause} {columns_str} {from_clause} {where_clause} {order_by_clause}"

            logger.info(f"Подключение к SQL Server: {DB_CONFIG['SERVER']}...")
            with pyodbc.connect(conn_str) as conn:
                logger.info(f"Выполнение запроса: {query}")
                df = pd.read_sql(query, conn, params=params)

        # --- Логика для PostgreSQL ---
        elif db_type == "POSTGRES":
            if not psycopg2:
                logger.error("Библиотека 'psycopg2' не установлена: pip install psycopg2-binary")
                return False

            conn_str = (
                f"dbname='{DB_CONFIG['DATABASE']}' "
                f"user='{DB_CONFIG['USER']}' "
                f"host='{DB_CONFIG['HOST']}' "
                f"password='{DB_CONFIG['PASSWORD']}' "
                f"port='{DB_CONFIG['PORT']}'"
            )

            columns_str = ', '.join([f'"{col}"' for col in columns_list])

            # ИСПРАВЛЕНИЕ: Корректная обработка имени таблицы со схемой
            view_name_full = DB_CONFIG["VIEW_NAME"]
            if '.' in view_name_full:
                schema, table = view_name_full.split('.', 1)
                from_clause = f'FROM "{schema}"."{table}"'
            else:
                from_clause = f'FROM "{view_name_full}"'

            query = f'SELECT {columns_str} {from_clause}'

            if filter_col and filter_val:
                query += f' WHERE "{filter_col}" = %s'
                params = (filter_val,)
                logger.info(f'Применен фильтр: WHERE "{filter_col}" = \'{filter_val}\'')

            if sort_column and sort_column in columns_list:
                query += f' ORDER BY "{sort_column}" ASC'

            if isinstance(top_n, int) and top_n > 0:
                query += f" LIMIT {top_n}"

            logger.info(f"Подключение к PostgreSQL: {DB_CONFIG['HOST']}...")
            with psycopg2.connect(conn_str) as conn:
                logger.info(f"Выполнение запроса: {query}")
                df = pd.read_sql(query, conn, params=params)

        else:
            logger.error(f"Неподдерживаемый тип базы данных: {db_type}. Проверьте DB_TYPE в .env файле.")
            return False

        if df is not None:
            logger.info(f"Загружено {len(df)} строк.")
            df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
            logger.info(f"Данные успешно сохранены в: {output_path}")
            return True
        else:
            logger.warning("Не удалось загрузить данные (df is None).")
            return False

    except Exception as e:
        logger.error(f"Ошибка на Шаге 1: {e}", exc_info=True)
        return False

