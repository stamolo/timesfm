import os
import pandas as pd
import pyodbc
import logging
from config import DB_CONFIG, PIPELINE_CONFIG

# Получаем настроенный логгер из основного файла
logger = logging.getLogger(__name__)


def run_step_1():
    """
    Шаг 1: Подключение к БД и выгрузка данных.
    Приоритетно использует фильтр по датам. Если он не задан,
    использует OFFSET/FETCH для пропуска и ограничения строк.
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

        # --- Формирование SQL запроса ---
        select_clause = "SELECT"
        columns_list = PIPELINE_CONFIG['ALL_COLUMNS']
        columns_str = f" {', '.join([f'[{col}]' for col in columns_list])}"
        from_clause = f" FROM {DB_CONFIG['VIEW_NAME']}"
        where_clause = ""
        order_by_clause = ""
        pagination_clause = ""

        # --- Получение параметров из конфига ---
        sort_column = PIPELINE_CONFIG.get("SORT_COLUMN")
        rows_to_skip = PIPELINE_CONFIG.get("ROWS_TO_SKIP_INITIAL", 0)
        limit_rows = PIPELINE_CONFIG.get("TOP_N")

        date_filter_col = PIPELINE_CONFIG.get("DATE_FILTER_COLUMN")
        start_date = PIPELINE_CONFIG.get("START_DATE")
        end_date = PIPELINE_CONFIG.get("END_DATE")

        # --- Логика фильтрации: приоритет у дат ---
        if date_filter_col and (start_date or end_date):
            logger.info("Обнаружена конфигурация фильтра по дате. Приоритет будет отдан ей.")
            conditions = []
            if start_date:
                # --- ИЗМЕНЕНИЕ: Использование канонического формата ODBC для дат ---
                conditions.append(f"[{date_filter_col}] >= {{ts '{start_date}'}}")
                logger.info(f"Условие: дата начала >= {start_date}")
            if end_date:
                # --- ИЗМЕНЕНИЕ: Использование канонического формата ODBC для дат ---
                conditions.append(f"[{date_filter_col}] <= {{ts '{end_date}'}}")
                logger.info(f"Условие: дата окончания <= {end_date}")

            where_clause = " WHERE " + " AND ".join(conditions)

            # Сортировка все еще важна, если она указана
            if sort_column:
                order_by_clause = f" ORDER BY [{sort_column}] ASC"
                logger.info(f"Данные будут отсортированы по '{sort_column}' ASC.")

        # --- Логика пагинации (если фильтр по датам не используется) ---
        else:
            logger.info("Фильтр по дате не настроен. Используется логика TOP_N / OFFSET.")
            if sort_column:
                order_by_clause = f" ORDER BY [{sort_column}] ASC"
                logger.info(f"Данные будут отсортированы по '{sort_column}' ASC (хронологический порядок).")

                offset_val = int(rows_to_skip) if isinstance(rows_to_skip, int) and rows_to_skip > 0 else 0

                pagination_clause = f" OFFSET {offset_val} ROWS"
                if offset_val > 0:
                    logger.info(f"Будет пропущено первых {offset_val} строк на стороне БД (OFFSET).")

                if isinstance(limit_rows, int) and limit_rows > 0:
                    pagination_clause += f" FETCH NEXT {limit_rows} ROWS ONLY"
                    logger.info(f"Будет выгружено не более {limit_rows} строк (FETCH).")

            else:
                if isinstance(rows_to_skip, int) and rows_to_skip > 0:
                    logger.error(
                        "Для пропуска строк (ROWS_TO_SKIP_INITIAL > 0) необходимо указать 'SORT_COLUMN' в config.py, "
                        "так как OFFSET требует ORDER BY."
                    )
                    return False

                if isinstance(limit_rows, int) and limit_rows > 0:
                    select_clause += f" TOP ({limit_rows})"
                    logger.info(f"Будет выгружено TOP {limit_rows} записей (сортировка не задана).")

        query = f"{select_clause}{columns_str}{from_clause}{where_clause}{order_by_clause}{pagination_clause}"

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

