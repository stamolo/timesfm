import os
import pyodbc
import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Параметры подключения (БД TEST)
server = 'DESKTOP-1CK097D\\SQL_VER_16;'
database = 'TEST'
username = 'sa'
password = '123456'
driver = '{SQL Server}'

# SQL-запрос для извлечения данных
select_query = """
SELECT
       [Вес_на_крюке_28],
       [Высота_крюка_103],
       [Давление_на_входе_18],
       [Обороты_ротора_72],
       [Время_204],
       [Глубина_долота_35],
       [Глубина_забоя_36],
       [klin_new],
       [klin],
       [Над_забоем_TF],
       [Давление_TF],
       [Обороты_ротора_TF],
       [Над_забоем_FL],
       [Delta_Глубина_долота],
       [Delta_Глубина_забоя_36]
FROM [TEST].[dbo].[v5_t_512c5869be1c40d28a83c4a0a2a5e416]
ORDER BY [Время_204];
"""

def fetch_data():
    """
    Извлекает все данные из таблицы v5_t_512c5869be1c40d28a83c4a0a2a5e416 (БД TEST).
    """
    try:
        conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        with pyodbc.connect(conn_str) as conn:
            logger.info("Подключение к базе данных %s успешно установлено.", database)
            df = pd.read_sql(select_query, conn)
            logger.info("Извлечено строк: %d", len(df))
        return df
    except Exception as e:
        logger.error("Ошибка подключения или выполнения запроса: %s", e)
        raise

def process_data(df):
    """
    Рассчитывает новый столбец [spo] по заданным правилам.
    """

    def calc_spo(row):
        # Правила:
        # 1) [klin] = 1 -> spo = 999
        # 2) [klin] = 0 и [Над_забоем_TF] = 1 -> spo = 999
        # 3) [klin] = 0 и [Над_забоем_TF] = 0 и Delta_Глубина_долота < 0 -> spo = 1
        # 4) [klin] = 0 и [Над_забоем_TF] = 0 и Delta_Глубина_долота > 0 -> spo = -1
        # 5) [klin] = 0 и [Над_забоем_TF] = 0 и Delta_Глубина_долота = 0 -> spo = 0

        if row["klin"] == 1:
            return 999
        elif row["klin"] == 0 and row["Над_забоем_TF"] == 1:
            return 999
        elif row["klin"] == 0 and row["Над_забоем_TF"] == 0:
            delta = row["Delta_Глубина_долота"]
            if delta < 0:
                return 1
            elif delta > 0:
                return -1
            else:  # delta == 0
                return 0
        else:
            # На всякий случай, если какое-то значение не попало под условия (не должно случиться)
            return None

    df["spo"] = df.apply(calc_spo, axis=1)
    logger.info("Столбец 'spo' рассчитан.")
    return df

def save_processed_data(df):
    """
    Удаляет существующую таблицу v5_t_512c5869be1c40d28a83c4a0a2a5e416 (если есть),
    создаёт новую и вставляет в неё все данные (старые столбцы + [spo]).
    """
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            # Если таблица существует, удаляем её и создаём заново (16 столбцов, включая spo)
            drop_create_sql = """
            IF OBJECT_ID('dbo.v5_t_512c5869be1c40d28a83c4a0a2a5e416', 'U') IS NOT NULL
                DROP TABLE dbo.v5_t_512c5869be1c40d28a83c4a0a2a5e416;
            CREATE TABLE dbo.v5_t_512c5869be1c40d28a83c4a0a2a5e416 (
                [Вес_на_крюке_28] FLOAT,
                [Высота_крюка_103] FLOAT,
                [Давление_на_входе_18] FLOAT,
                [Обороты_ротора_72] FLOAT,
                [Время_204] DATETIME,
                [Глубина_долота_35] FLOAT,
                [Глубина_забоя_36] FLOAT,
                [klin_new] FLOAT,
                [klin] INT,
                [Над_забоем_TF] INT,
                [Давление_TF] INT,
                [Обороты_ротора_TF] INT,
                [Над_забоем_FL] FLOAT,
                [Delta_Глубина_долота] FLOAT,
                [Delta_Глубина_забоя_36] FLOAT,
                [spo] FLOAT
            );
            """
            cursor.execute(drop_create_sql)
            conn.commit()
            logger.info("Существующая таблица удалена и создана новая (с полем [spo]).")

            # Подготовка данных для вставки
            # Внимание! Порядок столбцов в INSERT должен соответствовать DataFrame.values.tolist()
            insert_sql = """
            INSERT INTO dbo.v5_t_512c5869be1c40d28a83c4a0a2a5e416
            ([Вес_на_крюке_28],
             [Высота_крюка_103],
             [Давление_на_входе_18],
             [Обороты_ротора_72],
             [Время_204],
             [Глубина_долота_35],
             [Глубина_забоя_36],
             [klin_new],
             [klin],
             [Над_забоем_TF],
             [Давление_TF],
             [Обороты_ротора_TF],
             [Над_забоем_FL],
             [Delta_Глубина_долота],
             [Delta_Глубина_забоя_36],
             [spo])
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """
            cursor.fast_executemany = True

            # Убедимся, что в df колонки идут в нужном порядке:
            cols_order = [
                "Вес_на_крюке_28",
                "Высота_крюка_103",
                "Давление_на_входе_18",
                "Обороты_ротора_72",
                "Время_204",
                "Глубина_долота_35",
                "Глубина_забоя_36",
                "klin_new",
                "klin",
                "Над_забоем_TF",
                "Давление_TF",
                "Обороты_ротора_TF",
                "Над_забоем_FL",
                "Delta_Глубина_долота",
                "Delta_Глубина_забоя_36",
                "spo"
            ]
            df = df[cols_order]

            data_to_insert = df.values.tolist()
            cursor.executemany(insert_sql, data_to_insert)
            conn.commit()
            logger.info("Данные (включая столбец 'spo') успешно сохранены в таблицу.")
    except Exception as e:
        logger.error("Ошибка при сохранении данных: %s", e)
        raise

def main():
    try:
        df = fetch_data()
    except Exception as e:
        logger.error("Процесс остановлен на этапе извлечения данных.")
        return

    try:
        df_processed = process_data(df)
    except Exception as e:
        logger.error("Процесс остановлен на этапе обработки данных.")
        return

    try:
        save_processed_data(df_processed)
    except Exception as e:
        logger.error("Процесс остановлен на этапе сохранения данных в базу.")
        return

    logger.info("Скрипт успешно завершён.")

if __name__ == "__main__":
    main()
