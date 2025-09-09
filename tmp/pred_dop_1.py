import os
import pyodbc
import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Конфигурация параметров расчёта
delta_nad_zaboem = 3  # Порог для Над_забоем_TF
press_threshold = 15   # Порог для Давление_на_входе_18
rotor_threshold = 10   # Порог для Обороты_ротора_72

# Параметры подключения (БД TEST)
server = 'DESKTOP-1CK097D\\SQL_VER_16;'
database = 'TEST'
username = 'sa'
password = '123456'
driver = '{SQL Server}'

# SQL-запрос для извлечения данных
select_query = """
SELECT [Вес_на_крюке_28],
       [Высота_крюка_103],
       [Давление_на_входе_18],
       [Обороты_ротора_72],
       [Время_204],
       [Глубина_долота_35],
       [Глубина_забоя_36],
       [klin_new],
       [klin]
FROM [TEST].[dbo].[v5_t_512c5869be1c40d28a83c4a0a2a5e416]
ORDER BY [Время_204];
"""

def fetch_data():
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
    Добавляем и пересчитываем требуемые столбцы, в том числе [Над_забоем_FL]
    и [Над_забоем_TF], где [Над_забоем_TF] вычисляется на основе [Над_забоем_FL].
    """

    # 1. Рассчитываем столбец 'Над_забоем_FL':
    df['Над_забоем_FL'] = df['Глубина_забоя_36'] - df['Глубина_долота_35']
    logger.info("Столбец 'Над_забоем_FL' рассчитан.")

    # 2. Рассчитываем столбец 'Над_забоем_TF' на основе 'Над_забоем_FL':
    #    Если 'Над_забоем_FL' <= delta_nad_zaboem => 1, иначе 0
    df['Над_забоем_TF'] = df['Над_забоем_FL'].apply(
        lambda x: 1 if x <= delta_nad_zaboem else 0
    )
    logger.info("Столбец 'Над_забоем_TF' рассчитан (по 'Над_забоем_FL').")

    # 3. Рассчитываем столбец 'Давление_TF':
    df['Давление_TF'] = df['Давление_на_входе_18'].apply(lambda x: 1 if x >= press_threshold else 0)
    logger.info("Столбец 'Давление_TF' рассчитан.")

    # 4. Рассчитываем столбец 'Обороты_ротора_TF':
    df['Обороты_ротора_TF'] = df['Обороты_ротора_72'].apply(lambda x: 1 if x >= rotor_threshold else 0)
    logger.info("Столбец 'Обороты_ротора_TF' рассчитан.")

    # 5. Рассчитываем Delta_Глубина_долота
    df['Delta_Глубина_долота'] = df['Глубина_долота_35'].diff().fillna(999)
    logger.info("Столбец 'Delta_Глубина_долота' рассчитан.")

    # 6. Рассчитываем Delta_Глубина_забоя_36
    df['Delta_Глубина_забоя_36'] = df['Глубина_забоя_36'].diff().fillna(999)
    logger.info("Столбец 'Delta_Глубина_забоя_36' рассчитан.")

    return df

def save_processed_data(df):
    """
    Пересоздаёт таблицу v5_t_512c5869be1c40d28a83c4a0a2a5e416
    и записывает в неё все данные (включая новые столбцы).
    """
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            # Если таблица существует, удаляем её и создаём новую
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
                [Над_забоем_FL] FLOAT,
                [Над_забоем_TF] INT,
                [Давление_TF] INT,
                [Обороты_ротора_TF] INT,
                [Delta_Глубина_долота] FLOAT,
                [Delta_Глубина_забоя_36] FLOAT
            );
            """
            cursor.execute(drop_create_sql)
            conn.commit()
            logger.info("Существующая таблица удалена и создана новая.")

            # Подготовка данных для вставки
            # Обратите внимание, что порядок столбцов в INSERT должен совпадать
            # с порядком, в котором мы берём их из DataFrame
            insert_sql = """
            INSERT INTO dbo.v5_t_512c5869be1c40d28a83c4a0a2a5e416
            ([Вес_на_крюке_28], [Высота_крюка_103], [Давление_на_входе_18], [Обороты_ротора_72],
             [Время_204], [Глубина_долота_35], [Глубина_забоя_36], [klin_new], [klin],
             [Над_забоем_FL], [Над_забоем_TF], [Давление_TF], [Обороты_ротора_TF],
             [Delta_Глубина_долота], [Delta_Глубина_забоя_36])
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """
            cursor.fast_executemany = True

            # Формируем нужный порядок столбцов
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
                "Над_забоем_FL",
                "Над_забоем_TF",
                "Давление_TF",
                "Обороты_ротора_TF",
                "Delta_Глубина_долота",
                "Delta_Глубина_забоя_36"
            ]
            df = df[cols_order]

            data_to_insert = df.values.tolist()
            cursor.executemany(insert_sql, data_to_insert)
            conn.commit()
            logger.info("Обработанные данные успешно сохранены в таблицу.")
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
