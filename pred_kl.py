import os
import pyodbc
import pandas as pd
import subprocess
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Параметры подключения к исходной БД
server = 'DESKTOP-1CK097D\\SQL_VER_16;'
database = 'MeretoyahaRV'
username = 'sa'
password = '123456'
driver = '{SQL Server}'

# SQL-запрос для извлечения данных
select_query = """
SELECT  TOP (120000)
      [Вес_на_крюке_28],
      [Высота_крюка_103],
      [Давление_на_входе_18],
      [Обороты_ротора_72],
      [Время_204],
      [Глубина_долота_35],
      [Глубина_забоя_36]
FROM [MeretoyahaRV].[dbo].[v4_t_512c5869be1c40d28a83c4a0a2a5e416]
ORDER BY [Время_204];
"""


def fetch_data():
    try:
        conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        with pyodbc.connect(conn_str) as conn:
            logger.info("Подключение к базе данных %s успешно установлено.", database)
            df = pd.read_sql(select_query, conn)
            logger.info("Запрос выполнен, получено строк: %d", len(df))
        return df
    except Exception as e:
        logger.error("Ошибка подключения или выполнения запроса: %s", e)
        raise


def run_pred256(input_csv, output_csv):
    try:
        # Вызов внешнего скрипта pred256_3.py с передачей аргументов
        cmd = ["python", "pred256_3.py", "--csv", input_csv, "--output", output_csv]
        logger.info("Запуск внешнего скрипта: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Скрипт pred256_3.py успешно завершён.")
    except subprocess.CalledProcessError as e:
        logger.error("Ошибка при выполнении pred256_3.py: %s", e)
        raise


def process_predictions(input_csv='temp_input.csv', output_csv='temp_output.csv'):
    # Вызываем скрипт обработки (pred256_3.py)
    run_pred256(input_csv, output_csv)
    try:
        df_pred = pd.read_csv(output_csv, sep=";", decimal=",")
    except Exception as e:
        logger.error("Ошибка чтения выходного CSV от pred256_3.py: %s", e)
        raise

    if "predicted_label" not in df_pred.columns:
        raise ValueError("Ожидаемый столбец 'predicted_label' не найден в выходном CSV.")

    # Удаляем строки, где predicted_label (будущий klin_new) отсутствует
    df_pred.dropna(subset=["predicted_label"], inplace=True)
    logger.info("Удалены строки с отсутствующими значениями в 'predicted_label'.")

    # Переименовываем столбец predicted_label в klin_new
    df_pred.rename(columns={"predicted_label": "klin_new"}, inplace=True)
    logger.info("Столбец 'predicted_label' переименован в 'klin_new'.")

    # Добавляем новый столбец klin: если klin_new == 0, то 0, иначе 1
    df_pred["klin"] = df_pred["klin_new"].apply(lambda x: 0 if x == 0 else 1)
    logger.info("Столбец 'klin' добавлен согласно правилу.")
    return df_pred[["klin_new", "klin"]]


def save_to_target_db(df_final):
    # Параметры подключения к целевой БД "TEST"
    target_database = "TEST"
    target_conn_str = f"DRIVER={driver};SERVER={server};DATABASE={target_database};UID={username};PWD={password}"
    try:
        with pyodbc.connect(target_conn_str) as conn:
            logger.info("Подключение к базе данных %s успешно установлено.", target_database)
            cursor = conn.cursor()
            # Если таблица уже существует, удаляем её и создаём новую
            create_table_sql = """
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
                [klin] INT
            );
            """
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info(
                "Таблица v5_t_512c5869be1c40d28a83c4a0a2a5e416 создана (существующая таблица удалена) в базе данных %s.",
                target_database)
            # Подготовка данных для вставки
            insert_sql = """
            INSERT INTO dbo.v5_t_512c5869be1c40d28a83c4a0a2a5e416
            ([Вес_на_крюке_28], [Высота_крюка_103], [Давление_на_входе_18], [Обороты_ротора_72],
             [Время_204], [Глубина_долота_35], [Глубина_забоя_36], [klin_new], [klin])
            VALUES (?,?,?,?,?,?,?,?,?)
            """
            cursor.fast_executemany = True
            data_to_insert = df_final.values.tolist()
            cursor.executemany(insert_sql, data_to_insert)
            conn.commit()
            logger.info("Данные успешно вставлены в таблицу.")
    except Exception as e:
        logger.error("Ошибка при сохранении данных в целевую базу: %s", e)
        raise


def main():
    try:
        # Шаг 1. Извлечение данных из исходной базы
        df_all = fetch_data()
    except Exception as e:
        logger.error("Процесс остановлен на этапе извлечения данных.")
        return

    # Сохраняем подмножество столбцов для обработки (только 4 нужных колонки)
    subset_columns = ['Вес_на_крюке_28', 'Высота_крюка_103', 'Давление_на_входе_18', 'Обороты_ротора_72']
    df_subset = df_all[subset_columns]
    input_csv = "temp_input.csv"
    output_csv = "temp_output.csv"
    try:
        df_subset.to_csv(input_csv, index=False, sep=";", decimal=",")
        logger.info("Временный CSV для обработки создан: %s", input_csv)
    except Exception as e:
        logger.error("Ошибка записи входного CSV: %s", e)
        return

    # Шаг 2. Обработка через внешний скрипт pred256_3.py
    try:
        df_pred = process_predictions(input_csv, output_csv)
    except Exception as e:
        logger.error("Процесс остановлен на этапе обработки предсказаний.")
        return

    # Фильтруем исходные данные, оставляя только строки, для которых предсказания присутствуют
    df_all = df_all.loc[df_pred.index]

    # Шаг 3. Объединяем исходные данные с результатами предсказания
    if len(df_all) != len(df_pred):
        logger.error("Количество строк предсказаний (%d) не совпадает с количеством исходных данных (%d).",
                     len(df_pred), len(df_all))
        return

    df_all["klin_new"] = df_pred["klin_new"].values
    df_all["klin"] = df_pred["klin"].values

    # Шаг 4. Сохранение итоговых данных в целевую базу "TEST"
    try:
        save_to_target_db(df_all)
    except Exception as e:
        logger.error("Процесс остановлен на этапе сохранения в целевую базу.")
        return

    # Очистка временных файлов
    for f in [input_csv, output_csv]:
        try:
            if os.path.exists(f):
                os.remove(f)
                logger.info("Временный файл %s удалён.", f)
        except Exception as e:
            logger.warning("Ошибка при удалении файла %s: %s", f, e)

    logger.info("Процесс завершён успешно.")


if __name__ == "__main__":
    main()
