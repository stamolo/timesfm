import pyodbc
import csv
from tqdm import tqdm

# Настройки экспорта в CSV
CSV_SETTINGS = {"sep": ";", "decimal": ","}

# Конфигурация подключения к SQL Server
server = 'DESKTOP-1CK097D\\SQL_VER_16;'
database = 'MeretoyahaRV'
username = 'sa'
password = '123456'
driver = '{SQL Server}'


def format_number(num):
    """
    Форматирует числовое значение в строку с учетом настроек десятичного разделителя.
    Если разделитель отличается от точки, заменяет его.
    """
    # Форматируем число с тремя знаками после запятой
    formatted = f"{num:.3f}"
    if CSV_SETTINGS["decimal"] != ".":
        formatted = formatted.replace(".", CSV_SETTINGS["decimal"])
    return formatted


def generate_sql_query_for_table_with_cte(full_table_name):
    """
    Формирует SQL-запрос для указанной таблицы с использованием конструкции CTE.
    full_table_name имеет вид 'schema.table', поэтому происходит его разделение
    для корректного формирования FROM [MeretoyahaRV].[schema].[table].
    """
    try:
        schema, table_name = full_table_name.split('.')
    except ValueError:
        raise ValueError(f"Неверный формат имени таблицы: {full_table_name}")

    return f"""
    WITH CTE AS (
      SELECT TOP (2000000)
             [18] AS [Давление на входе],
             ([28] / 9806.65) AS [Вес на крюке],
             [35] AS [Глубина долота],
             [36] AS [Глубина забоя],
             [68] AS [Нагрузка на долото],
             ([72] / 0.1047197) AS [Обороты ротора],
             [103] AS [Высота крюка],
             [204],
             CASE 
               WHEN [m_41] = 1 THEN 'Время бурения свечи'
               WHEN [m_42] = 1 THEN 'Время от снятия нагрузки до следующей нагрузки'
               WHEN [m_43] = 1 THEN 'Время до наращивания'
               WHEN [m_44] = 1 THEN 'Время наращивания'
               WHEN [m_45] = 1 THEN 'Время после наращивания'
               WHEN [m_46] = 1 THEN 'Время между клиньями'
               ELSE NULL
             END AS [Метка класса время],
             CASE 
               WHEN [m_41] = 1 THEN 41
               WHEN [m_42] = 1 THEN 42
               WHEN [m_43] = 1 THEN 43
               WHEN [m_44] = 1 THEN 44
               WHEN [m_45] = 1 THEN 45
               WHEN [m_46] = 1 THEN 46
               ELSE NULL
             END AS [Метка класса время (число)],
             CASE 
               WHEN [m_10] = 1 THEN 'Роторное бурение'
               WHEN [m_11] = 1 THEN 'Направленное бурение'
               WHEN [m_12] = 1 THEN 'Спуск с проработкой'
               WHEN [m_13] = 1 THEN 'Подъем с проработкой'
               WHEN [m_14] = 1 THEN 'Спуск с промывкой'
               WHEN [m_15] = 1 THEN 'Подъем с промывкой'
               WHEN [m_16] = 1 THEN 'Промывка в покое'
               WHEN [m_17] = 1 THEN 'Промывка с вращением'
               WHEN [m_18] = 1 THEN 'Спуск с вращением'
               WHEN [m_19] = 1 THEN 'Подъем с вращением'
               WHEN [m_20] = 1 THEN 'Вращение без циркуляции'
               WHEN [m_30] = 1 THEN 'Спуск в скважину'
               WHEN [m_31] = 1 THEN 'Подъем из скважины'
               WHEN [m_32] = 1 THEN 'Неподвижное состояние'
               WHEN [m_33] = 1 THEN 'Удержание в клиньях'
               ELSE NULL
             END AS [Метка класса процесс],
             CASE 
               WHEN [m_10] = 1 THEN 0
               WHEN [m_11] = 1 THEN 1
               WHEN [m_12] = 1 THEN 2
               WHEN [m_13] = 1 THEN 3
               WHEN [m_14] = 1 THEN 4
               WHEN [m_15] = 1 THEN 5
               WHEN [m_16] = 1 THEN 6
               WHEN [m_17] = 1 THEN 7
               WHEN [m_18] = 1 THEN 8
               WHEN [m_19] = 1 THEN 9
               WHEN [m_20] = 1 THEN 10
               WHEN [m_30] = 1 THEN 11
               WHEN [m_31] = 1 THEN 12
               WHEN [m_32] = 1 THEN 13
               WHEN [m_33] = 1 THEN 14
               ELSE NULL
             END AS [Метка класса процесс (число)]
      FROM [MeretoyahaRV].[{schema}].[{table_name}]
      ORDER BY [204]
    )
SELECT 
    round ([Вес на крюке],3) as [Вес на крюке],
    round ([Высота крюка],3) as [Высота крюка],
    round ([Давление на входе]/101325, 3) as [Давление на входе],
    round ([Обороты ротора], 3) as [Обороты ротора],
    [Глубина долота],
    [Глубина забоя],
    [Метка класса процесс (число)]
FROM CTE
WHERE [Метка класса время] IS NOT NULL 
  AND [Метка класса процесс] IS NOT NULL
ORDER BY [204];
    """


def main():
    # Формирование строки подключения
    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Получаем список таблиц из INFORMATION_SCHEMA.TABLES
        cursor.execute("""
            SELECT TOP (1) TABLE_SCHEMA + '.' + TABLE_NAME AS FullTableName
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
              AND TABLE_NAME LIKE 't\\_%' ESCAPE '\\'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        tables = [row[0] for row in cursor.fetchall()]

        # Открытие CSV-файла для записи результатов с учетом настроек разделителя
        with open('G:\\output1.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=CSV_SETTINGS["sep"])
            header_written = False

            for full_table_name in tqdm(tables, desc="Обработка таблиц"):
                sql_query = generate_sql_query_for_table_with_cte(full_table_name)
                cursor.execute(sql_query)
                columns = [column[0] for column in cursor.description]
                rows = cursor.fetchall()

                # Записываем заголовок только для первой таблицы
                if not header_written:
                    writer.writerow(columns)
                    header_written = True

                # Обработка строк для замены десятичного разделителя в числовых значениях
                for row in rows:
                    processed_row = []
                    for item in row:
                        if isinstance(item, float):
                            processed_row.append(format_number(item))
                        else:
                            processed_row.append(item)
                    writer.writerow(processed_row)

        print("Экспорт данных завершен успешно.")
    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    main()
