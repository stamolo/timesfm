import pyodbc

# Параметры подключения
server = '10.8.0.5\\SQL_VER_16'
database = 'MeretoyahaRV'
username = 'sa'
password = '123456'
driver = '{SQL Server}'

# Формирование строки подключения
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};'

# Устанавливаем подключение
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Запрос для получения списка таблиц, начинающихся на "t_"
query_tables = r"""
SELECT TABLE_SCHEMA, TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
  AND TABLE_NAME LIKE 't\_%' ESCAPE '\'
ORDER BY TABLE_SCHEMA, TABLE_NAME;
"""

cursor.execute(query_tables)
tables = cursor.fetchall()

# Обходим каждую таблицу и создаем для неё представление
for schema, table in tables:
    view_name = f"v4_{table}"
    full_table_name = f"[{schema}].[{table}]"
    full_view_name = f"[{schema}].[{view_name}]"

    # Если представление уже существует, удаляем его
    drop_view_script = f"""
    IF OBJECT_ID('{full_view_name}', 'V') IS NOT NULL
        DROP VIEW {full_view_name};
    """
    cursor.execute(drop_view_script)
    conn.commit()

    # Скрипт создания представления
    create_view_script = f"""
    CREATE VIEW {full_view_name} AS
    SELECT TOP 100 PERCENT
          [204] AS Время_204,
          ROUND([28]/9806, 3) AS Вес_на_крюке_28,
          ROUND([103], 3) AS Высота_крюка_103,
          ROUND([18]/101325, 3) AS Давление_на_входе_18,
          ROUND([72]*9.54, 3) AS Обороты_ротора_72,
          ROUND([35], 3) AS Глубина_долота_35,
          ROUND([36], 3) AS Глубина_забоя_36
    FROM {full_table_name}
    ORDER BY [204];
    """

    print(f"Создаём представление: {full_view_name}")
    cursor.execute(create_view_script)
    conn.commit()

cursor.close()
conn.close()
