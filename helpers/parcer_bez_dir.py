import struct
import os
import pyodbc
import gc
from tqdm import tqdm

def decode_length(byte):
    if byte[0] & 0b10000000:
        num_bytes = byte[0] & 0b01111111
        return int.from_bytes(byte[1:1 + num_bytes], byteorder='little'), num_bytes + 1
    return byte[0], 1

def parse_value(byte_data, length):
    if length == 8:
        return struct.unpack("d", byte_data)[0]
    elif length == 4:
        return int.from_bytes(byte_data, byteorder='little')
    return byte_data

def parse_binary_file_updated(filepath):
    with open(filepath, "rb") as file:
        data = file.read()

    records = []
    unique_param_ids = set()
    i = 0
    print(filepath)

    for i in tqdm(range(len(data)), desc="Processing формирование массива из файла", unit="byte"):
        if data[i:i + 3] == b"REC":
            i += 4
            record_key = int.from_bytes(data[i:i + 4], byteorder='little')
            i += 4
            record = [record_key]

            while i < len(data) and data[i:i + 3] != b"REC":
                param_id = int.from_bytes(data[i:i + 4], byteorder='little')
                unique_param_ids.add(param_id)
                i += 4

                length, length_size = decode_length(data[i:i + 2])
                i += length_size
                param_value = None if length == 0 else parse_value(data[i:i + length], length)
                i += length if param_value is not None else 0

                record.append((param_id, length, param_value))
            records.append(record)

    return records, list(unique_param_ids)

def connect_to_server(database_name="master"):
    conn_str = (
        r"Driver={SQL Server};"
        f"Server=127.0.0.1\SQL_VER_16;"
        f"Database={database_name};"
        r"UID=sa;"
        r"PWD=123456;"
    )
    return pyodbc.connect(conn_str)

def create_database_if_not_exists(database_name, cursor):
    cursor.execute(f"SELECT name FROM sys.databases WHERE name = '{database_name}'")
    if not cursor.fetchone():
        cursor.commit()  # Завершаем текущую транзакцию перед созданием базы данных
        cursor.execute(f"CREATE DATABASE {database_name}")
        cursor.execute(f"ALTER DATABASE {database_name} SET RECOVERY SIMPLE")

def create_table_if_not_exists(database_name, table_name, cursor):
    cursor.execute(f"USE {database_name};")
    cursor.execute(f"IF OBJECT_ID(N'{table_name}', N'U') IS NULL "
                   f"BEGIN "
                   f"    CREATE TABLE {table_name} (id INT); "
                   f"END")

def sanitized_table_name(name):
    sanitized_name = name.replace("-", "").replace("}", "").replace("{", "")
    # Добавляем префикс ко всем именам
    return "t_" + sanitized_name

def add_columns_if_not_exists(database_name, table_name, columns, cursor):
    for column in columns:
        column_name = f"[{column}]"
        column_type = "datetime" if column == 204 else "FLOAT"
        sql_query = f"IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS " \
                    f"WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column}') " \
                    f"ALTER TABLE {table_name} ADD {column_name} {column_type};"
        try:
            cursor.execute(sql_query)
            cursor.commit()
        except Exception as e:
            print(f"Ошибка при добавлении столбца {column_name} в таблицу {table_name}: {e}")

def insert_records_into_table(database_name, table_name, corrected_records, cursor):

    for record in tqdm(corrected_records, desc="Inserting records", unit="record"):
        record_id = record[0]
        params = record[1:]
        columns = ['id']
        values = [str(record_id)]
        for param in params:
            columns.append(f"[{param[0]}]")
            values.append(str(param[2]))

        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)})"
        try:
            cursor.execute(query)
        except Exception as e:
            print(f"Ошибка при вставке записи в таблицу {table_name}: {e}")

    cursor.commit()

def find_base_files_updated(directory):
    result = []
    for subdirname in os.listdir(directory):
        subdirpath = os.path.join(directory, subdirname)
        if os.path.isdir(subdirpath):
            base_files = [os.path.join(root, file) for root, _, files in os.walk(subdirpath) for file in files if file.endswith('.base')]
            if base_files:
                result.append((subdirname, base_files))
    return result

def process_all_files(database_name, filelistarray):
    conn = connect_to_server()
    cursor = conn.cursor()
    create_database_if_not_exists(database_name, cursor)
    cursor.close()
    conn.close()
    conn = connect_to_server(database_name)
    cursor = conn.cursor()

    for file_info in filelistarray:
        table_name = sanitized_table_name(file_info[0])
        create_table_if_not_exists(database_name, table_name, cursor)

        for file_path in file_info[1]:
            corrected_records, identifiers = parse_binary_file_updated(file_path)
            add_columns_if_not_exists(database_name, table_name, identifiers, cursor)
            insert_records_into_table(database_name, table_name, corrected_records, cursor)

            del corrected_records, identifiers  # Удаляем ссылки на объекты
            gc.collect()

    cursor.close()
    conn.close()

databasename123 = "SPS2"
pachtodatabase = "G:\\EAM\\Srv\\СПС\\0\\11"

filelistarray = find_base_files_updated(pachtodatabase)
process_all_files(databasename123, filelistarray)
