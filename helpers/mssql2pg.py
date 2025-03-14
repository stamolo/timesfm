import pyodbc
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import quote_ident
from tqdm import tqdm

# Конфигурация подключения к SQL Server
MSSQL_CONFIG = {
    'server': 'DESKTOP-1CK097D\\SQL_VER_16',
    'database': 'MeretoyahaRV',
    'user': 'sa',
    'password': '123456',
    'driver': '{SQL Server}'
}

# Конфигурация PostgreSQL (Docker контейнер)
PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',  # при необходимости укажите имя БД, созданной в контейнере
    'user': 'sa',
    'password': '123456'
}

# Маппинг типов данных MSSQL -> PostgreSQL
TYPE_MAPPING = {
    'int': 'INTEGER',
    'bigint': 'BIGINT',
    'smallint': 'SMALLINT',
    'tinyint': 'SMALLINT',  # в PG нет tinyint, используем smallint
    'bit': 'BOOLEAN',
    'float': 'DOUBLE PRECISION',
    'real': 'REAL',
    'decimal': 'DECIMAL',
    'numeric': 'NUMERIC',
    'char': 'CHAR',
    'varchar': 'VARCHAR',
    'text': 'TEXT',
    'nchar': 'CHAR',
    'nvarchar': 'VARCHAR',
    'ntext': 'TEXT',
    'date': 'DATE',
    'datetime': 'TIMESTAMP',
    'datetime2': 'TIMESTAMP',
    'smalldatetime': 'TIMESTAMP',
    'time': 'TIME',
    'uniqueidentifier': 'UUID'
}


def get_mssql_connection():
    conn_str = (
        f'DRIVER={MSSQL_CONFIG["driver"]};'
        f'SERVER={MSSQL_CONFIG["server"]};'
        f'DATABASE={MSSQL_CONFIG["database"]};'
        f'UID={MSSQL_CONFIG["user"]};'
        f'PWD={MSSQL_CONFIG["password"]}'
    )
    return pyodbc.connect(conn_str)


def get_pg_connection():
    return psycopg2.connect(**PG_CONFIG)


def get_table_list(mssql_conn):
    cursor = mssql_conn.cursor()
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_type = 'BASE TABLE'
          AND table_schema = 'dbo'
    """)
    return [row.table_name for row in cursor.fetchall()]


def get_column_definitions(mssql_conn, table_name, pg_conn):
    cursor = mssql_conn.cursor()
    query = """
        SELECT column_name, data_type, character_maximum_length, 
               numeric_precision, numeric_scale, is_nullable
        FROM information_schema.columns
        WHERE table_name = ?
        ORDER BY ordinal_position
    """
    cursor.execute(query, table_name)
    columns = []
    for row in cursor.fetchall():
        mssql_type = row.data_type.lower()
        pg_type = TYPE_MAPPING.get(mssql_type, 'TEXT')

        # Обработка дополнительных параметров типов
        if pg_type == 'VARCHAR' and row.character_maximum_length and row.character_maximum_length > 0:
            pg_type = f'VARCHAR({row.character_maximum_length})'
        elif mssql_type in ['decimal', 'numeric']:
            pg_type = f'NUMERIC({row.numeric_precision}, {row.numeric_scale})'

        nullability = 'NULL' if row.is_nullable == 'YES' else 'NOT NULL'
        # Безопасное экранирование имен столбцов
        columns.append(f'{quote_ident(row.column_name, pg_conn)} {pg_type} {nullability}')
    return columns


def create_pg_table(pg_conn, table_name, columns_def):
    cursor = pg_conn.cursor()
    try:
        # При необходимости можно предварительно удалить таблицу:
        # cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))

        create_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} (\n    {}\n)").format(
            sql.Identifier(table_name),
            sql.SQL(',\n    ').join(map(sql.SQL, columns_def))
        )
        cursor.execute(create_query)
        pg_conn.commit()
    except Exception as e:
        pg_conn.rollback()
        raise e


def transfer_data(mssql_conn, pg_conn, table_name, batch_size=10000):
    mssql_cursor = mssql_conn.cursor()
    pg_cursor = pg_conn.cursor()

    try:
        # Получаем общее количество строк для таблицы
        mssql_cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        total_count = mssql_cursor.fetchone()[0]

        # Получаем структуру таблицы (без выборки данных)
        mssql_cursor.execute(f"SELECT * FROM [{table_name}] WHERE 1=0")
        columns = [column[0] for column in mssql_cursor.description]

        # Подготавливаем запросы
        select_query = f"SELECT * FROM [{table_name}]"
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join([sql.Placeholder()] * len(columns))
        )

        # Запускаем прогресс-бар
        progress_bar = tqdm(total=total_count, desc=f"Transfer {table_name}", unit="rows")

        mssql_cursor.execute(select_query)
        total_rows = 0

        while True:
            rows = mssql_cursor.fetchmany(batch_size)
            if not rows:
                break

            # Преобразование данных
            converted_rows = []
            for row in rows:
                converted_row = []
                for value in row:
                    if isinstance(value, bytes):
                        try:
                            converted_row.append(value.decode('utf-8'))
                        except UnicodeDecodeError:
                            converted_row.append(value.hex())
                    elif isinstance(value, str):
                        converted_row.append(value.strip())
                    else:
                        converted_row.append(value)
                converted_rows.append(tuple(converted_row))

            pg_cursor.executemany(insert_query, converted_rows)
            pg_conn.commit()
            rows_transferred = len(converted_rows)
            total_rows += rows_transferred
            progress_bar.update(rows_transferred)

        progress_bar.close()
        print(f"Table {table_name} completed. Total rows transferred: {total_rows}")

    except Exception as e:
        pg_conn.rollback()
        raise e


def main():
    mssql_conn = None
    pg_conn = None

    try:
        print("Connecting to databases...")
        mssql_conn = get_mssql_connection()
        pg_conn = get_pg_connection()

        print("Getting table list from MSSQL...")
        tables = get_table_list(mssql_conn)
        print(f"Tables to transfer: {tables}")

        for table in tables:
            print(f"\n{'=' * 40}\nProcessing table: {table}")

            print("Getting column definitions...")
            columns_def = get_column_definitions(mssql_conn, table, pg_conn)

            print("Creating table in PostgreSQL...")
            create_pg_table(pg_conn, table, columns_def)

            print("Transferring data...")
            transfer_data(mssql_conn, pg_conn, table)

            print(f"Table {table} transferred successfully!")

    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        if mssql_conn:
            mssql_conn.close()
        if pg_conn:
            pg_conn.close()
        print("\nConnections closed.")


if __name__ == "__main__":
    main()
