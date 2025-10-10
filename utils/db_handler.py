import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_CONFIG, PIPELINE_CONFIG
from tqdm import tqdm
import io

logger = logging.getLogger(__name__)

# --- Глобальные переменные для кеширования ---
_engine = None
_object_name = None


def get_db_engine():
    """
    Создает и возвращает SQLAlchemy engine для подключения к PostgreSQL.
    Кеширует engine, чтобы не создавать новые подключения постоянно.
    """
    global _engine
    if _engine is not None:
        return _engine

    try:
        db_url = (
            f"postgresql+psycopg2://{DB_CONFIG['USER']}:{DB_CONFIG['PASSWORD']}"
            f"@{DB_CONFIG['HOST']}:{DB_CONFIG['PORT']}/{DB_CONFIG['DATABASE']}"
        )
        _engine = create_engine(db_url)
        logger.info(f"Успешно создано подключение к PostgreSQL: {DB_CONFIG['HOST']}/{DB_CONFIG['DATABASE']}")
        return _engine
    except Exception as e:
        logger.error(f"Не удалось создать подключение к PostgreSQL: {e}", exc_info=True)
        raise


def get_target_object_name():
    """
    Определяет имя целевого объекта для именования таблиц и представлений.
    """
    global _object_name
    if _object_name is not None:
        return _object_name

    name = DB_CONFIG.get("FILTER_COLUMN_VALUE")
    if not name or name.upper() == 'NONE':
        name = DB_CONFIG.get("VIEW_NAME")
        logger.info(f"DB_FILTER_VALUE не указан. Используется VIEW_NAME: {name}")

    if not name:
        raise ValueError("Не удалось определить имя целевого объекта. Проверьте DB_FILTER_VALUE и *VIEW_NAME в .env")

    cleaned_name = ''.join(c if c.isalnum() else '_' for c in name)
    cleaned_name = '_'.join(filter(None, cleaned_name.split('_')))

    _object_name = cleaned_name.lower()
    logger.info(f"Определено и очищено имя целевого объекта: {_object_name}")
    return _object_name


def execute_sql(sql_query, connection):
    """Выполняет текстовый SQL-запрос."""
    try:
        connection.execute(text(sql_query))
        logger.debug(f"Успешно выполнен SQL: {sql_query.strip()}")
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL: {sql_query.strip()}\n{e}", exc_info=True)
        raise


def setup_schema():
    """Создает схему 'nsr' в БД, если она не существует."""
    engine = get_db_engine()
    schema_name = PIPELINE_CONFIG.get("NSR_SCHEMA", "nsr")
    with engine.connect() as conn:
        with conn.begin():
            execute_sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name};", conn)
        logger.info(f"Схема '{schema_name}' готова к работе.")


def cleanup_object_artifacts(object_name=None):
    """
    ПОЛНОСТЬЮ удаляет все артефакты (таблицы, представления) для объекта.
    """
    if not object_name:
        object_name = get_target_object_name()

    engine = get_db_engine()
    schema_name = PIPELINE_CONFIG.get("NSR_SCHEMA", "nsr")

    logger.warning(f"ПОЛНАЯ ОЧИСТКА: Удаление всех артефактов для объекта '{object_name}' в схеме '{schema_name}'...")

    with engine.connect() as conn:
        with conn.begin():
            execute_sql(f'DROP VIEW IF EXISTS {schema_name}."{object_name}_results" CASCADE;', conn)
            execute_sql(f'DROP TABLE IF EXISTS {schema_name}."{object_name}_step11" CASCADE;', conn)
            execute_sql(f'DROP TABLE IF EXISTS {schema_name}."{object_name}" CASCADE;', conn)

    logger.info(f"Очистка для объекта '{object_name}' завершена.")


def cleanup_step_artifacts(step_number, object_name=None):
    """
    Удаляет артефакты КОНКРЕТНОГО шага.
    """
    if not object_name:
        object_name = get_target_object_name()

    engine = get_db_engine()
    schema_name = PIPELINE_CONFIG.get("NSR_SCHEMA", "nsr")

    table_to_drop = f'{schema_name}."{object_name}_step{step_number}"'
    view_to_drop = f'{schema_name}."{object_name}_results"'

    logger.warning(f"ОЧИСТКА ШАГА {step_number}: Удаление таблицы {table_to_drop} и представления {view_to_drop}...")

    with engine.connect() as conn:
        with conn.begin():
            execute_sql(f"DROP VIEW IF EXISTS {view_to_drop} CASCADE;", conn)
            execute_sql(f"DROP TABLE IF EXISTS {table_to_drop} CASCADE;", conn)

    logger.info(f"Очистка для шага {step_number} завершена.")


def _copy_df_to_db(df, table_name, schema, engine, index=False, index_label=None):
    """
    Вспомогательная функция, выполняющая только быструю загрузку данных через COPY.
    Предполагается, что таблица уже существует.
    """
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cursor:
            string_buffer = io.StringIO()
            df.to_csv(string_buffer, index=index, header=False, sep='\t', na_rep='', quoting=3, escapechar='\\',
                      decimal='.')

            # ИСПРАВЛЕНИЕ: Получаем размер буфера после записи и перед перемоткой
            buffer_size = string_buffer.tell()
            string_buffer.seek(0)

            full_table_name_quoted = f'"{schema}"."{table_name}"'
            columns = []
            if index:
                columns.append(f'"{index_label}"')
            columns.extend([f'"{col}"' for col in df.columns])

            copy_sql = f"COPY {full_table_name_quoted} ({','.join(columns)}) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '')"

            # ИСПРАВЛЕНИЕ: Используем полученный размер для tqdm
            with tqdm(total=buffer_size, desc=f"Загрузка в {table_name}", unit='B', unit_scale=True,
                      unit_divisor=1024) as pbar:
                cursor.copy_expert(copy_sql, string_buffer, size=8192)
                # ИСПРАВЛЕНИЕ: Обновляем прогресс-бар на фактический размер
                pbar.update(buffer_size)

        raw_conn.commit()
    except Exception as e:
        raw_conn.rollback()
        logger.error(f"Ошибка во время быстрой записи с помощью COPY: {e}", exc_info=True)
        raise
    finally:
        raw_conn.close()


def write_df_to_hypertable(df, time_column, object_name=None):
    """
    Записывает DataFrame в гипертаблицу TimescaleDB, используя сверхбыстрый метод COPY.
    Сначала создается пустая гипертаблица, затем загружаются данные.
    """
    if not object_name:
        object_name = get_target_object_name()

    engine = get_db_engine()
    schema_name = PIPELINE_CONFIG.get("NSR_SCHEMA", "nsr")
    table_name = object_name

    logger.info(f"Подготовка к быстрой записи {len(df)} строк в гипертаблицу {schema_name}.{table_name}...")

    # 1. Создаем пустую таблицу
    df.head(0).to_sql(table_name, engine, schema=schema_name, if_exists='replace', index=False)

    # 2. Превращаем ее в гипертаблицу и настраиваем сжатие
    with engine.connect() as conn:
        with conn.begin():
            try:
                logger.info(f"Преобразование пустой таблицы {table_name} в гипертаблицу по ключу '{time_column}'...")
                execute_sql(f"SELECT create_hypertable('{schema_name}.\"{table_name}\"', '{time_column}');", conn)

                logger.info(f"Включение сжатия для гипертаблицы {table_name}...")
                compress_sql = f"""
                ALTER TABLE {schema_name}."{table_name}"
                SET (timescaledb.compress = true, timescaledb.compress_orderby = '"{time_column}" DESC');
                """
                execute_sql(compress_sql, conn)

                policy_sql = f"""
                SELECT add_compression_policy('{schema_name}."{table_name}"', compress_after => INTERVAL '1 minutes');
                """
                execute_sql(policy_sql, conn)
                logger.info(f"Политика сжатия (через 7 дней) добавлена для {table_name}.")
            except Exception as e:
                logger.error(f"Не удалось преобразовать таблицу в гипертаблицу или настроить сжатие. Ошибка: {e}")
                raise

    # 3. Загружаем данные в уже готовую пустую гипертаблицу
    logger.info(f"Загрузка данных в готовую гипертаблицу {table_name}...")
    _copy_df_to_db(df, table_name, schema_name, engine, index=False)

    logger.info("Запись в гипертаблицу успешно завершена.")


def write_df_to_table(df, table_name, object_name=None):
    """
    Записывает DataFrame в обычную таблицу PostgreSQL, используя сверхбыстрый метод COPY.
    """
    if not object_name:
        object_name = get_target_object_name()

    engine = get_db_engine()
    schema_name = PIPELINE_CONFIG.get("NSR_SCHEMA", "nsr")
    full_table_name = f"{object_name}_{table_name}"

    logger.info(f"Быстрая запись {len(df)} строк в таблицу {schema_name}.{full_table_name} с помощью COPY...")

    # 1. Создаем пустую таблицу
    df.head(0).to_sql(full_table_name, engine, schema=schema_name, if_exists='replace', index=True,
                      index_label=PIPELINE_CONFIG['SORT_COLUMN'])

    # 2. Загружаем данные
    _copy_df_to_db(df, full_table_name, schema_name, engine, index=True, index_label=PIPELINE_CONFIG['SORT_COLUMN'])

    logger.info(f"Запись в таблицу {schema_name}.{full_table_name} успешно завершена.")


def read_data_from_db(table_or_view_name, object_name=None):
    """Читает данные из таблицы или представления в DataFrame."""
    if not object_name:
        object_name = get_target_object_name()

    engine = get_db_engine()
    schema_name = PIPELINE_CONFIG.get("NSR_SCHEMA", "nsr")

    logger.info(f"Чтение данных из {schema_name}.{table_or_view_name}...")
    df = pd.read_sql_table(table_or_view_name, engine, schema=schema_name)
    logger.info(f"Успешно загружено {len(df)} строк.")
    return df


def create_or_replace_results_view(object_name=None):
    """
    Создает или заменяет представление, которое объединяет базовую таблицу
    с результатами шага 11.
    """
    if not object_name:
        object_name = get_target_object_name()

    engine = get_db_engine()
    schema_name = PIPELINE_CONFIG.get("NSR_SCHEMA", "nsr")
    base_table = f'"{schema_name}"."{object_name}"'
    step11_table = f'"{schema_name}"."{object_name}_step11"'
    view_name = f'"{schema_name}"."{object_name}_results"'
    time_col = PIPELINE_CONFIG['SORT_COLUMN']

    logger.info(f"Создание/обновление итогового представления {view_name}...")

    # Получаем столбцы из step11_table, исключая временной ключ
    with engine.connect() as conn:
        try:
            step11_cols_query = f"""
                SELECT column_name 
                FROM information_schema.columns
                WHERE table_schema = '{schema_name}' AND table_name = '{object_name}_step11'
                AND column_name != '{time_col}';
            """
            step11_cols_result = conn.execute(text(step11_cols_query)).fetchall()
            if not step11_cols_result:
                logger.warning(f"Не найдены столбцы в таблице {step11_table}. Представление не будет создано.")
                return

            step11_cols = [f's11."{col[0]}"' for col in step11_cols_result]

        except Exception:
            logger.warning(f"Таблица {step11_table} еще не существует. Представление будет создано позже.")
            return

    # Составляем запрос для создания представления
    view_sql = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT
        base.*,
        {', '.join(step11_cols)}
    FROM
        {base_table} base
    LEFT JOIN
        {step11_table} s11 ON base."{time_col}" = s11."{time_col}";
    """

    with engine.connect() as conn:
        with conn.begin():
            execute_sql(view_sql, conn)

    logger.info(f"Представление {view_name} успешно создано/обновлено.")

