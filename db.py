import sqlite3
import os


def execute_query(database, query, values=None):
    """
    Выполняет SQL-запрос к базе данных.

    :param database: Путь к файлу базы данных.
    :param query: SQL-запрос для выполнения.
    :param values: Значения для подстановки в запрос (если есть).
    """
    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        if values:
            cursor.execute(query, values)  # Выполняем запрос с параметрами
        else:
            cursor.execute(query)  # Выполняем запрос без параметров
        conn.commit()  # Сохраняем изменения в базе данных


def fetch_data(database, query):
    """
    Извлекает данные из базы данных по заданному запросу.

    :param database: Путь к файлу базы данных.
    :param query: SQL-запрос для извлечения данных.
    :return: Список результатов запроса.
    """
    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        cursor.execute(query)  # Выполняем запрос
        return cursor.fetchall()  # Возвращаем все результаты


def check_database_exists(db_file):
    """
    Проверяет, существует ли база данных.

    :param db_file: Путь к файлу базы данных.
    :return: True, если база данных существует, иначе False.
    """
    return os.path.exists(db_file)  # Проверяем существование файла


def initiate(db):
    """
    Инициализирует базу данных: проверяет существование и создает таблицы.

    :param db: Путь к файлу базы данных.
    """

    # Проверка существования базы данных
    if check_database_exists(db):
        print(f"Database '{db}' exists")  # База данных существует
    else:
        print(f"Database '{db}' doesn't exist. Creating new database")  # База данных не существует

    # Создание таблицы Journal, если она не существует
    execute_query(db, '''CREATE TABLE IF NOT EXISTS Journal (
        id INTEGER PRIMARY KEY,
        time TEXT NOT NULL,
        color TEXT NOT NULL,
        license_number TEXT NOT NULL,
        type_auto TEXT NOT NULL
        )''')

    # Создание таблицы Error, если она не существует
    execute_query(db, '''
        CREATE TABLE IF NOT EXISTS Error (
        id INTEGER PRIMARY KEY,
        time TEXT NOT NULL,
        log TEXT NOT NULL
        )
        ''')


def add_entry(db: str, table: str, row: dict[str, str]):
    """
    Добавляет запись в указанную таблицу базы данных.

    :param db: Путь к файлу базы данных.
    :param table: Название таблицы для добавления записи.
    :param row: Словарь с данными для записи.
    """

    sql_query = ""
    values = ()

    if table == "Journal":
        sql_query = "INSERT INTO Journal (time, license_number, color, type_auto) VALUES (?, ?, ?, ?)"
        values = (row['time'], row['license_number'], row['color'], row['type_auto'])  # Подготовка значений для вставки
    elif table == "Error":
        sql_query = "INSERT INTO Error (time, log) VALUES (?, ?)"
        values = (row['time'], row['log'])  # Подготовка значений для вставки

    if sql_query and values:
        execute_query(db, sql_query, values)  # Выполнение запроса на добавление записи


def delete_entry(db: str, table: str, id_el):
    """
    Удаляет запись из указанной таблицы по ID.

    :param db: Путь к файлу базы данных.
    :param table: Название таблицы из которой нужно удалить запись.
    :param id_el: ID записи для удаления.
    """

    sql_query = f"DELETE FROM {table} WHERE id={id_el}"  # Формируем SQL-запрос на удаление записи
    execute_query(db, sql_query)  # Выполняем запрос на удаление


def select_data(db: str, table: str):
    """
    Извлекает все данные из указанной таблицы.

    :param db: Путь к файлу базы данных.
    :param table: Название таблицы из которой нужно извлечь данные.

    :return: Результаты запроса на выборку.
    """

    results = fetch_data(db, f"SELECT * FROM {table}")  # Получаем данные из таблицы
    return results  # Возвращаем результаты
