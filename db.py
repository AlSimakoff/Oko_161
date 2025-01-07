import sqlite3
import os



def execute_query(database, query, values=None):
    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        if values:
            cursor.execute(query, values)
        else:
            cursor.execute(query)
        conn.commit()

def fetch_data(database, query):
    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()

def check_database_exists(db_file):
    """
    Проверяет, существует ли база данных.

    :param db_file: Путь к файлу базы данных
    :return: True, если база данных существует, иначе False
    """
    return os.path.exists(db_file)

def initiate(db):

    # Проверка существования базы данных
    if check_database_exists(db):
        print(f"Database '{db}' exist")
    else:
        print(f"Database '{db}' don't exist. Create new database")

    execute_query(db, '''CREATE TABLE IF NOT EXISTS Journal (
        id INTEGER PRIMARY KEY,
        time TEXT NOT NULL,
        color TEXT NOT NULL,
        license_number TEXT NOT NULL,
        type_auto TEXT NOT NULL
        )''')
    execute_query(db,'''
        CREATE TABLE IF NOT EXISTS Error (
        id INTEGER PRIMARY KEY,
        time TEXT NOT NULL,
        log TEXT NOT NULL
        )
        ''' )

    return None

def add_entry(db:str, table :str, row :dict[str,str]):
    sql_query=""
    values=()
    if table=="Journal":
        sql_query="INSERT INTO Journal (time, license_number, color, type_auto) VALUES (?, ?, ?, ?)"
        values= (row['time'],row['license_number'],row['color'], row['type_auto'])
    elif table=="Error":
        sql_query = "INSERT INTO Error (time, log) VALUES (?, ?)"
        values = (row['time'], row['log'])
    if sql_query and values:
        execute_query(db,sql_query,values)

    return None

def delete_entry(db:str, table :str, id_el):
    sql_query=f"DELETE FROM {table} WHERE id={id_el}"
    execute_query(db, sql_query)
    return None

def select_data(db:str, table :str):
    results=fetch_data(db,f"SELECT * FROM {table}")
    return results