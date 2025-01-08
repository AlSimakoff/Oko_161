import requests
from ultralytics import settings
import settings

def add_blog(row):
    """Отправляет данные на сервер."""
    url = 'http://localhost:5000/oko161'  # URL API для добавления записи
    entry_data = {
        'time': row['time'],  # Время записи
        'color': row['color'],  # Цвет автомобиля
        'license_number': row['license_number'],  # Номер автомобиля
        'type_auto': row['type_auto'],  # Тип автомобиля (например, "car", "truck", "bus")
        'table_name': settings.name_company_object
    }

    try:
        # Отправляем POST-запрос с данными записи в формате JSON
        response = requests.post(url, json=entry_data)
        response.raise_for_status()  # Проверка на ошибки HTTP

        # Обработка успешного ответа
        print("Запись успешно добавлена:", response.json())  # Выводим ответ от сервера
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP ошибка: {http_err}")  # Обработка ошибок HTTP
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Ошибка подключения: {conn_err}")  # Обработка ошибок подключения
    except requests.exceptions.Timeout as timeout_err:
        print(f"Время ожидания истекло: {timeout_err}")  # Обработка тайм-аутов
    except requests.exceptions.RequestException as req_err:
        print(f"Ошибка запроса: {req_err}")  # Обработка других ошибок запроса

def fetch_blog():
    """Получает данные о записях из API."""
    try:
        param={'table':settings.name_company_object}
        # Обработка успешного ответа
        entries = requests.get('http://localhost:5000/oko161/', params=param)
        print("Записи успешно получены:", entries)  # Выводим полученные записи
        return entries  # Возвращаем записи
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP ошибка: {http_err}")  # Обработка ошибок HTTP
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Ошибка подключения: {conn_err}")  # Обработка ошибок подключения
    except requests.exceptions.Timeout as timeout_err:
        print(f"Время ожидания истекло: {timeout_err}")  # Обработка тайм-аутов
    except requests.exceptions.RequestException as req_err:
        print(f"Ошибка запроса: {req_err}")  # Обработка других ошибок запроса
