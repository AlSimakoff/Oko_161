import requests


def add_blog(row):
    """Отправляет данные на сервер."""
    url = 'http://localhost:5000/journalblog'
    entry_data = {
        'time': row['time'],
        'color': row['color'],
        'license_number': row['license_number'],
        'type_auto': row['type_auto']

    }

    try:
        response = requests.post(url, json=entry_data)
        response.raise_for_status()  # Проверка на ошибки HTTP

        # Обработка успешного ответа
        print("Запись успешно добавлена:", response.json())
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP ошибка: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Ошибка подключения: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Время ожидания истекло: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Ошибка запроса: {req_err}")


def fetch_blog():
    """Получает данные о записях из API."""
    try:
        response = requests.get('http://localhost:5000/journalblog')
        response.raise_for_status()  # Проверка на ошибки HTTP

        # Обработка успешного ответа
        entries = response.json()  # Предполагаем, что ответ в формате JSON
        print("Записи успешно получены:", entries)
        return entries
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP ошибка: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Ошибка подключения: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Время ожидания истекло: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Ошибка запроса: {req_err}")