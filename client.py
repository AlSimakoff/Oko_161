import requests


def add_blog(time, color, license_number, type_auto):
    """Отправляет данные на сервер."""
    url = 'http://localhost:5000/journalblog'
    entry_data = {
        'time': time,
        'color': color,
        'license_number': license_number,
        'type_auto': type_auto

    }

    response = requests.post(url, json=entry_data)

    if response.status_code == 201:
        print("Запись успешно добавлена.")
    else:
        print(f"Ошибка при добавлении записи: {response.json().get('error')}")


def fetch_blog():
    """Получает данные о записях из API."""
    response = requests.get('http://localhost:5000/journalblog')

    if response.status_code == 200:
        rows = response.json()  # Десериализация JSON-ответа
        print("Полученные записи:", rows)
    else:
        print(f"Ошибка при получении данных: {response.status_code}")