import requests


def add_user(name, age):
    """Отправляет данные пользователя на сервер."""
    url = 'http://localhost:5000/users'
    user_data = {
        'name': name,
        'age': age
    }

    response = requests.post(url, json=user_data)

    if response.status_code == 201:
        print("Пользователь успешно добавлен.")
    else:
        print(f"Ошибка при добавлении пользователя: {response.json().get('error')}")


def fetch_users():
    """Получает данные пользователей из API."""
    response = requests.get('http://localhost:5000/users')

    if response.status_code == 200:
        users = response.json()  # Десериализация JSON-ответа
        print("Полученные записи:", users)
    else:
        print(f"Ошибка при получении данных: {response.status_code}")