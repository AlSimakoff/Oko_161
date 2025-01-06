import client


def main():
    # Пример добавления пользователей
    client.add_user("Alice", 30)
    client.add_user("Bob", 25)

    # Получение списка пользователей
    client.fetch_users()


if __name__ == "__main__":
    main()