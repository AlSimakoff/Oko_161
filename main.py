import client


def main():
    # Пример добавления пользователей
    client.add_blog("07.01.2025 0:02", "red", "A424YE161","truck")
    client.add_blog("07.01.2025 0:12", "black", "E891AA61", "truck")

    # Получение списка пользователей
    client.fetch_blog()


if __name__ == "__main__":
    main()