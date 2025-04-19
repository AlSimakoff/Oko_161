import os
import cv2
import recognition  # твой модуль

# Путь к папке с изображениями
folder_path = r"D:\Sanya\Reco_project\Data\autoriaNumberplateOcrRu-2021-09-01\test\img"

# Счётчики совпадений
true_count = 0
false_count = 0

# Перебираем все изображения в папке
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"{filename} - [Ошибка загрузки изображения]")
            continue

        # Получаем эталонное значение: всё до первого "_" или "."
        base_name = filename.split("_")[0].split(".")[0]

        # Распознавание
        result = recognition.recognition_character_level(img)

        # Сравнение
        is_match = result == base_name
        if is_match:
            true_count += 1
        else:
            false_count += 1

        # Подсчёт процентов
        total = true_count + false_count
        true_percent = (true_count / total) * 100 if total else 0
        false_percent = (false_count / total) * 100 if total else 0

        print(f"{filename} - {result} - {is_match}")
        print(f"Совпадений: {true_percent:.2f}% | Несовпадений: {false_percent:.2f}%\n")
