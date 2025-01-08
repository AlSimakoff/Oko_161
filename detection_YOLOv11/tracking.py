from ultralytics import YOLO
import cv2

'''
Функция для тестирования модели YOLO.
'''


def fullrecognise():
    # Загружаем модель YOLO из указанного пути
    model = YOLO("detection_YOLOv11/Yolov11n_CarTruckPlate.pt")

    # Указываем путь к видеофайлу для обработки
    video_path = "../data/video_test/Blog_05_20241228_08.36.22-08.36.35.h264"

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    # Проверяем, успешно ли открыт видеофайл
    res = cap.isOpened()

    while cap.isOpened():
        success, frame = cap.read()  # Читаем кадр из видео

        if success:
            # Выполняем отслеживание объектов на текущем кадре
            results = model.track(
                frame,
                persist=True,
                classes=7  # Указываем класс для отслеживания (например, номерные знаки)
            )

            # Получаем аннотированный кадр с результатами отслеживания
            annotaded_frame = results[0].plot()

            # Отображаем аннотированный кадр в окне
            cv2.imshow("Yolo11 Tracking", annotaded_frame)

            # Выход из цикла при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break  # Если не удалось прочитать кадр, выходим из цикла

    cap.release()  # Освобождаем ресурсы видео
    cv2.destroyAllWindows()  # Закрываем все открытые окна


'''
Основная функция для распознавания объектов с использованием YOLO.
'''


def recognise(model_path, frame):
    """
    Распознает объекты на переданном кадре с использованием модели YOLO.

    :param model_path: Путь к файлу модели YOLO.
    :param frame: Кадр изображения для распознавания.
    :return: Метки классов и координаты обнаруженных объектов.
    """

    model = YOLO(model_path)  # Загружаем модель YOLO из указанного пути

    # Выполняем отслеживание объектов на переданном кадре
    results = model.track(
        frame,
        persist=True  # Указываем, что нужно сохранять состояние между кадрами
    )

    cords = results[0].boxes.xyxyn  # Получаем координаты обнаруженных объектов
    labls = results[0].boxes.cls  # Получаем метки классов обнаруженных объектов

    return labls, cords  # Возвращаем метки и координаты объектов
