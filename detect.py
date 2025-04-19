from datetime import datetime
import time
import cv2

import client
from detection_YOLOv11 import tracking
import settings
from lpr_net.model.lpr_net import build_lprnet
from lpr_net.rec_plate import rec_plate, CHARS
import torch
from track_logic import *
import numpy as np
from colour_detection.detect_color import detect_color
import db
import re
from recognition import  recognition_character_level


def preprocess(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Препроцесс перед отправкой на YOLO.
    Ресайз изображения до нужного размера.
    """
    image = cv2.resize(
        image, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC  # Изменение разрешения изображения
    )
    return image


def get_frames(video_src: str) -> np.ndarray:
    """
    Генератор, который читает видео и отдает фреймы.
    Используется для обработки видео по кадрам.
    """
    cap = cv2.VideoCapture(video_src)

    fps=cap.get(cv2.CAP_PROP_FPS)

    interval = round(int(fps) / settings.FPS_detect) #Установка FPS для распознавания
    if interval == 0:
        interval = 1
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End video")  # Сообщение о завершении видео
            break
        if index % interval == 0:
            yield frame  # Возвращаем текущий кадр
        index +=1
    cap.release()


def get_boxes(results, frame):
    """
    Извлекает метки и координаты обнаруженных объектов из результатов модели.

    :param results: Результаты инференса модели.
    :param frame: Фрейм, на котором рассчитываются координаты.
    :return: Словарь с метками и координатами объектов.
    """

    labels, cord = results  # Получаем метки и координаты объектов

    n = len(labels)  # Количество обнаруженных объектов
    x_shape, y_shape = frame.shape[1], frame.shape[0]  # Размеры фрейма

    labls_cords = {}
    numbers = []  # Список для хранения координат номерных знаков
    cars = []  # Список для хранения координат автомобилей
    trucks = []  # Список для хранения координат грузовиков
    buses = []  # Список для хранения координат автобусов

    for i in range(n):
        row = cord[i]  # Получаем координаты объекта
        x1, y1, x2, y2 = (
            int(row[0] * x_shape),
            int(row[1] * y_shape),
            int(row[2] * x_shape),
            int(row[3] * y_shape),
        )

        if labels[i] == 0:
            numbers.append((x1, y1, x2, y2))  # Добавляем номерные знаки в список
        elif labels[i] == 2:
            cars.append((x1, y1, x2, y2))  # Добавляем автомобили в список
        elif labels[i] == 1:
            trucks.append((x1, y1, x2, y2))  # Добавляем грузовики в список
        elif labels[i] == 3:
            buses.append((x1, y1, x2, y2))  # Добавляем автобусы в список

    labls_cords["numbers"] = numbers  # Сохраняем номера в словаре
    labls_cords["cars"] = cars  # Сохраняем автомобили в словаре
    labls_cords["trucks"] = trucks  # Сохраняем грузовики в словаре
    labls_cords["busses"] = buses  # Сохраняем автобусы в словаре

    return labls_cords


def check_roi(coords):
    """
    Проверяет, находится ли объект в заданной области интереса (ROI).

    :param coords: Координаты объекта.
    :return: True если объект в ROI, иначе False.
    """

    detection_area = settings.DETECTION_AREA  # Задаем область интереса из настроек

    xc = int((coords[0] + coords[2]) / 2)  # Центр по оси X объекта
    yc = int((coords[1] + coords[3]) / 2)  # Центр по оси Y объекта

    if (
        (detection_area[0][0] < xc < detection_area[1][0])
        and
        (detection_area[0][1] < yc < detection_area[1][1])
        ):
        return True
    else:
        return False

def plot_boxes(cars_list: list, frame: np.ndarray) -> np.ndarray:
    """
    Отрисовывает рамки вокруг обнаруженных объектов на кадре.

    :param cars_list: Список обнаруженных объектов с их координатами и метками.
    :param frame: Кадр на котором отрисовываются рамки.

    :return: Кадр с отрисованными рамками.
    """

    n = len(cars_list)

    for car in cars_list:
        car_type = car[2]  # Тип автомобиля

        x1_number, y1_number, x2_number, y2_number = car[0][0]
        number = car[0][1] + "_" + car[3]

        x1_car, y1_car, x2_car, y2_car = car[1][0]
        colour = car[1][1]

        car_bgr = (0, 0, 0)  # Цвет рамки автомобиля по умолчанию

        if car_type == "car":
            car_bgr = (0, 0, 255)  # Красный для автомобилей
        elif car_type == "truck":
            car_bgr = (0, 255, 0)  # Зеленый для грузовиков
        elif car_type == "bus":
            car_bgr = (255, 0, 0)  # Синий для автобусов

        number_bgr = (255, 255, 255)  # Белый цвет рамки для номерного знака

        cv2.rectangle(frame, (x1_car, y1_car), (x2_car, y2_car), car_bgr, 2)  # Рисуем рамку вокруг автомобиля

        cv2.putText(
            frame,
            car_type + " " + colour,
            (x1_car, y2_car + 15),
            0,
            1,
            car_bgr,
            thickness=2,
            lineType=cv2.LINE_AA,
        )  # Подписываем тип и цвет автомобиля

        cv2.rectangle(
            frame,
            (x1_number, y1_number),
            (x2_number, y2_number),
            number_bgr,
            2
        )  # Рисуем рамку вокруг номерного знака

        cv2.putText(
            frame,
            number,
            (x1_number - 20, y2_number + 30),
            0,
            1,
            number_bgr,
            thickness=2,
            lineType=cv2.LINE_AA,
        )  # Подписываем номерной знак

    detection_area = settings.DETECTION_AREA  # Получаем область интереса из настроек

    cv2.rectangle(frame, detection_area[0], detection_area[1], (0, 0, 0), 2)  # Рисуем рамку вокруг области интереса

    return frame


def db_entry(time_detected, lic_number, color, type_auto):
    """
    Записывает данные о распознанном автомобиле в базу данных.

    :param time_detected: Время обнаружения автомобиля.
    :param lic_number: Номер автомобиля.
    :param color: Цвет автомобиля.
    :param type_auto: Тип автомобиля (например "car", "truck", "bus").

     Запись добавляется как в базу данных через функцию db.add_entry(), так и через клиентский интерфейс client.add_blog().
     """


    db_entry_row = {"time": time_detected, "license_number": lic_number, "color": color, "type_auto": type_auto}

    print(db_entry_row)  # Выводим данные о записи в консоль

    db.add_entry(settings.database_path, "Journal", db_entry_row)  # Добавляем запись в базу данных

    client.add_blog(db_entry_row)  # Добавляем запись через клиентский интерфейс





def detect(
        video_file_path,
        yolo_model_path,
        yolo_conf,
        yolo_iou,
        lpr_model_path,
        lpr_max_len,
        lpr_dropout_rate,
        device):
    """
    Основная функция для обнаружения объектов на видео.

    :param video_file_path: Путь к видеофайлу для обработки.
    :param yolo_model_path: Путь к модели YOLO для обнаружения объектов.
    :param yolo_conf: Уровень уверенности для YOLO.
    :param yolo_iou: Порог IoU для YOLO.
    :param lpr_model_path: Путь к модели LPR (распознавание номерных знаков).
    :param lpr_max_len: Максимальная длина номерного знака.
    :param lpr_dropout_rate: Уровень дропаута для модели LPR.
    :param device: Устройство (CPU или GPU) для выполнения модели.

    Запускает процесс обработки видео и обнаружения объектов.
    """

    cv2.startWindowThread()  # Запускаем поток окна OpenCV

    LPRnet = build_lprnet(  ### Создаем модель LPR ###
        lpr_max_len=lpr_max_len,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=lpr_dropout_rate
    )

    LPRnet.to(torch.device(device))  ### Переносим модель на нужное устройство ###

    LPRnet.load_state_dict(
        torch.load(lpr_model_path, map_location=torch.device('cpu'))  ### Загружаем веса модели ###
    )

    last_number = ""  ### Переменная для отслеживания последнего распознанного номера ###
    k=0
    for raw_frame in get_frames(video_file_path):  ### Читаем кадры из видео ###

        proc_frame = preprocess(raw_frame, settings.FINAL_FRAME_RES)  ### Препроцессинг кадра ###

        results = tracking.recognise(yolo_model_path, proc_frame)  ### Обнаружение объектов ###

        labls_cords = get_boxes(results, raw_frame)  ### Получаем метки и координаты объектов ###

        new_cars = check_numbers_overlaps(labls_cords)  ### Проверяем пересечения номеров с автомобилями ###

        cars = []  ### Список автомобилей ###

        for car in new_cars:

            plate_coords = car[0]  ### Координаты номерного знака ###
            car_coords = car[1]  ### Координаты автомобиля ###

            if check_roi(plate_coords):  ### Проверяем нахождение номера в области интереса ###
                x1_car, y1_car = car_coords[0], car_coords[1]
                x2_car, y2_car = car_coords[2], car_coords[3]

                ## Определяем цвет автомобиля ##
                car_box_image = raw_frame[y1_car:y2_car, x1_car:x2_car]
                colour = detect_color(car_box_image)

                car[1] = [car_coords, colour]

                x1_plate, y1_plate = plate_coords[0], plate_coords[1]
                x2_plate, y2_plate = plate_coords[2], plate_coords[3]

                ## Определяем номер на номерном знаке ##
                plate_box_image = raw_frame[y1_plate:y2_plate, x1_plate:x2_plate]

                # Проверка разрешения изображения номерного знака (минимум 130x30)
                plate_height, plate_width = plate_box_image.shape[:2]
                if plate_height >= 20 and plate_width >= 80:
                    k = k + 1
                    output_path = f'save_image/plate_box_image{k}.png'
                    cv2.imwrite(output_path, plate_box_image)

                    plate_text = recognition_character_level(plate_box_image)  # Распознаем номер
                    # Здесь можно добавить дополнительные проверки или действия после распознавания
                else:
                    print(f"Пропущено изображение номерного знака с размерами: {plate_width}x{plate_height}")
                    break

                ## Проверяем соответствует ли номер российскому формату ##
                if (
                        not re.match("[A-Z]{1}[0-9]{3}[A-Z]{2}[0-9]{2,3}", plate_text)
                            is None):
                    car[0] = [plate_coords, plate_text]
                    car.append("OK")  ### Номер успешно распознан ###

                    ## Добавляем запись в базу данных ##
                    ## db_entry может быть вызван здесь или позже ##

                else:
                    car[0] = [plate_coords, plate_text]
                    car.append("NOK")  ### Номер не распознан корректно ###

                cars.append(car)

        for car in cars:
            if car[0][1] != last_number and car[3] == "OK":  ### Если номер изменился и он успешно распознан ###
                db_entry(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), car[0][1], car[1][1], car[2])
                last_number = car[0][1]

        drawn_frame = plot_boxes(cars, raw_frame)  ### Отрисовываем рамки вокруг обнаруженных объектов ###
        proc_frame = preprocess(drawn_frame, settings.FINAL_FRAME_RES)

        cv2.imshow("video", proc_frame)  ### Показываем обработанный кадр на экране ###

        ## Ожидание нажатия клавиши 's' или 'q' ##
        if cv2.waitKey(30) & 0xFF == ord("s"):
            time.sleep(5)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break  ## Выход из цикла при нажатии 'q' ##


def detectSource(
        video_source,
        yolo_model_path,
        yolo_conf,
        yolo_iou,
        lpr_model_path,
        lpr_max_len,
        lpr_dropout_rate,
        device):
    """
    Основная функция для обнаружения объектов на видеопотоке.

    :param video_source: Источник видеопотока (номер камеры или URL).
    :param yolo_model_path: Путь к модели YOLO для обнаружения объектов.
    :param yolo_conf: Уровень уверенности для YOLO.
    :param yolo_iou: Порог IoU для YOLO.
    :param lpr_model_path: Путь к модели LPR (распознавание номерных знаков).
    :param lpr_max_len: Максимальная длина номерного знака.
    :param lpr_dropout_rate: Уровень дропаута для модели LPR.
    :param device: Устройство (CPU или GPU) для выполнения модели.
    """

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеопоток.")
        return

    LPRnet = build_lprnet(
        lpr_max_len=lpr_max_len,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=lpr_dropout_rate
    )
    LPRnet.to(torch.device(device))
    LPRnet.load_state_dict(torch.load(lpr_model_path, map_location=torch.device(device)))

    last_number = ""
    k = 1

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Ошибка: Кадр не считался.")
            break

        proc_frame = preprocess(raw_frame, settings.FINAL_FRAME_RES)
        results = tracking.recognise(yolo_model_path, proc_frame)
        labls_cords = get_boxes(results, raw_frame)
        new_cars = check_numbers_overlaps(labls_cords)
        cars = []

        for car in new_cars:
            plate_coords = car[0]
            car_coords = car[1]

            if check_roi(plate_coords):
                x1_car, y1_car = car_coords[0], car_coords[1]
                x2_car, y2_car = car_coords[2], car_coords[3]
                car_box_image = raw_frame[y1_car:y2_car, x1_car:x2_car]
                colour = detect_color(car_box_image)
                car[1] = [car_coords, colour]

                x1_plate, y1_plate = plate_coords[0], plate_coords[1]
                x2_plate, y2_plate = plate_coords[2], plate_coords[3]
                plate_box_image = raw_frame[y1_plate:y2_plate, x1_plate:x2_plate]
                plate_text = recognition_character_level(plate_box_image)

                if re.match("[A-Z]{1}[0-9]{3}[A-Z]{2}[0-9]{2,3}", plate_text):
                    car[0] = [plate_coords, plate_text]
                    car.append("OK")
                else:
                    car[0] = [plate_coords, plate_text]
                    car.append("NOK")

                cars.append(car)

        '''for car in cars:
            if car[0][1] != last_number and car[3] == "OK":
                db_entry(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), car[0][1], car[1][1], car[2])
                last_number = car[0][1]'''

        drawn_frame = plot_boxes(cars, raw_frame)
        proc_frame = preprocess(drawn_frame, settings.FINAL_FRAME_RES)
        cv2.imshow("video", proc_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            time.sleep(5)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

