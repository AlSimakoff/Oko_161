import math
import operator
import os

import cv2
import numpy as np

# Список для хранения векторов признаков для обучения
training_feature_vector = []


# Функция для вычисления евклидова расстояния между двумя векторами
def calculate_euclidean_distance(variable1, variable2, length):
    """
    Вычисляет евклидово расстояние между двумя векторами.

    :param variable1: Первый вектор.
    :param variable2: Второй вектор.
    :param length: Длина векторов.
    :return: Евклидово расстояние между векторами.
    """
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)  # Суммируем квадрат разностей
    return math.sqrt(distance)  # Возвращаем корень из суммы квадратов


# Функция для нахождения k ближайших соседей
def k_nearest_neighbors(test_instance, k):
    """
    Находит k ближайших соседей для тестового экземпляра.

    :param test_instance: Вектор признаков тестового экземпляра.
    :param k: Количество ближайших соседей для поиска.
    :return: Список ближайших соседей.
    """
    distances = []  # Список для хранения расстояний до обучающих векторов
    length = len(test_instance)

    for x in range(len(training_feature_vector)):
        dist = calculate_euclidean_distance(test_instance, training_feature_vector[x], length)  # Вычисляем расстояние
        distances.append((training_feature_vector[x], dist))  # Сохраняем вектор и его расстояние

    distances.sort(key=operator.itemgetter(1))  # Сортируем по расстоянию
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# Функция для голосования соседей и определения наиболее частого класса
def response_of_neighbors(neighbors):
    """
    Определяет класс на основе голосования ближайших соседей.

    :param neighbors: Список ближайших соседей.
    :return: Наиболее частый класс среди соседей.
    """

    all_possible_neighbors = {}

    for x in range(len(neighbors)):
        response = neighbors[x][-1]  # Получаем класс соседа

        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1  # Увеличиваем счетчик для класса
        else:
            all_possible_neighbors[response] = 1  # Инициализируем счетчик для нового класса

    sortedVotes = sorted(all_possible_neighbors.items(), key=operator.itemgetter(1),
                         reverse=True)  # Сортируем по количеству голосов

    return sortedVotes[0][0]  # Возвращаем класс с максимальным количеством голосов


def color_histogram_of_image(image):
    """
    Вычисляет цветовую гистограмму изображения.

    :param image: Исходное изображение.
    :return: Вектор признаков цветовой гистограммы.
    """

    chans = cv2.split(image)  # Разделяем изображение на цветовые каналы (B, G, R)
    colors = ("b", "g", "r")

    features = []  # Список для хранения значений гистограммы
    feature_data = []  # Список для хранения пиковых значений

    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = elem
        elif counter == 2:
            green = elem
        elif counter == 3:
            red = elem

    feature_data.append(red)
    feature_data.append(green)
    feature_data.append(blue)

    return feature_data


def color_histogram_of_training_image(img_name):
    """
    Вычисляет цветовую гистограмму обучающего изображения и добавляет его в вектор признаков.

    :param img_name: Имя файла изображения.
    """

    # Определяем цвет изображения по имени файла


    if "red" in img_name:
        data_source = "red"
    elif "yellow" in img_name:
        data_source = "yellow"
    elif "green" in img_name:
        data_source = "green"
    elif "orange" in img_name:
        data_source = "orange"
    elif "white" in img_name:
        data_source = "white"
    elif "black" in img_name:
        data_source = "black"
    elif "blue" in img_name:
        data_source = "blue"
    elif "violet" in img_name:
        data_source = "violet"

    # Загружаем изображение по имени файла
    image = cv2.imread(img_name)

    feature_data = color_histogram_of_image(image)  # Вычисляем цветовую гистограмму изображения
    feature_data.append(data_source)  # Добавляем цвет как метку

    training_feature_vector.append(feature_data)  # Добавляем данные в вектор признаков обучения


def training():
    """
    Обучает модель на наборе изображений различных цветов.

     Загружает изображения из папок с соответствующими цветами и вычисляет их гистограммы.
     """

    # red color training images
    for f in os.listdir("colour_detection/training_dataset/red"):
        color_histogram_of_training_image("colour_detection/training_dataset/red/" + f)

    # yellow color training images
    for f in os.listdir("colour_detection/training_dataset/yellow"):
        color_histogram_of_training_image("colour_detection/training_dataset/yellow/" + f)

    # green color training images
    for f in os.listdir("colour_detection/training_dataset/green"):
        color_histogram_of_training_image("colour_detection/training_dataset/green/" + f)

    # orange color training images
    for f in os.listdir("colour_detection/training_dataset/orange"):
        color_histogram_of_training_image("colour_detection/training_dataset/orange/" + f)

    # white color training images
    for f in os.listdir("colour_detection/training_dataset/white"):
        color_histogram_of_training_image("colour_detection/training_dataset/white/" + f)

    # black color training images
    for f in os.listdir("colour_detection/training_dataset/black"):
        color_histogram_of_training_image("colour_detection/training_dataset/black/" + f)

    # blue color training images
    for f in os.listdir("colour_detection/training_dataset/blue"):
        color_histogram_of_training_image("colour_detection/training_dataset/blue/" + f)


def main(image):
    """
     Основная функция для классификации цвета тестового изображения.

     :param image: Изображение для классификации.
     :return: Предсказанный цвет объекта на изображении.
     """


    test_feature_vector = color_histogram_of_image(image)  # Получаем вектор признаков тестового изображения
    classifier_prediction = []  # Список предсказаний

    k = 3  # Количество ближайших соседей

    for x in range(len(test_feature_vector)):
        neighbors = k_nearest_neighbors(test_feature_vector, k)
        result = response_of_neighbors(neighbors)
        classifier_prediction.append(result)

    return classifier_prediction[0]  # Возвращаем первое предсказание


def detect_color(box_image):
    """
     Определяет цвет объекта на изображении.

     :param box_image: Изображение объекта (например, автомобиль).
     :return: Предсказанный цвет объекта.
     """


    source_image = box_image
    prediction = "n.a."

    if training_feature_vector:
        prediction = main(source_image)
    else:
        training()
        prediction = main(source_image)

    return prediction