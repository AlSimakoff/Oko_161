import datetime

import cv2
import numpy as np
import torch
from os import path

from numpy.distutils.lib2def import output_def

# Список символов, которые могут быть распознаны на номерных знаках
CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    # Заменяем I и O на другие символы, чтобы избежать путаницы
    "I",
    "O",
    "_",
]

def rec_plate(lprnet, img) -> str:
    """
    Распознает номерной знак на изображении с использованием модели LPR.

    :param lprnet: Объект модели LPR для распознавания.
    :param img: Изображение номерного знака для распознавания.
    :return: Распознанный номерной знак в виде строки.
    """

    # Предобработка изображения
    image = img
    width, length, _ = image.shape  # Получаем размеры изображения
    image = cv2.resize(image, (94, 24))  # Изменяем размер изображения для модели LPR
    image = image.astype("float32")  # Приводим к типу float32
    image -= 127.5  # Нормализация значений пикселей
    image *= 0.0078125  # Масштабирование значений пикселей
    # Перестановка осей для соответствия входным данным модели
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).cpu()  # Преобразуем в тензор PyTorch и переносим на CPU
    image = image.unsqueeze(0)  # Добавляем размерность для батча

    # Прямой проход через модель LPR
    preds = lprnet(image)

    # Декодирование предсказаний
    preds = preds.cpu().detach().numpy()  # Переносим данные обратно на CPU и преобразуем в NumPy массив
    label = ""  # Инициализация строки для распознанного номера

    for i in range(preds.shape[0]):  # Проходим по всем предсказаниям
        preds = preds[i, :, :]  # Получаем предсказания для текущего номера
        preds_label = list()  # Список для хранения меток предсказаний

        for j in range(preds.shape[1]):  # Проходим по каждому столбцу предсказаний
            preds_label.append(np.argmax(preds[:, j], axis=0))  # Находим индекс максимального значения

        pre_c = preds_label[0]  # Инициализация предыдущего символа

        if pre_c != len(CHARS) - 1:  # Проверяем, не является ли символ пробелом
            label += CHARS[pre_c]  # Добавляем символ к распознанному номеру

        for c in preds_label:  # Убираем повторяющиеся символы и пробелы
            if (pre_c == c) or (c == len(CHARS) - 1):  # Если текущий символ такой же как предыдущий или пробел
                if c == len(CHARS) - 1:
                    pre_c = c  # Обновляем предыдущий символ на пробел
                continue

            label += CHARS[c]  # Добавляем текущий символ к распознанному номеру
            pre_c = c  # Обновляем предыдущий символ

    return label  # Возвращаем распознанный номерной знак в виде строки



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rec_plate_with_confidence(lprnet, img) -> tuple[str, float]:
    image = cv2.resize(img, (94, 24)).astype("float32")
    image = (image - 127.5) * 0.0078125
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).cpu()

    with torch.no_grad():
        preds = lprnet(image)

    preds_np = preds.cpu().numpy()[0]  # (num_classes, seq_len)
    label = ""
    confidence_scores = []

    preds_label = []
    pre_c = None

    for j in range(preds_np.shape[1]):
        softmax_probs = softmax(preds_np[:, j])
        c = np.argmax(softmax_probs)
        prob = softmax_probs[c]

        if c == len(CHARS) - 1:  # пустой символ
            pre_c = c
            continue
        if c == pre_c:
            continue

        preds_label.append((c, prob))
        pre_c = c

    for c, p in preds_label:
        label += CHARS[c]
        confidence_scores.append(p)

    avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    return label, avg_confidence

