import cv2
import re
import numpy as np
import torch
import torch.nn.functional as F
import easyocr
from paddleocr import PaddleOCR
from collections import Counter
from lpr_net.model.lpr_net import build_lprnet
from lpr_net.rec_plate import CHARS, rec_plate_with_confidence

# === Константы и инициализация ===
ALLOWED_CHARS = "ABEKMHOPCTYX0123456789"
easy_reader = easyocr.Reader(['en'], gpu=False)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# === Вспомогательные функции ===
def is_valid_plate(plate):
    pattern = r'^[ABEKMHOPCTYX]{1}\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$'
    return bool(re.fullmatch(pattern, plate))

def clean_plate_text(text):
    text = text.upper().replace(" ", "").replace("-", "")
    return ''.join([c for c in text if c in ALLOWED_CHARS])



def expected_char_type_by_position(index):
    """Возвращает 'letter' или 'digit' для заданной позиции в номере"""
    if index == 0 or index in [4, 5]:
        return 'letter'
    elif index in [1, 2, 3, 6, 7, 8]:
        return 'digit'
    else:
        return None  # fallback для позиции вне 9 символов


def is_letter(char):
    return char in "ABEKMHOPCTYX"


def is_digit(char):
    return char in "0123456789"


def merge_by_confidence(per_char_results):
    """
    Объединяет символы от разных моделей по уверенности на каждой позиции,
    с приоритезацией символов, подходящих по ГОСТ.
    """
    from itertools import zip_longest
    final_plate = ""

    for i, chars in enumerate(zip_longest(*per_char_results, fillvalue=('', 0.0))):
        expected = expected_char_type_by_position(i)

        valid = []
        fallback = []

        for char, conf in chars:
            if (expected == 'letter' and is_letter(char)) or (expected == 'digit' and is_digit(char)):
                valid.append((char, conf))
            else:
                fallback.append((char, conf))

        if valid:
            best_char = max(valid, key=lambda x: x[1])[0]
        else:
            best_char = max(fallback, key=lambda x: x[1])[0]  # если ни один не подходит — берём самый уверенный

        final_plate += best_char

    return final_plate

# === Распознавание с кастомной моделью LPRNet ===
def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def recognize_custom_model_chars(img):
    """
    Распознаёт символы с модели LPRNet, возвращает список (char, confidence)
    """
    # Загрузка модели
    LPRnet = build_lprnet(
        lpr_max_len=9,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0
    )
    LPRnet.to(torch.device('cpu'))
    LPRnet.load_state_dict(torch.load(
        "lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth",
        map_location=torch.device('cpu')
    ))
    LPRnet.eval()

    # Предобработка изображения
    image = cv2.resize(img, (94, 24)).astype("float32")
    image = (image - 127.5) * 0.0078125
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).cpu()

    # Предсказание
    with torch.no_grad():
        preds = LPRnet(image)  # (B, C, T)

    # Логарифмическая вероятность
    log_probs = F.log_softmax(preds, dim=1)  # (B, C, T)
    log_probs = log_probs[0].cpu().numpy()  # (C, T)

    result = []
    total_log_conf = 0.0
    pre_c = None

    for j in range(log_probs.shape[1]):
        c = np.argmax(log_probs[:, j])
        if c == len(CHARS) - 1 or c == pre_c:
            pre_c = c
            continue

        confidence = np.exp(log_probs[c][j])  # преобразуем log-пробу в вероятность
        result.append((CHARS[c], float(confidence)))
        total_log_conf += log_probs[c][j]
        pre_c = c

    # Дополнительно: нормализуем вероятности, чтобы они были более "реалистичны"
    if result:
        max_conf = max([conf for _, conf in result])
        result = [(char, conf / max_conf) for char, conf in result]

    return result


# === EasyOCR ===
def recognize_with_easyocr_chars(img):
    result = easy_reader.readtext(img)
    if result:
        best_result = max(result, key=lambda x: x[2])
        text = clean_plate_text(best_result[1])
        conf = best_result[2]
        return [(c, conf / len(text)) for c in text]
    return []

# === PaddleOCR ===
def recognize_with_paddleocr_chars(img):
    result = paddle_ocr.ocr(img, cls=True)
    if result and result[0]:
        text = clean_plate_text(result[0][0][1][0])
        conf = result[0][0][1][1]  # Уверенность всей строки

        return [(c, conf) for c in text]  # Раздаём всем символам одинаковую
    return []


# === Главная функция объединения ===
def recognition_character_level(img):
    custom = recognize_custom_model_chars(img)
    paddle = recognize_with_paddleocr_chars(img)
   # easy = recognize_with_easyocr_chars(img)

    all_results = [res for res in [custom, paddle] if res]
    final_plate = merge_by_confidence(all_results)

    # Ограничение по длине
    final_plate = final_plate[:9]

    # Убираем, если невалидный и слишком длинный
    if not is_valid_plate(final_plate) and len(final_plate) > 9:
        final_plate = final_plate[:8]

    return final_plate

