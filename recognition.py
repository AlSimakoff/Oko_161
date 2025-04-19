import cv2
import re
import easyocr
from paddleocr import PaddleOCR
from collections import Counter
from lpr_net.rec_plate import rec_plate, CHARS
from lpr_net.model.lpr_net import build_lprnet
import torch

# Инициализация моделей
easy_reader = easyocr.Reader(['en'], gpu=False)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Разрешённые символы (буквы из номеров РФ + цифры)
ALLOWED_CHARS = "ABEKMHOPCTYX0123456789"

# Проверка на валидный формат гос. номера РФ
def is_valid_plate(plate):
    pattern = r'^[ABEKMHOPCTYX]{1}\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$'
    return bool(re.fullmatch(pattern, plate))

# Очистка и фильтрация текста
def clean_plate_text(text):
    text = text.upper().replace(" ", "").replace("-", "")
    return ''.join([c for c in text if c in ALLOWED_CHARS])

# EasyOCR
def recognize_with_easyocr(img):
    result = easy_reader.readtext(img, detail=0)
    if result:
        return clean_plate_text(''.join(result))
    return ""

# PaddleOCR
def recognize_with_paddleocr(img):
    result = paddle_ocr.ocr(img, cls=True)
    if result and result[0]:
        return clean_plate_text(''.join([line[1][0] for line in result[0]]))
    return ""

# Твоя модель — подключи тут
def recognize_custom_model(img):
    LPRnet = build_lprnet(  ### Создаем модель LPR ###
        lpr_max_len=9,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0
    )

    LPRnet.to(torch.device('cpu'))  ### Переносим модель на нужное устройство ###

    LPRnet.load_state_dict(
        torch.load("lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth", map_location=torch.device('cpu'))  ### Загружаем веса модели ###
    )
    plate_text = rec_plate(LPRnet,img)
    return plate_text

# Главная функция объединения
def recognition(plate_box_image):
    results = {
        "custom": clean_plate_text(recognize_custom_model(plate_box_image)),
        "paddle": recognize_with_paddleocr(plate_box_image),
        "easy": recognize_with_easyocr(plate_box_image)
    }

    # Отфильтруем только допустимые
    valid = [plate for plate in results.values() if is_valid_plate(plate)]

    # Если есть валидные — берём наиболее частый
    if valid:
        final = Counter(valid).most_common(1)[0][0]
    else:
        # Если нет валидных — берём самую длинную строку
        final = max(results.values(), key=len)

    return final
