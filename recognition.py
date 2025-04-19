import cv2
import re
import numpy as np
import torch
import easyocr
from paddleocr import PaddleOCR
from collections import Counter
from lpr_net.model.lpr_net import build_lprnet
from lpr_net.rec_plate import CHARS

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def merge_by_confidence(per_char_results):
    """
    Объединяет символы от разных моделей по уверенности на каждой позиции.
    """
    from itertools import zip_longest
    final_plate = ""
    for chars in zip_longest(*per_char_results, fillvalue=('', 0.0)):
        best_char, _ = max(chars, key=lambda x: x[1])
        final_plate += best_char
    return final_plate

# === Распознавание с кастомной моделью LPRNet ===
def recognize_custom_model_chars(img):
    LPRnet = build_lprnet(
        lpr_max_len=9,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0
    )
    LPRnet.to(torch.device('cpu'))
    LPRnet.load_state_dict(torch.load("lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth", map_location=torch.device('cpu')))
    LPRnet.eval()

    image = cv2.resize(img, (94, 24)).astype("float32")
    image = (image - 127.5) * 0.0078125
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).cpu()

    with torch.no_grad():
        preds = LPRnet(image)

    preds_np = preds.cpu().numpy()[0]
    label_chars = []
    pre_c = None

    for j in range(preds_np.shape[1]):
        softmax_probs = softmax(preds_np[:, j])
        c = np.argmax(softmax_probs)
        prob = softmax_probs[c]

        if c == len(CHARS) - 1 or c == pre_c:
            pre_c = c
            continue

        label_chars.append((CHARS[c], prob))
        pre_c = c

    return label_chars

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
        conf = result[0][0][1][1]
        return [(c, conf / len(text)) for c in text]
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

