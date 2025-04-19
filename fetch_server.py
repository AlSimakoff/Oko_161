import requests
import cv2
import base64
from ultralytics import settings
import settings

def encode_image_to_base64(img):
    """Кодирует изображение OpenCV в строку base64."""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def send_entry(time, license_number, img_plate, img_car, color='unknown', type_auto='car'):
    """Отправляет данные на сервер, включая изображения и метаинформацию."""
    url = settings.server_url

    entry_data = {
        'time': time,
        'color': color,
        'license_number': license_number,
        'type_auto': type_auto,
        'table_name': settings.name_company_object+'_img',
        'img_plate': encode_image_to_base64(img_plate),
        'img_car': encode_image_to_base64(img_car)
    }

    try:
        response = requests.post(url, json=entry_data)
        response.raise_for_status()
        print("Запись успешно добавлена:", response.json())
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP ошибка: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Ошибка подключения: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Время ожидания истекло: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Ошибка запроса: {req_err}")
