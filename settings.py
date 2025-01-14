import os

import torch

#Устройство, на котором будут проводиться вычисления
DEVICE = 'cpu'

#Путь для распознавания
FILE_PATH = os.environ.get(
    'file_path', 
    os.path.normpath("data/video_test/Blog_05_20241228_08.36.22-08.36.35.h264")
)
#Путь к модели YOLO
YOLO_MODEL_PATH = os.environ.get(
    'yolo_model', 
    os.path.normpath("detection_YOLOv11/southcity.pt")
)
#Путь к модели LPR
LPR_MODEL_PATH = os.environ.get(
    'lpr_model', 
    os.path.normpath("lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth")
)

#Настройки моделей
YOLO_CONF = 0.5
YOLO_IOU = 0.4
LPR_MAX_LEN = 9
LPR_DROPOUT = 0

#Размер окна, выводимого на экран
FINAL_FRAME_RES = (1080, 720)
#Зона, в которой производится детекция
DETECTION_AREA = [(0, 0), (1920, 1000)]

#Путь к локальной базе данных
database_path='data/database/oko161.db'

#Название площадки для передачи записей в бд
name_company_object="BolshoiLog"

#Количество кадров в секунду, которые будет обрабатывать модель
FPS_detect=5

#Адрес сервера
server_url= 'http://localhost:5000/oko161'