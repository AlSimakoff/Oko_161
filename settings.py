import os

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FILE_PATH = os.environ.get(
    'file_path', 
    os.path.normpath("data/video_test/test.mp4")
)
YOLO_MODEL_PATH = os.environ.get(
    'yolo_model', 
    os.path.normpath("detection_YOLOv11/Yolov11n_CarTruckPlate.pt")
)
LPR_MODEL_PATH = os.environ.get(
    'lpr_model', 
    os.path.normpath("lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth")
)

YOLO_CONF = 0.5
YOLO_IOU = 0.4
LPR_MAX_LEN = 9
LPR_DROPOUT = 0

FINAL_FRAME_RES = (1080, 720)
DETECTION_AREA = [(0, 0), (1920, 1000)]

database_path='data/database/oko161.db'

#Название площадки для передачи записей в бд
name_company_object="BolshoiLog"