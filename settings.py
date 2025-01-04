import os

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FILE_PATH = os.environ.get(
    'file_path', 
    os.path.normpath("data/video_test/Blog_05_20241228_08.36.22-08.36.35.h264")
)
YOLO_MODEL_PATH = os.environ.get(
    'yolo_model', 
    os.path.normpath("object_detection/YOLOS_cars.pt")
)
LPR_MODEL_PATH = os.environ.get(
    'lpr_model', 
    os.path.normpath("lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth")
)

YOLO_CONF = 0.5
YOLO_IOU = 0.4
LPR_MAX_LEN = 9
LPR_DROPOUT = 0

FINAL_FRAME_RES = (640, 480)
DETECTION_AREA = [(0, 650), (1920, 1000)]