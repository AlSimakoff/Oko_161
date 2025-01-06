import settings
from detect import detect

if __name__ == "__main__":
    detect(
        settings.FILE_PATH,
        settings.YOLO_MODEL_PATH,
        settings.YOLO_CONF,
        settings.YOLO_IOU,
        settings.LPR_MODEL_PATH,
        settings.LPR_MAX_LEN,
        settings.LPR_DROPOUT,
        settings.DEVICE
    )

