import settings
from detect import detect
import db
import client

def main(video_file_path,
         yolo_model_path,
         yolo_conf,
         yolo_iou,
         lpr_model_path,
         lpr_max_len,
         lpr_dropout_rate,
         device,
         database
         ):

    db.initiate(database)
    '''
    # Пример добавления пользователей
    client.add_blog("07.01.2025 0:02", "red", "A424YE161", "truck")
    client.add_blog("07.01.2025 0:12", "black", "E891AA61", "truck")

    # Получение списка пользователей
    client.fetch_blog()
    '''
    detect(
        video_file_path,
        yolo_model_path,
        yolo_conf,
        yolo_iou,
        lpr_model_path,
        lpr_max_len,
        lpr_dropout_rate,
        device
    )


if __name__ == "__main__":
    main(
        settings.FILE_PATH,
        settings.YOLO_MODEL_PATH,
        settings.YOLO_CONF,
        settings.YOLO_IOU,
        settings.LPR_MODEL_PATH,
        settings.LPR_MAX_LEN,
        settings.LPR_DROPOUT,
        settings.DEVICE,
        settings.database_path
    )




