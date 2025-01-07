import settings
from detect import detect
import db
from db import add_entry

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
    res=db.select_data(database,"Journal")
    print(res)
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

