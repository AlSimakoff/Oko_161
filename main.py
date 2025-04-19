import settings
from detect import detect, detectSource
import db


def main(video_file_path,
         video_source_path,
         yolo_model_path,
         yolo_conf,
         yolo_iou,
         lpr_model_path,
         lpr_max_len,
         lpr_dropout_rate,
         device,
         database,
         from_file):
    """
    Основная функция для инициализации базы данных и запуска процесса обнаружения.

    :param video_file_path: Путь к видеофайлу для обработки.
    :param video_source_path: Ссылка на доступ к видеопотоку камеры
    :param yolo_model_path: Путь к модели YOLO для обнаружения объектов.
    :param yolo_conf: Уровень уверенности для YOLO.
    :param yolo_iou: Порог IoU для YOLO.
    :param lpr_model_path: Путь к модели LPR (распознавание номерных знаков).
    :param lpr_max_len: Максимальная длина номерного знака.
    :param lpr_dropout_rate: Уровень дропаута для модели LPR.
    :param device: Устройство (CPU или GPU) для выполнения модели.
    :param database: Путь к базе данных.
    :param from_file: Запуск распознавания в файле?
    """

    # Инициализация базы данных
    db.initiate(database)

    '''
    # Пример добавления записей в базу данных
    client.add_blog("07.01.2025 0:02", "red", "A424YE161", "truck")
    client.add_blog("07.01.2025 0:12", "black", "E891AA61", "truck")

    # Получение списка записей из базы данных
    client.fetch_blog()
    '''
    if from_file:

        # Запуск функции обнаружения с заданными параметрами
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
    else:
        detectSource(
            video_source_path,
            yolo_model_path,
            yolo_conf,
            yolo_iou,
            lpr_model_path,
            lpr_max_len,
            lpr_dropout_rate,
            device
        )

if __name__ == "__main__":
    # Запуск основной функции с параметрами из файла настроек
    main(
        settings.FILE_PATH,          # Путь к видеофайлу
        settings.video_source_path,
        settings.YOLO_MODEL_PATH,   # Путь к модели YOLO
        settings.YOLO_CONF,         # Уровень уверенности для YOLO
        settings.YOLO_IOU,          # Порог IoU для YOLO
        settings.LPR_MODEL_PATH,    # Путь к модели LPR
        settings.LPR_MAX_LEN,       # Максимальная длина номерного знака
        settings.LPR_DROPOUT,       # Уровень дропаута для модели LPR
        settings.DEVICE,            # Устройство (CPU или GPU)
        settings.database_path,
        settings.FROM_FILE# Путь к базе данных
    )
