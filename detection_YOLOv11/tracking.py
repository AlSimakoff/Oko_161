from ultralytics import YOLO
import cv2
model = YOLO("detection_YOLOv11/yolo11n.pt")

def fullrecognise():
    video_path= "../data/video_test/Blog_05_20241228_08.36.22-08.36.35.h264"
    cap=cv2.VideoCapture(video_path)
    res = cap.isOpened()
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(
                frame,
                persist=True, classes=7)

            annotaded_frame = results[0].plot()
            cv2.imshow("Yolo11 Tracking", annotaded_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def recognise (frame):
    results = model.track(
        frame,
        persist=True,
    classes = [2,7])

    cords = results[0].boxes.xyxyn
    labls = results[0].boxes.cls
    return labls,cords