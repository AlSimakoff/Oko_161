from ultralytics import YOLO
import cv2
model = YOLO("yolo11n.pt")

video_path="data/video_test/Blog_05_20241228_08.36.22-08.36.35.h264"
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

