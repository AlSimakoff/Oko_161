from ultralytics import YOLO

# Load the model.
model = YOLO("yolo11s.pt")

# Training.
results = model.train(
    data='D:/repos/VKR/Oko_161/data/dataset/archive/License-Plate-Data/data.yaml',
    imgsz=400,
    epochs=50,
    batch=8,
    name='yolov11s_carPlate')