from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="/root/autodl-tmp/dataset/data.yaml",  
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,
    workers=2
)