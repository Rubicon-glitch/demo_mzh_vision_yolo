from ultralytics import YOLO


model = YOLO("/root/autodl-tmp/runs/detect/train/weights/best.pt")


model.export(format="onnx")