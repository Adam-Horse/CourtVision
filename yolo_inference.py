from ultralytics import YOLO

model = YOLO('yolov8n')

model.predict('./input_videos/image.png', save=True)