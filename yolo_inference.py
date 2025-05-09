from ultralytics import YOLO

model = YOLO('yolov8')

result = model.predict('./input_videos/videoclip.mp4', save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)