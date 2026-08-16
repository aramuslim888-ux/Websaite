import cv2
import numpy as np
from ultralytics import YOLO

# 1. LOAD MODEL & VIDEO
MODEL_PATH = "yolov8m.pt"
VIDEO_IN = "video.mp4"
VIDEO_OUT = "output_speed.mp4"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_IN)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# دیاریکردنی هێڵەکان و ناوچەکان
ROAD_WIDTH_M = 7.6

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # جێبەجێکردنی مۆدێل بۆ دۆزینەوە و بەدواداچوون
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            # کێشانەوەی چوارگۆچە دەوروبەری ئۆتۆمبێلەکە
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # نوسینی خێرایی و زانیاری لەسەر شاشە
            label = f"Car ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("شیکاری ڤیدۆکە کۆتایی هات!")
