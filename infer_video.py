from ultralytics import YOLO
import cv2
import time

MODEL = "best.pt"
CONF = 0.35
SOURCE = 0  # webcam

model = YOLO(MODEL)
cap = cv2.VideoCapture(SOURCE)

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF)
    annotated = results[0].plot()

    curr = time.time()
    fps = 1 / (curr - prev_time) if prev_time else 0
    prev_time = curr

    cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("YOLOv11", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
