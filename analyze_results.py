from ultralytics import YOLO

model = YOLO("best.pt")
model.val(conf=0.25, plots=True) 

