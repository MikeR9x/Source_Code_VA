from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val(data="data.yaml")

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
