import os
import yaml
from datetime import datetime
from roboflow import Roboflow
from ultralytics import YOLO


CONFIG = {
    "api_key": "P2tjKy8PEFQv7qPleQs2",
    "workspace": "miker-rwvlm",
    "project_name": "yolo11_object_detector_w-u4gdf",
    "version": 2,
    "model_name": "yolo11s.pt",   # options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "experiment_name": "exp_yolo11s_50epochs"
}


timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M")
exp_folder = f"results/{CONFIG['experiment_name']}_{timestamp}"
os.makedirs(exp_folder, exist_ok=True)

with open(os.path.join(exp_folder, "config.yaml"), "w") as f:
    yaml.dump(CONFIG, f)

rf = Roboflow(api_key=CONFIG["api_key"])
project = rf.workspace(CONFIG["workspace"]).project(CONFIG["project_name"])
version = project.version(CONFIG["version"])
dataset = version.download("yolov11")

data_path = f"{CONFIG['project_name']}-{CONFIG['version']}/data.yaml"


model = YOLO(CONFIG["model_name"])

results = model.train(
    data=data_path,
    epochs=CONFIG["epochs"],
    imgsz=CONFIG["imgsz"],
    batch=CONFIG["batch"],
    project=exp_folder,
    name="train_logs",
)

print("\nTraining completed ✔")
print("Best model saved at:", f"{exp_folder}/train_logs/weights/best.pt")
