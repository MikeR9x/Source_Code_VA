import os
import yaml
import time
from datetime import datetime
from ultralytics import YOLO
from roboflow import Roboflow


EXPERIMENTS = [
    {
        "model_name": "yolo11n.pt",
        "epochs": 30,
        "imgsz": 640,
        "batch": 16,
        "experiment_name": "exp_yolo11n_30ep"
    },
    {
        "model_name": "yolo11s.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "experiment_name": "exp_yolo11s_50ep"
    },
    {
        "model_name": "yolo11m.pt",
        "epochs": 60,
        "imgsz": 640,
        "batch": 16,
        "experiment_name": "exp_yolo11m_60ep"
    }
]


DATA_CONFIG = {
    "api_key": "P2tjKy8PEFQv7qPleQs2",
    "workspace": "miker-rwvlm",
    "project_name": "yolo11_object_detector_w-u4gdf",
    "version": 2
}



def download_dataset():

    rf = Roboflow(api_key=DATA_CONFIG["api_key"])
    project = rf.workspace(DATA_CONFIG["workspace"]).project(DATA_CONFIG["project_name"])
    version = project.version(DATA_CONFIG["version"])

    dataset = version.download("yolov11")

    print("\nDataset downloaded to:", dataset.location)
    yaml_path = f"{dataset.location}/data.yaml"

    print("Looking for YAML:", yaml_path)

    return yaml_path

def run_experiments():
    data_path = download_dataset()

    for exp in EXPERIMENTS:

        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M")
        exp_folder = f"results/{exp['experiment_name']}_{timestamp}"
        os.makedirs(exp_folder, exist_ok=True)

        with open(os.path.join(exp_folder, "config.yaml"), "w") as f:
            yaml.dump(exp, f)

        model = YOLO(exp["model_name"])

        model.train(
            data=data_path,
            epochs=exp["epochs"],
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            project=exp_folder,
            name="training",
        )

        best_model_path = f"{exp_folder}/training/weights/best.pt"
        trained_model = YOLO(best_model_path)
        metrics = trained_model.val(data=data_path)

        eval_results = {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "f1": (2 * metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6)
        }

        with open(os.path.join(exp_folder, "metrics.yaml"), "w") as f:
            yaml.dump(eval_results, f)

        print("Saved in:", exp_folder)
        print(eval_results)


if __name__ == "__main__":
    run_experiments()
