import sys, csv
from pathlib import Path
from ultralytics import YOLO, RTDETR

BASE = Path.home() / "Portfolio/Smart_Traffic_System/datasets/folds"
RESULTS = Path.home() / "Portfolio/Smart_Traffic_System/results"
K = 5

def completed_epochs(run_dir):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    return max(len(rows) - 1, 0)  # minus header

def train_model(model_name, weights_path, epochs, imgsz, batch):
    for fold in range(K):
        data_yaml = BASE / f"fold_{fold}" / "data.yaml"
        run_name = f"{model_name}_fold{fold}"
        run_dir = RESULTS / run_name
        last_pt = run_dir / "weights" / "last.pt"

        done = completed_epochs(run_dir)
        if done >= epochs:
            print(f"=== {run_name}: already complete ({done}/{epochs} epochs), skipping ===")
            continue

        if last_pt.exists() and done > 0:
            print(f"=== {run_name}: resuming from epoch {done} ===")
            model = YOLO(str(last_pt)) if model_name.startswith("yolo") else __import__("ultralytics").RTDETR(str(last_pt))
            model.train(resume=True)
        else:
            print(f"=== {run_name}: starting fresh ===")
            ctor = YOLO if model_name.startswith("yolo") else RTDETR
            model = ctor(weights_path)
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=0,
                project=str(RESULTS),
                name=run_name,
                patience=15,
                exist_ok=True,
            )

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "yolo":
        train_model("yolov8n", str(Path.home() / "Portfolio/Smart_Traffic_System/datasets/yolov8n.pt"), epochs=60, imgsz=640, batch=16)
    elif which == "rtdetr":
        train_model("rtdetr_l", str(Path.home() / "Portfolio/Smart_Traffic_System/datasets/rtdetr-l.pt"), epochs=60, imgsz=640, batch=8)
