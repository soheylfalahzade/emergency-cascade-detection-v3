from pathlib import Path
from ultralytics import YOLO

# SET THIS to a real traffic video path before running
VIDEO_PATH = "/mnt/c/Users/Admin/Downloads/15643210_3840_2160_25fps.mp4"

RESULTS = Path.home() / "Portfolio/Smart_Traffic_System/results"
OUT = Path.home() / "Portfolio/Smart_Traffic_System/datasets/bytetrack_out"
OUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(RESULTS / "yolov8n_fold0" / "weights" / "best.pt"))

results = model.track(
    source=VIDEO_PATH,
    tracker="botsort.yaml",
    imgsz=1536,
    conf=0.25,
    iou=0.5,
    persist=True,
    save=True,
    project=str(OUT),
    name="track_run",
    exist_ok=True,
)

track_log = []
for frame_idx, r in enumerate(results):
    if r.boxes.id is None:
        continue
    for box, tid, cls, conf in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.id.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
    ):
        track_log.append({
            "frame": frame_idx,
            "track_id": int(tid),
            "class": int(cls),
            "conf": float(conf),
            "box": box.tolist(),
        })

import json
with open(OUT / "track_log.json", "w") as f:
    json.dump(track_log, f, indent=2)

ids = set(t["track_id"] for t in track_log)
print(f"total frames processed: {frame_idx+1}")
print(f"unique track IDs: {len(ids)}")
