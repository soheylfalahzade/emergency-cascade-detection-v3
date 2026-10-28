import glob, os, shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

BASE = Path.home() / "Portfolio/Smart_Traffic_System/datasets"
RAW = BASE / "kitti_raw"
OUT = BASE / "kitti_yolo"

if OUT.exists():
    shutil.rmtree(OUT)
for split in ["train", "valid", "test"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

label_files = sorted((RAW / "labels").glob("*.txt"))
kept = []

for lbl_path in label_files:
    img_path = RAW / "images" / (lbl_path.stem + ".png")
    if not img_path.exists():
        continue
    with open(lbl_path) as f:
        lines = f.readlines()
    car_lines = [l for l in lines if l.split()[0] == "Car"]
    if not car_lines:
        continue
    kept.append((img_path, lbl_path, car_lines))

train_val, test = train_test_split(kept, test_size=0.08, random_state=42)
train, valid = train_test_split(train_val, test_size=0.10, random_state=42)

splits = {"train": train, "valid": valid, "test": test}

for split_name, items in splits.items():
    for img_path, lbl_path, car_lines in items:
        with Image.open(img_path) as im:
            w, h = im.size
        yolo_lines = []
        for line in car_lines:
            parts = line.split()
            left, top, right, bottom = map(float, parts[4:8])
            cx = ((left + right) / 2) / w
            cy = ((top + bottom) / 2) / h
            bw = (right - left) / w
            bh = (bottom - top) / h
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        shutil.copy(img_path, OUT / split_name / "images" / img_path.name)
        with open(OUT / split_name / "labels" / (img_path.stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

print("kept images with Car:", len(kept))
print({k: len(v) for k, v in splits.items()})
