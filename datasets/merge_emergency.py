import shutil
from pathlib import Path
import yaml

BASE = Path.home() / "Portfolio/Smart_Traffic_System/datasets"
OUT  = BASE / "emergency_merged"

SOURCES = [
    {"dir": BASE / "emergency_vehicles_raw", "map": {0: "ambulance", 1: "police"}},
    {"dir": BASE / "ev_bahrain_v6", "map": {0: "ambulance", 1: "ambulance", 2: "firetruck", 3: "firetruck"}},
    {"dir": BASE / "ev_yolo_v1", "map": {0: "ambulance", 1: "ambulance", 2: "firetruck", 3: "firetruck"}},
    {"dir": BASE / "ev_police_v1", "map": {0: "police"}},
    {"dir": BASE / "kitti_yolo", "map": {0: "car"}},
    {"dir": BASE / "uadetrac_yolo", "map": {0: "car"}},
    {"dir": BASE / "visdrone_yolo", "map": {0: "car"}},
    {"dir": BASE / "miotcd_yolo", "map": {0: "car"}},
]

FINAL_CLASSES = ["ambulance", "police", "firetruck", "car"]
NAME_TO_ID = {n: i for i, n in enumerate(FINAL_CLASSES)}

if OUT.exists():
    shutil.rmtree(OUT)
for split in ["train", "valid", "test"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

counter = 0
stats = {c: 0 for c in FINAL_CLASSES}
img_counts = {"train": 0, "valid": 0, "test": 0}

for src in SOURCES:
    sdir = src["dir"]
    cmap = src["map"]
    for split in ["train", "valid", "test"]:
        img_dir = sdir / split / "images"
        lbl_dir = sdir / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            new_lines = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_id = int(parts[0])
                    if old_id not in cmap:
                        continue
                    cname = cmap[old_id]
                    new_id = NAME_TO_ID[cname]
                    stats[cname] += 1
                    new_lines.append(" ".join([str(new_id)] + parts[1:]))
            if not new_lines:
                continue
            counter += 1
            new_stem = f"{sdir.name}_{counter:06d}"
            shutil.copy(img_path, OUT / split / "images" / (new_stem + img_path.suffix))
            with open(OUT / split / "labels" / (new_stem + ".txt"), "w") as f:
                f.write("\n".join(new_lines) + "\n")
            img_counts[split] += 1

data_yaml = {
    "train": "../train/images",
    "val": "../valid/images",
    "test": "../test/images",
    "nc": len(FINAL_CLASSES),
    "names": FINAL_CLASSES,
}
with open(OUT / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print("box counts per class:", stats)
print("image counts per split:", img_counts)
