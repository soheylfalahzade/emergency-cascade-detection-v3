import os, shutil, random
from pathlib import Path
from collections import Counter
import yaml

random.seed(42)

BASE = Path.home() / "Portfolio/Smart_Traffic_System/datasets"
SRC = BASE / "emergency_merged"
FOLDS_DIR = BASE / "folds"
K = 5
CLASSES = ["ambulance", "police", "firetruck", "car"]

# oversampling factor: how many times to duplicate a training image
# depending on the rarest class it contains (police=157 boxes is scarcest)
DUP_FACTOR = {"police": 8, "firetruck": 3, "ambulance": 2, "car": 1}
PRIORITY = ["police", "firetruck", "ambulance", "car"]

def collect_pool():
    items = []
    for split in ["train", "valid"]:
        img_dir = SRC / split / "images"
        lbl_dir = SRC / split / "labels"
        for img_path in img_dir.glob("*"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            with open(lbl_path) as f:
                class_ids = set(int(l.split()[0]) for l in f if l.strip())
            class_names = {CLASSES[i] for i in class_ids}
            items.append((img_path, lbl_path, class_names))
    return items

def rarity_key(class_names):
    for c in PRIORITY:
        if c in class_names:
            return c
    return "car"

def stratified_fold_assignment(items, k):
    buckets = {}
    for idx, (img, lbl, cnames) in enumerate(items):
        buckets.setdefault(rarity_key(cnames), []).append(idx)
    fold_of = [None] * len(items)
    for key, idxs in buckets.items():
        random.shuffle(idxs)
        for i, idx in enumerate(idxs):
            fold_of[idx] = i % k
    return fold_of

def main():
    items = collect_pool()
    print("pool size:", len(items))
    fold_of = stratified_fold_assignment(items, K)

    if FOLDS_DIR.exists():
        shutil.rmtree(FOLDS_DIR)

    for fold in range(K):
        fold_path = FOLDS_DIR / f"fold_{fold}"
        for split in ["train", "val"]:
            (fold_path / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_path / split / "labels").mkdir(parents=True, exist_ok=True)

        train_counts = Counter()
        for idx, (img_path, lbl_path, cnames) in enumerate(items):
            split = "val" if fold_of[idx] == fold else "train"
            dup = DUP_FACTOR[rarity_key(cnames)] if split == "train" else 1
            for d in range(dup):
                suffix = "" if d == 0 else f"_dup{d}"
                dst_img = fold_path / split / "images" / (img_path.stem + suffix + img_path.suffix)
                dst_lbl = fold_path / split / "labels" / (img_path.stem + suffix + ".txt")
                if not dst_img.exists():
                    os.symlink(img_path.resolve(), dst_img)
                if not dst_lbl.exists():
                    os.symlink(lbl_path.resolve(), dst_lbl)
                if split == "train":
                    for c in cnames:
                        train_counts[c] += 1

        data_yaml = {
            "train": str((fold_path / "train" / "images").resolve()),
            "val": str((fold_path / "val" / "images").resolve()),
            "nc": len(CLASSES),
            "names": CLASSES,
        }
        with open(fold_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"fold {fold} train class presence (after oversampling): {dict(train_counts)}")

if __name__ == "__main__":
    main()
