#!/usr/bin/env python
"""Export CARLA captures (images + gt.jsonl) to a YOLO detection dataset (classes as in Phase 1: 0 ambulance, 1 police, 2 firetruck, 3 car).
Only vehicles with instance pixels (visible) and a minimum box area are labelled. Parked props (not CARLA actors) stay unlabelled.
Usage: python export_yolo_dataset.py --train DIR_OR_DIRS --val DIR --out OUT   (train may be comma-separated)"""
import argparse
import glob
import json
import os
import shutil
from collections import Counter

import cv2

CLS = {"ambulance": 0, "police": 1, "firetruck": 2, "car": 3}


def export(roots, split, out, min_inst, min_area, counts):
    n_img = 0
    for root in roots:
        for gt in sorted(glob.glob(os.path.join(root, "*", "gt.jsonl"))):
            d = os.path.dirname(gt)
            cap = os.path.basename(d)
            for line in open(gt, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                w, h = r["cam"]["w"], r["cam"]["h"]
                rows = []
                for v in r["vehicles"]:
                    x1, y1, x2, y2 = v["bbox"]
                    if v["inst_px"] < min_inst or (x2 - x1) * (y2 - y1) < min_area or v["cls"] not in CLS:
                        continue
                    rows.append("%d %.6f %.6f %.6f %.6f" % (CLS[v["cls"]], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
                    counts[(split, v["cls"])] += 1
                img = cv2.imread(os.path.join(d, r["file"]))
                if img is None:
                    continue
                name = "%s_%s_%s" % (os.path.basename(os.path.normpath(root)), cap, r["file"][:-4])
                cv2.imwrite(os.path.join(out, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                open(os.path.join(out, "labels", split, name + ".txt"), "w").write("\n".join(rows) + ("\n" if rows else ""))
                n_img += 1
    return n_img


ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-inst", type=int, default=30)
ap.add_argument("--min-area", type=float, default=150.0)
a = ap.parse_args()
if os.path.exists(a.out):
    shutil.rmtree(a.out)
for s in ("train", "val"):
    os.makedirs(os.path.join(a.out, "images", s))
    os.makedirs(os.path.join(a.out, "labels", s))
counts = Counter()
nt = export(a.train.split(","), "train", a.out, a.min_inst, a.min_area, counts)
nv = export(a.val.split(","), "val", a.out, a.min_inst, a.min_area, counts)
open(os.path.join(a.out, "data.yaml"), "w").write(
    "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: ambulance\n  1: police\n  2: firetruck\n  3: car\n" % a.out)
print("images: train=%d val=%d" % (nt, nv))
for sp in ("train", "val"):
    print(sp, {c: counts[(sp, c)] for c in CLS})
print("wrote", os.path.join(a.out, "data.yaml"))
