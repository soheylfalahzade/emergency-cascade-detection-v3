"""Draw detector predictions (blue) and CARLA ground truth (green/red/orange) on the same frames, to check by eye whether the
low 'found' rates are a real domain gap or a pipeline bug. Reads the preds_*.pkl written by eval_on_carla.py.
Usage: python overlay_preds.py <capture_dir> <preds.pkl> [thr=0.25] [n=6]   -> <capture_dir>/predviz/*.jpg"""
import json
import os
import pickle
import sys

import cv2

CLS = ["ambulance", "police", "firetruck", "car"]
GT_COL = {"ambulance": (0, 0, 255), "firetruck": (0, 128, 255), "police": (255, 0, 255), "car": (0, 200, 0)}
d, pk = sys.argv[1], sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
preds = pickle.load(open(pk, "rb"))
index = {os.path.basename(os.path.dirname(k)) + "/" + os.path.basename(k): v for k, v in preds.items()}
cap = os.path.basename(os.path.normpath(d))
rows = [json.loads(l) for l in open(os.path.join(d, "gt.jsonl"), encoding="utf-8") if l.strip()]
emr = [r for r in rows if any(v["cls"] != "car" for v in r["vehicles"])]
pick = ([emr[int(i * (len(emr) - 1) / max(n // 2 - 1, 1))] for i in range(min(n // 2, len(emr)))] if emr else []) + \
       [rows[int(i * (len(rows) - 1) / max(n // 2 - 1, 1))] for i in range(n // 2)]
os.makedirs(os.path.join(d, "predviz"), exist_ok=True)
for r in pick:
    img = cv2.imread(os.path.join(d, r["file"]))
    pr = index.get(cap + "/" + r["file"])
    if img is None or pr is None:
        print("missing image or preds for", r["file"])
        continue
    for v in r["vehicles"]:
        x1, y1, x2, y2 = [int(round(c)) for c in v["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COL.get(v["cls"], (200, 200, 200)), 1)
    b, c, k = pr
    keep = c >= thr
    for (x1, y1, x2, y2), cf, kk in zip(b[keep], c[keep], k[keep]):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 80, 0), 2)
        cv2.putText(img, "%s %.2f" % (CLS[kk], cf), (int(x1), max(int(y1) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 1)
    out = os.path.join(d, "predviz", r["file"].replace(".png", ".jpg"))
    cv2.imwrite(out, img)
    print(out, " GT:", len(r["vehicles"]), " preds>=%.2f:" % thr, int(keep.sum()))
