import pickle, sys
from pathlib import Path
import numpy as np
from stackcore import evaluate_map

D = Path.home() / "Portfolio/Smart_Traffic_System/results_v3/dets"
fold = sys.argv[1] if len(sys.argv) > 1 else "0"
for m in ("yolov8n", "rtdetr_l", "fasterrcnn"):
    p = D / f"{m}_fold{fold}_val.pkl"
    if not p.exists():
        print(m, "missing"); continue
    rows = pickle.load(open(p, "rb"))
    dets, gts = [], []
    for r in rows:
        o = np.argsort(-r["conf"])[:100]
        dets.append((r["boxes"][o], r["conf"][o], r["cls"][o]))
        gts.append((r["gt_boxes"], r["gt_cls"]))
    res = evaluate_map(dets, gts)
    print(m, {k: round(v, 3) for k, v in res.items()}, "| images:", len(rows))
