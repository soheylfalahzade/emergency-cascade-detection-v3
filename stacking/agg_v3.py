import json, numpy as np
from pathlib import Path
from ultralytics import YOLO

R = Path.home() / "Portfolio/Smart_Traffic_System"
names = ["ambulance", "police", "firetruck", "car"]
rows = []
for f in range(5):
    w = R / f"results_v3/yolov8n_fold{f}/weights/best.pt"
    m = YOLO(str(w)).val(data=str(R / f"datasets/folds_v3/fold_{f}/data.yaml"), device=0, batch=32,
                         plots=False, verbose=False)
    d = dict(fold=f, P=float(m.box.mp), R=float(m.box.mr), map50=float(m.box.map50), map=float(m.box.map))
    for i, c in enumerate(m.box.ap_class_index):
        d[f"ap50_{names[int(c)]}"] = float(m.box.ap50[i]); d[f"r_{names[int(c)]}"] = float(m.box.r[i]); d[f"p_{names[int(c)]}"] = float(m.box.p[i])
    rows.append(d)
keys = [k for k in rows[0] if k != "fold"]
print(f"{'metric':<18}" + "".join(f"{'fold'+str(r['fold']):>9}" for r in rows) + f"{'mean±std':>18}")
for k in keys:
    v = np.array([r[k] for r in rows])
    print(f"{k:<18}" + "".join(f"{x:>9.3f}" for x in v) + f"{v.mean():>10.3f}±{v.std(ddof=1):.3f}")
json.dump(rows, open(R / "results_v3/yolo_val_5fold.json", "w"), indent=1)
