import sys, os, json
from pathlib import Path
from collections import Counter
from ultralytics import YOLO, RTDETR
from stats_utils import wilson

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
HOLD = ROOT / "datasets/holdout_clean"
CLASSES = ["ambulance", "police", "firetruck", "car"]
model, folds = sys.argv[1], [int(x) for x in sys.argv[2:]]
device = os.environ.get("DEVICE", "cpu")
ctor = RTDETR if model.startswith("rtdetr") else YOLO

n_box = Counter()
for lp in (HOLD / "labels").iterdir():
    for l in open(lp):
        if l.strip():
            n_box[int(l.split()[0])] += 1
print("holdout boxes:", {CLASSES[k]: v for k, v in sorted(n_box.items())})

out = []
for f in folds:
    w = ROOT / "results_clean" / f"{model}_fold{f}" / "weights" / "best.pt"
    assert w.exists(), w
    m = ctor(str(w)).val(data=str(HOLD / "data.yaml"), imgsz=640, device=device,
                         batch=16, plots=False, verbose=False)
    res = dict(fold=f, mAP50=float(m.box.map50), mAP50_95=float(m.box.map), per_class={})
    for i, c in enumerate(m.box.ap_class_index):
        c = int(c); r = float(m.box.r[i]); n = n_box[c]
        lo, hi = wilson(round(r * n), n)
        res["per_class"][CLASSES[c]] = dict(AP50=float(m.box.ap50[i]), recall=r, n=n,
                                            recall_wilson95=[lo, hi])
    out.append(res)
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
p = ROOT / "results_clean" / f"holdout_clean_{model}.json"
json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
print("saved", p)
