import json, pickle
from pathlib import Path
import numpy as np
from stackcore import evaluate_map

R = Path.home() / "Portfolio/Smart_Traffic_System"
D = R / "results_v3/dets"
LIGHT, HEAVIES = "yolov8n", ("rtdetr_l", "fasterrcnn")
MINC, FLOOR = 0.02, 0.25
HIS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load(m, f):
    return pickle.load(open(D / f"{m}_fold{f}_val.pkl", "rb"))


def dets(r):
    k = r["conf"] >= MINC
    b, s, l = r["boxes"][k], r["conf"][k], r["cls"][k]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


folds = [f for f in range(5) if all((D / f"{m}_fold{f}_val.pkl").exists() for m in (LIGHT,) + HEAVIES)]
assert len(folds) >= 2, folds
res = {h: {} for h in HEAVIES}
for f in folds:
    L = load(LIGHT, f)
    gts = [(r["gt_boxes"], r["gt_cls"]) for r in L]
    dl = [dets(r) for r in L]
    conf = [r["conf"][r["conf"] >= FLOOR] for r in L]
    n = len(L)
    light_res = evaluate_map(dl, gts)
    for h in HEAVIES:
        H = load(h, f)
        assert [r["image"] for r in H] == [r["image"] for r in L], "image order mismatch"
        dh = [dets(r) for r in H]
        pol = [("heavy_only", np.ones(n, bool))] + [(f"band<{hi}", np.array([bool((c < hi).any()) for c in conf])) for hi in HIS]
        res[h].setdefault("light_only", []).append(dict(fold=f, esc=0.0, map50=light_res["map50"],
                                                         police=light_res.get("ap50_police", -1), amb=light_res.get("ap50_ambulance", -1)))
        for name, esc in pol:
            fin = [dh[i] if esc[i] else dl[i] for i in range(n)]
            r = evaluate_map(fin, gts)
            res[h].setdefault(name, []).append(dict(fold=f, esc=float(esc.mean()), map50=r["map50"],
                                                    police=r.get("ap50_police", -1), amb=r.get("ap50_ambulance", -1)))
    print(f"fold{f} done", flush=True)
json.dump(dict(light=LIGHT, folds=folds, res=res), open(R / "results_v3/cascade_cv.json", "w"), indent=1)
for h in HEAVIES:
    print(f"\nheavy={h}; folds={folds}")
    print(f"{'policy':<14}{'esc':>7}{'mAP50':>9}{'std':>8}")
    for name, v in res[h].items():
        m = [e["map50"] for e in v]
        print(f"{name:<14}{np.mean([e['esc'] for e in v]):>7.2f}{np.mean(m):>9.4f}{(np.std(m, ddof=1) if len(m) > 1 else 0):>8.4f}")
