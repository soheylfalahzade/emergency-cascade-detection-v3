import pickle, sys
from pathlib import Path
import numpy as np
from stackcore import evaluate_map

D = Path.home() / "Portfolio/Smart_Traffic_System/results_v3/dets"
LIGHT, HEAVY = "yolov8n", "rtdetr_l"
MINC, FLOOR, RARE = 0.02, 0.25, (0, 1, 2)
HIS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
folds = [int(x) for x in sys.argv[1:]] or [0, 1]


def load(m, f):
    return pickle.load(open(D / f"{m}_fold{f}_val.pkl", "rb"))


def dets(r):
    k = r["conf"] >= MINC
    b, s, l = r["boxes"][k], r["conf"][k], r["cls"][k]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


table = {}
for f in folds:
    L, H = load(LIGHT, f), load(HEAVY, f)
    assert [r["image"] for r in L] == [r["image"] for r in H], "image order mismatch"
    gts = [(r["gt_boxes"], r["gt_cls"]) for r in L]
    dl, dh = [dets(r) for r in L], [dets(r) for r in H]
    conf = [r["conf"][r["conf"] >= FLOOR] for r in L]
    rare = np.array([bool(((r["conf"] >= FLOOR) & np.isin(r["cls"], RARE)).any()) for r in L])
    band = lambda hi: np.array([bool((c < hi).any()) for c in conf])
    pol = [("light_only", np.zeros(len(L), bool)), ("heavy_only", np.ones(len(L), bool)), ("rare_only", rare)]
    pol += [(f"band<{hi}", band(hi)) for hi in HIS] + [(f"rare|band<{hi}", rare | band(hi)) for hi in HIS]
    for name, esc in pol:
        fin = [dh[i] if esc[i] else dl[i] for i in range(len(L))]
        r = evaluate_map(fin, gts)
        table.setdefault(name, []).append((esc.mean(), r["map50"], r.get("ap50_police", np.nan), r.get("ap50_ambulance", np.nan)))
    print(f"fold{f} done", flush=True)

print("\nimage-level escalation to heavy model on the FULL image (conservative cost); mean over folds:", folds)
print(f"{'policy':<18}{'esc_rate':>9}{'mAP50':>8}{'police':>8}{'amb':>8}")
for name, v in table.items():
    a = np.nanmean(np.array(v, float), 0)
    print(f"{name:<18}{a[0]:>9.2f}{a[1]:>8.4f}{a[2]:>8.3f}{a[3]:>8.3f}")
