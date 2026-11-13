"""Dense threshold sweep and risk-controlled threshold selection for the light->heavy cascade.
Uses cross-fitted validation detections only (split 'val'); the held-out split is never read."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from stackcore import *
from stack_eval import aligned

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
RES = ROOT / "results_v3"
ap = argparse.ArgumentParser()
ap.add_argument("--light", default="yolov8n")
ap.add_argument("--heavy", default="rtdetr_l")
ap.add_argument("--floor", type=float, default=0.25)
ap.add_argument("--det-conf", type=float, default=0.5)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--alphas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
ap.add_argument("--no-map", action="store_true")
A = ap.parse_args()
MINC, EM = 0.02, (0, 1, 2)
GRID = np.round(np.arange(0.30, 1.0001, 0.02), 2)


def cf(r):
    k = r["conf"] >= MINC
    b, s, l = r["boxes"][k], r["conf"][k], r["cls"][k]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


def em_detected(r):
    gb, gc = r["gt_boxes"], r["gt_cls"]
    m = np.isin(gc, EM)
    out = np.zeros(len(gc), bool)
    k = r["conf"] >= A.det_conf
    if m.any() and k.any():
        same = gc[m][:, None] == r["cls"][k][None, :]
        out[np.where(m)[0]] = ((iou_mat(gb[m], r["boxes"][k]) >= 0.5) & same).any(1)
    return out


man = pd.read_csv(ROOT / "datasets/folds_v3_manifest.csv")
gmap = dict(zip(man["id"], man["group"]))
folds, meta = [], []
for f in range(5):
    pm = aligned([A.light, A.heavy], f, "val")
    assert pm is not None, f"missing val dets for fold{f}"
    Lr, Hr = pm[A.light], pm[A.heavy]
    cmin = np.full(len(Lr), np.inf)
    empty = np.zeros(len(Lr), bool)
    for i, r in enumerate(Lr):
        c = r["conf"][r["conf"] >= A.floor]
        empty[i] = len(c) == 0
        if len(c):
            cmin[i] = c.min()
        dl, dh = em_detected(r), em_detected(Hr[i])
        meta.append(dict(gid=str(gmap.get(Path(r["image"]).stem, f"solo{f}_{i}")),
                         pop=bool(dh.any()), lost=bool((dh & ~dl).any())))
    folds.append(dict(Ld=[cf(r) for r in Lr], Hd=[cf(r) for r in Hr],
                      gts=[(r["gt_boxes"], r["gt_cls"]) for r in Lr], cmin=cmin, empty=empty))
M = pd.DataFrame(meta)
POP, LOST, GID = M["pop"].values, M["lost"].values, M["gid"].values
CM = np.concatenate([d["cmin"] for d in folds])
EMP = np.concatenate([d["empty"] for d in folds])
print(f"images={len(M)} population(heavy finds an emergency object)={int(POP.sum())} groups={len(set(GID[POP]))}")


def esc_mask(cmin, empty, hi, rule):
    e = cmin < hi
    return (e | empty) if rule == "B" else e


if not A.no_map:
    for tag, key in (("light_only", "Ld"), ("heavy_only", "Hd")):
        v = [evaluate_map(d[key], d["gts"])["map50"] for d in folds]
        print(f"{tag}: mAP50 {np.mean(v):.4f}+-{np.std(v, ddof=1):.4f}")

rows = []
for rule in ("A", "B"):
    for hi in GRID:
        e_all = esc_mask(CM, EMP, hi, rule)
        loss = LOST & ~e_all
        g = pd.Series(loss[POP].astype(float)).groupby(GID[POP]).mean().values
        n = len(g)
        R = float(g.mean())
        row = dict(rule=rule, hi=float(hi), esc=float(e_all.mean()), risk=R, n_groups=n,
                   ucb=R + float(np.sqrt(np.log(1 / A.delta) / (2 * n))))
        if not A.no_map:
            m = []
            for d in folds:
                e = esc_mask(d["cmin"], d["empty"], hi, rule)
                m.append(evaluate_map([d["Hd"][i] if e[i] else d["Ld"][i] for i in range(len(e))], d["gts"])["map50"])
            row.update(map50=float(np.mean(m)), map50_std=float(np.std(m, ddof=1)))
        rows.append(row)
    print(f"rule {rule} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "cascade_threshold_dev.csv", index=False)
for rule in ("A", "B"):
    d = df[df.rule == rule].set_index("hi")
    print(f"\n=== rule {rule} (fixed-sequence selection, delta={A.delta}, det_conf={A.det_conf}, floor={A.floor})")
    if rule == "A" and 0.7 in d.index:
        print("reference frozen HI=0.70:", d.loc[0.7].round(4).to_dict())
    for al in A.alphas:
        sel = None
        for hi in sorted(d.index, reverse=True):
            if d.loc[hi, "ucb"] <= al:
                sel = hi
            else:
                break
        print(f"alpha={al}: " + ("none certified" if sel is None else f"HI={sel} " + str(d.loc[sel].round(4).to_dict())))
print("saved", RES / "cascade_threshold_dev.csv")
