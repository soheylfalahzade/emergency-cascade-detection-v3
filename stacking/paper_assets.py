import pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stackcore import iou_mat, CLASSES
from stats_utils import reliability

R = Path.home() / "Portfolio/Smart_Traffic_System"
OUT = R / "results_v3/paper"; OUT.mkdir(parents=True, exist_ok=True)


def best_map50(p):
    d = pd.read_csv(p); d.columns = [c.strip() for c in d.columns]
    fit = 0.1 * d["metrics/mAP50(B)"].values + 0.9 * d["metrics/mAP50-95(B)"].values
    return float(d["metrics/mAP50(B)"].iloc[int(np.argmax(fit))]), len(d)


print("\n=== (a) val mAP50: OLD random split vs CLEAN split (best epoch of each run)")
rows = []
for model, fl in (("yolov8n", range(5)), ("rtdetr_l", (0, 1))):
    for f in fl:
        po, pn = R / f"results/{model}_fold{f}/results.csv", R / f"results_v3/{model}_fold{f}/results.csv"
        if po.exists() and pn.exists():
            (a, na), (b, nb) = best_map50(po), best_map50(pn)
            rows.append(dict(model=model, fold=f, old_val_map50=a, clean_val_map50=b, old_epochs=na, clean_epochs=nb))
df = pd.DataFrame(rows); print(df.round(4).to_string(index=False))
for m, g in df.groupby("model"):
    print(f"{m}: old {g.old_val_map50.mean():.4f}±{g.old_val_map50.std(ddof=1):.4f} | clean {g.clean_val_map50.mean():.4f}±{g.clean_val_map50.std(ddof=1):.4f} (n={len(g)})")
print("CAVEAT: val sets and pool composition differ between the two splits, so the gap is not a pure leakage effect.")
df.to_csv(OUT / "leakage_inflation.csv", index=False)

print("\n=== (b) dataset statistics (originals only, labels from pool_v3)")
man = pd.read_csv(R / "datasets/folds_v3_manifest.csv")
print(man.groupby(["set", "src"]).size().unstack(fill_value=0).to_string())
cnt = {}
for _, r in man.iterrows():
    p = R / "datasets/pool_v3/labels" / (r["id"] + ".txt")
    if p.exists():
        for ln in p.read_text().splitlines():
            t = ln.split()
            if len(t) >= 5:
                k = (r["set"], CLASSES[int(t[0])]); cnt[k] = cnt.get(k, 0) + 1
bx = pd.Series(cnt).unstack(fill_value=0); print("\nboxes per class:"); print(bx.to_string())
man.groupby(["set", "src"]).size().unstack(fill_value=0).to_csv(OUT / "images_per_source_set.csv"); bx.to_csv(OUT / "boxes_per_class_set.csv")

MODELS = ["yolov8n", "rtdetr_l", "fasterrcnn"]


def load(m, f):
    p = R / f"results_v3/dets/{m}_fold{f}_val.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


FOLDS = [f for f in range(5) if all((R / f"results_v3/dets/{m}_fold{f}_val.pkl").exists() for m in MODELS)]
print("\nfolds with dets for all models:", FOLDS)
D = {m: [r for f in FOLDS for r in load(m, f)] for m in MODELS}


def det_correct(rows, thr=0.25):
    conf, ok = [], []
    for r in rows:
        k = (r["conf"] >= thr) & (r["cls"] < 3)
        b, s, c = r["boxes"][k], r["conf"][k], r["cls"][k]
        gb, gc = r["gt_boxes"], r["gt_cls"]
        for cl in (0, 1, 2):
            di = np.where(c == cl)[0]; gi = np.where(gc == cl)[0]
            di = di[np.argsort(-s[di])]
            iou = iou_mat(b[di], gb[gi]) if len(di) and len(gi) else None
            used = set()
            for n, i in enumerate(di):
                hit = False
                if iou is not None:
                    row = iou[n].copy()
                    for u in used:
                        row[u] = -1
                    j = int(row.argmax())
                    if row[j] >= 0.5:
                        used.add(j); hit = True
                conf.append(float(s[i])); ok.append(float(hit))
    return np.array(conf), np.array(ok)


print("\n=== (c) calibration of detection confidence (emergency classes only, conf>=0.25, IoU 0.5)")
plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "k--", lw=1)
for m in MODELS:
    c, o = det_correct(D[m]); rel, ece = reliability(c, o, 10)
    plt.plot([x[3] for x in rel], [x[4] for x in rel], "o-", label=f"{m} ECE={ece:.3f}")
    print(f"{m}: n_dets={len(c)} precision={o.mean():.3f} ECE={ece:.3f}")
plt.xlabel("confidence"); plt.ylabel("precision"); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "reliability.png", dpi=150); plt.close()


def confusion(rows, thr=0.25):
    M = np.zeros((4, 5), int); fp = np.zeros(4, int)
    for r in rows:
        k = r["conf"] >= thr
        b, s, c = r["boxes"][k], r["conf"][k], r["cls"][k]
        gb, gc = r["gt_boxes"], r["gt_cls"]
        iou = iou_mat(gb, b) if len(gb) and len(b) else np.zeros((len(gb), len(b)))
        for g in range(len(gb)):
            cand = np.where(iou[g] >= 0.5)[0] if len(b) else []
            if len(cand) == 0:
                M[gc[g], 4] += 1
            else:
                M[gc[g], int(c[cand[np.argmax(s[cand])]])] += 1
        if len(b):
            near = (iou.max(0) >= 0.5) if len(gb) else np.zeros(len(b), bool)
            for i in np.where(~near)[0]:
                fp[int(c[i])] += 1
    return M, fp


print("\n=== (d) error analysis: rows=true class, cols=predicted class | missed (conf>=0.25, IoU 0.5)")
for m in MODELS:
    M, fp = confusion(D[m])
    t = pd.DataFrame(M, index=CLASSES, columns=CLASSES + ["missed"]); t["n"] = M.sum(1)
    print(f"\n{m}"); print(t.to_string())
    print("detections with no GT overlap (by predicted class):", dict(zip(CLASSES, fp.tolist())), "(car FPs inflated by incomplete car labels)")
    t.to_csv(OUT / f"confusion_{m}.csv")
print("\nsaved to", OUT)
