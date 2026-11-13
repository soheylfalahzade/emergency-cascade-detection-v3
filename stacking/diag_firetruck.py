import numpy as np, pandas as pd
from pathlib import Path
import crop_cascade as cc
from stack_eval import aligned
from stackcore import evaluate_map

man = pd.read_csv(cc.ROOT / "datasets/folds_v3_manifest.csv"); print("manifest columns:", list(man.columns))
gmap = dict(zip(man["id"], man["group"]))
SRC = next((c for c in man.columns if c.lower() in ("source", "src", "dataset")), None)
smap = dict(zip(man["id"], man[SRC])) if SRC else {}
MODELS = ["yolov8n", "rtdetr_l"]
pm = {f: aligned(cc.MODELS, f, "holdout") for f in range(5)}
names = [r["image"] for r in pm[0]["yolov8n"]]
gid = np.array([str(gmap.get(Path(n).stem, f"solo{i}")) for i, n in enumerate(names)])
gts = [(r["gt_boxes"], r["gt_cls"]) for r in pm[0]["yolov8n"]]
nft = np.array([int((g[1] == 2).sum()) for g in gts])
DET = {m: {f: [cc.cf(r) for r in pm[f][m]] for f in range(5)} for m in MODELS}


def ap_ft(m, idx):
    v = []
    for f in range(5):
        r = evaluate_map([DET[m][f][i] for i in idx], [gts[i] for i in idx])
        v.append(r.get("ap50_firetruck", np.nan))
    return float(np.nanmean(v))


ALL = np.arange(len(names))
base = {m: ap_ft(m, ALL) for m in MODELS}
print("firetruck AP50, all groups:", {m: round(v, 4) for m, v in base.items()}, "| firetruck boxes:", int(nft.sum()))
rows = []
for g in sorted(set(gid[nft > 0])):
    mem = np.where(gid == g)[0]; keep = np.where(gid != g)[0]
    row = dict(group=g, images=len(mem), firetruck_boxes=int(nft[mem].sum()), source=smap.get(Path(names[mem[0]]).stem) if smap else None)
    for m in MODELS:
        row[f"{m}_AP_without_group"] = ap_ft(m, keep)
    rows.append(row)
df = pd.DataFrame(rows)
df["yolo_gain_if_removed"] = df["yolov8n_AP_without_group"] - base["yolov8n"]
df["rtdetr_gain_if_removed"] = df["rtdetr_l_AP_without_group"] - base["rtdetr_l"]
print(df.sort_values("yolo_gain_if_removed", ascending=False).round(3).to_string(index=False))
