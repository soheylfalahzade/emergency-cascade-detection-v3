import sys, yaml
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, recall_score, accuracy_score

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
OUT = ROOT / "results/stacking"
NAMES = yaml.safe_load(open(ROOT / "datasets/folds/fold_0/data.yaml"))["names"]
if isinstance(NAMES, dict):
    NAMES = [NAMES[k] for k in sorted(NAMES)]
NC = len(NAMES)
POLICE = NAMES.index("police")
KEY = ["fold", "image", "gt_idx"]
models = sys.argv[1:] or ["yolov8n", "rtdetr_l", "fasterrcnn"]


def load(m):
    fs = [pd.read_csv(f) for f in sorted(OUT.glob(f"scores_{m}_fold*.csv"))]
    return pd.concat(fs) if fs else None


dfs = {m: load(m) for m in models}
dfs = {m: d for m, d in dfs.items() if d is not None}
assert dfs, "no score files found"
common_folds = sorted(set.intersection(*[set(d.fold.unique()) for d in dfs.values()]))
print("models:", list(dfs), "| folds available for all:", common_folds, "| names:", NAMES)
assert len(common_folds) >= 2, "need >=2 folds with all models"

base = None
for m, d in dfs.items():
    d = d[d.fold.isin(common_folds)].copy()
    cols = {f"s{c}": f"{m}_s{c}" for c in range(NC)}; cols["iou_best"] = f"{m}_iou"
    keep = KEY + ["gt_cls", "gt_area"] + list(cols)
    d = d[keep].rename(columns=cols)
    base = d if base is None else base.merge(d.drop(columns=["gt_cls", "gt_area"]), on=KEY, how="inner")
base = base.reset_index(drop=True)
y = base.gt_cls.values; folds = base.fold.values
feat = [f"{m}_s{c}" for m in dfs for c in range(NC)] + ["gt_area"]
X = base[feat].values
print(f"{len(base)} GT boxes, {len(feat)} features\n")


def metrics(yt, yp):
    return dict(acc=accuracy_score(yt, yp),
                f1=f1_score(yt, yp, labels=list(range(NC)), average="macro", zero_division=0),
                police_recall=recall_score(yt, yp, labels=[POLICE], average="macro", zero_division=0))


def single_pred(m, idx):
    S = base.loc[idx, [f"{m}_s{c}" for c in range(NC)]].values
    return np.where(S.max(1) > 0, S.argmax(1), -1)  # -1 = miss


def hard_vote(idx):
    P = np.stack([single_pred(m, idx) for m in dfs], 1)
    out = []
    for row in P:
        v = row[row >= 0]
        out.append(-1 if len(v) == 0 else np.bincount(v, minlength=NC).argmax())
    return np.array(out)


def soft_vote(idx):
    S = sum(base.loc[idx, [f"{m}_s{c}" for c in range(NC)]].values for m in dfs)
    return np.where(S.max(1) > 0, S.argmax(1), -1)


metas = {
    "meta_LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "meta_RF": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=2, random_state=0),
    "meta_MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), max_iter=500, random_state=0)),
}
per_fold = {}
for f in common_folds:
    te = folds == f; tr = ~te; ite = np.where(te)[0]
    r = {}
    for m in dfs:
        r[f"single_{m}"] = metrics(y[te], single_pred(m, te))
    r["hard_vote"] = metrics(y[te], hard_vote(te))
    r["soft_vote"] = metrics(y[te], soft_vote(te))
    for name, mk in metas.items():
        clf = mk().fit(X[tr], y[tr])
        r[name] = metrics(y[te], clf.predict(X[te]))
    per_fold[f] = r

names = list(per_fold[common_folds[0]])
print(f"{'method':<22}" + "".join(f"{k:>24}" for k in ["acc", "macroF1", "police_recall"]))
for n in names:
    cells = []
    for k in ["acc", "f1", "police_recall"]:
        v = np.array([per_fold[f][n][k] for f in common_folds])
        cells.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}")
    print(f"{n:<22}" + "".join(f"{c:>24}" for c in cells))
print("\nمهم: single/vote با miss (بدون detection) خطا حساب می‌شن؛ meta فقط روی GTهایی که حداقل امتیازی دارن عادلانه‌ست؛")
print("false positive ها در این جدول GT-محور نیستن (بعداً در مرحله‌ی cascade ارزیابی می‌شن).")
