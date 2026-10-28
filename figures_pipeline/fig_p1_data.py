import pandas as pd

from common import *

setup_style()
mf = ROOT / "datasets/folds_v3_manifest.csv"
if not mf.exists():
    skip("folds_v3_manifest.csv missing")
man = pd.read_csv(mf)
order = [s for s in ["fold0", "fold1", "fold2", "fold3", "fold4", "holdout", "train_only"] if s in set(man["set"])]
cnt = man.groupby(["set", "src"]).size().unstack(fill_value=0).reindex(order)
fig, ax = plt.subplots(figsize=(7, 3.6))
left = np.zeros(len(order))
for k, src in enumerate(cnt.columns):
    ax.barh(range(len(order)), cnt[src].values, left=left, label=src, color=plt.cm.tab10(k % 10))
    left += cnt[src].values
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("original images")
ax.legend(fontsize=6, ncol=2)
ax.set_title("Clean split v3: images per set and source")
save(fig, "p1_dataset_images")

L = ROOT / "datasets/pool_v3/labels"
bx = {}
for _, r in man.iterrows():
    f = L / (r["id"] + ".txt")
    if not f.exists():
        continue
    for ln in f.read_text().splitlines():
        t = ln.split()
        if len(t) >= 5 and 0 <= int(t[0]) < 4:
            key = (r["set"], CLASSES[int(t[0])])
            bx[key] = bx.get(key, 0) + 1
if bx:
    bdf = pd.Series(bx).unstack(fill_value=0).reindex(order)
    cols = [c for c in CLASSES if c in bdf.columns]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    w = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(np.arange(len(order)) + (i - (len(cols) - 1) / 2) * w, bdf[c].values, w, color=CLASS_COL[c], label=c)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20)
    ax.set_yscale("log")
    ax.set_ylabel("boxes (originals, log)")
    ax.legend()
    ax.set_title("Boxes per class and set (police is the scarce class)")
    save(fig, "p1_dataset_boxes")

lp = RES / "paper/leakage_inflation.csv"
if lp.exists():
    df = pd.read_csv(lp)
    nm = df["model"].nunique()
    fig, axes = plt.subplots(1, nm, figsize=(3.2 * nm, 3.2), sharey=True, squeeze=False)
    for ax, (m, g) in zip(axes[0], df.groupby("model")):
        for _, r in g.iterrows():
            ax.plot([0, 1], [r["old_val_map50"], r["clean_val_map50"]], "o-", color=MODEL_COL.get(m, PAL["grey"]), alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random split\n(leaky)", "clean split\n(v3)"])
        ax.set_title(MODEL_LABEL.get(m, m))
        ax.set_ylabel("val mAP50")
    fig.text(0.5, -0.05, "val sets differ between the two splits: the gap is not a pure leakage effect", ha="center", fontsize=7)
    save(fig, "p1_leakage_inflation")
