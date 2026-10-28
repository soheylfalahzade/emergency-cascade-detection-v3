from common import *
from det_utils import *
from stats_utils import reliability

setup_style()
folds = available_folds(MODELS)
if not folds:
    skip("no dets for all models")
rows = {m: [r for f in folds for r in load_rows(m, f)] for m in MODELS}

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6))
for ax, c in zip(axes.ravel(), range(4)):
    for m in MODELS:
        conf, tp, n = match_class(rows[m], c, 0.001)
        rec, prec, ap = pr_curve(conf, tp, n)
        ax.plot(rec, prec, color=MODEL_COL[m], label=f"{MODEL_LABEL[m]} (AP {ap:.3f})")
    ax.set_title(f"{CLASSES[c]} (n={n} GT, {len(folds)} folds pooled)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(loc="lower left")
save(fig, "p1_pr_curves")

fig, ax = plt.subplots(figsize=(4.4, 4.2))
ax.plot([0, 1], [0, 1], "k--", lw=1)
tpl = {}
for m in MODELS:
    cs, ts = [], []
    for c in (0, 1, 2):
        cf, tp, _ = match_class(rows[m], c, 0.25)
        cs.append(cf)
        ts.append(tp)
    cf, tp = np.concatenate(cs), np.concatenate(ts)
    tpl[m] = (cf, tp)
    rel, ece = reliability(cf, tp.astype(float), 10)
    ax.plot([x[3] for x in rel], [x[4] for x in rel], "o-", color=MODEL_COL[m], label=f"{MODEL_LABEL[m]} ECE={ece:.3f}")
ax.set_xlabel("confidence")
ax.set_ylabel("precision")
ax.set_title("Reliability, emergency detections (conf>=0.25)")
ax.legend()
save(fig, "p1_reliability")

fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
for ax, m in zip(axes, MODELS):
    cf, tp = tpl[m]
    bins = np.linspace(0.25, 1.0, 16)
    ax.hist(cf[tp], bins, alpha=0.7, color=PAL["green"], label="true positive")
    ax.hist(cf[~tp], bins, alpha=0.7, color=PAL["red"], label="false positive")
    ax.set_yscale("log")
    ax.set_xlabel("confidence")
    ax.set_title(MODEL_LABEL[m])
    ax.legend()
save(fig, "p1_confidence_tp_fp")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, m in zip(axes, MODELS):
    M = confusion(rows[m])
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=7, color="white" if Mn[i, j] > 0.5 else "black")
    ax.set_xticks(range(5))
    ax.set_xticklabels([c[:4] for c in CLASSES] + ["missed"])
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASSES)
    ax.set_title(MODEL_LABEL[m])
    ax.grid(False)
fig.suptitle("Confusion (rows: true class, conf>=0.25, IoU 0.5; car row affected by incomplete labels)", y=1.02, fontsize=9)
save(fig, "p1_confusion")
