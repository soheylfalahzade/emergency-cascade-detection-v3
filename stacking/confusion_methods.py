"""Confusion matrices of the combined systems on out-of-fold validation data.
Same convention as the single-model figure: rows = true class, columns = predicted class or missed; detections with confidence >= 0.25
and IoU >= 0.5; the highest-confidence matching detection decides the column.
Stacking: leave-one-fold-out meta-learner (the learner selected in the frozen protocol). Cascade: the frozen operating points."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import crop_cascade as cc
import image_cascade_final as ic
from stackcore import make_meta, proba5, per_image
from stack_eval import oof_training

THR, NAMES = 0.25, ["ambulance", "police", "firetruck", "car"]
P = json.load(open(cc.RES / "image_cascade_protocol_frozen.json"))
META = json.load(open(cc.RES / "protocol_frozen.json"))["meta"]
d = ic.dev_data(); GT = d["GT"]


def conf_mat(dets, gts):
    M = np.zeros((4, 5), int)
    for (b, s, l), (gb, gc) in zip(dets, gts):
        k = s >= THR
        for q in range(len(gc)):
            r = int(gc[q])
            if not k.any():
                M[r, 4] += 1; continue
            ok = cc.iou_mat(gb[q:q + 1], b[k])[0] >= 0.5
            if not ok.any():
                M[r, 4] += 1; continue
            j = np.where(ok)[0][np.argmax(s[k][ok])]
            M[r, int(l[k][j])] += 1
    return M


folds, pms, built = oof_training(cc.MODELS, False, 0.55, cc.MINC)
assert list(folds) == [0, 1, 2, 3, 4]
S = {"YOLOv8n": d["LD"], "RT-DETR-l": d["HD"]}
stack = []
for k in folds:
    n = len(d["LD"][k]); assert int(built[k]["img"].max()) < n, "fold order mismatch"
    X = np.concatenate([built[g]["X"] for g in folds if g != k]); y = np.concatenate([built[g]["y"] for g in folds if g != k])
    clf = make_meta(META).fit(X, y)
    stack.append(per_image(built[k], n, proba5(clf, built[k]["X"])))
S[f"Stacking ({META})"] = stack
for al, hi in P["hi"].items():
    if hi is not None:
        S[f"Cascade alpha={al}"] = [ic.cascade(d["L"][k], d["LD"][k], d["HD"][k], hi)[0] for k in range(5)]

MAT = {k: sum(conf_mat(v[f], GT[f]) for f in range(5)) for k, v in S.items()}
print("check against the single-model figure (diagonal ambulance, police, firetruck, car):")
print("  YOLOv8n  ", np.diag(MAT["YOLOv8n"])[:4].tolist(), "expected [2044, 139, 1317, 2219]")
print("  RT-DETR-l", np.diag(MAT["RT-DETR-l"])[:4].tolist(), "expected [2066, 139, 1343, 2562]")
rows = {}
for name, v in S.items():
    per = [cc.metrics(v[f], GT[f], ic.TAU) for f in range(5)]
    rows[name] = {q: float(np.nanmean([p.get(q, np.nan) for p in per])) for q in ("map50", "emerg_map50", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car")}
    print(f"{name:<22}", {q: round(x, 4) for q, x in rows[name].items()})

show = ["YOLOv8n", "RT-DETR-l", f"Stacking ({META})", "Cascade alpha=0.02"]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
for ax, k in zip(axs, show):
    M = MAT[k]; Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=7, color="white" if Mn[i, j] > 0.5 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(["ambu", "poli", "fire", "car", "missed"], fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(NAMES, fontsize=8); ax.set_title(k, fontsize=9)
fig.suptitle("Confusion on out-of-fold validation data (rows: true class; confidence >= 0.25, IoU >= 0.5; car row affected by incomplete labels)", fontsize=9)
out = cc.ROOT / "paper_pack/figs"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_confusion_methods.png", dpi=200); fig.savefig(out / "fig_confusion_methods.pdf")
json.dump({k: v.tolist() for k, v in MAT.items()}, open(cc.RES / "confusion_methods.json", "w"), indent=1)
json.dump(rows, open(cc.RES / "per_class_ap_methods.json", "w"), indent=1)
print("saved", out / "fig_confusion_methods.png")
