"""Simulator-image figures for the documentation (WSL). Frame selection rule (fixed in advance): for each weather, capture seed 20,
first frame in which an ambulance has inst_px>0, bbox area>=2000 px and dist<=40 m (fallback: first frame with a visible ambulance)."""
import json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.image import imread

CAP, EV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = sys.argv[4] if len(sys.argv) > 4 else "20"
os.makedirs(OUT, exist_ok=True)
CLS = ["ambulance", "police", "firetruck", "car"]
COL = {"ambulance": "#e41a1c", "police": "#377eb8", "firetruck": "#ff7f00", "car": "#4daf4a"}
TAGS = [("carla_ft", "carla_ft (fine-tuned on CARLA)"), ("yolov8n_fold0", "yolov8n_fold0 (Phase 1, real images only)")]
WEATHERS = ["ClearNoon", "ClearNight", "Fog", "HardRainNoon"]
THR = 0.5


def norm(p):
    return p.replace("\\", "/")


preds = {}
for t, _ in TAGS:
    f = os.path.join(EV, "preds_%s_960.pkl" % t)
    if os.path.exists(f):
        preds[t] = {norm(k): v for k, v in pickle.load(open(f, "rb")).items()}


def pick(w):
    d = os.path.join(CAP, "%s_s%s" % (w, SEED))
    g = os.path.join(d, "gt.jsonl")
    if not os.path.exists(g):
        return None
    rows = [json.loads(l) for l in open(g, encoding="utf-8") if l.strip()]
    def amb(r):
        return [v for v in r["vehicles"] if v["cls"] == "ambulance" and v["inst_px"] > 0]
    for r in rows:
        if any((v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]) >= 2000 and v["dist_m"] <= 40 for v in amb(r)):
            return d, r
    for r in rows:
        if amb(r):
            return d, r
    return None


def draw(ax, img, boxes, title):
    ax.imshow(img)
    for b, label, c in boxes:
        ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=1.4))
        if label:
            ax.text(b[0], max(b[1] - 3, 0), label, color=c, fontsize=6, va="bottom")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


sel = [(w, pick(w)) for w in WEATHERS]
sel = [(w, s) for w, s in sel if s]
if sel:
    fig, axs = plt.subplots(len(sel), 3, figsize=(18, 3.45 * len(sel)), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for i, (w, (d, r)) in enumerate(sel):
        img = imread(os.path.join(d, r["file"]))
        gtb = [(v["bbox"], v["cls"] if v["cls"] != "car" else "", COL[v["cls"]]) for v in r["vehicles"] if v["inst_px"] > 0]
        draw(axs[i, 0], img, gtb, "%s | seed %s | t=%.0f s | ground truth (CARLA actors)" % (w, SEED, r["t"]))
        key = norm(os.path.join(d, r["file"]))
        for j, (t, lab) in enumerate(TAGS):
            if t not in preds or key not in preds[t]:
                axs[i, j + 1].axis("off"); continue
            b, c, k = preds[t][key]
            m = c >= THR
            pb = [(bb, "%s %.2f" % (CLS[kk], cc) if CLS[kk] != "car" else "", COL[CLS[kk]]) for bb, cc, kk in zip(b[m], c[m], k[m])]
            draw(axs[i, j + 1], img, pb, "%s | conf>=%.1f" % (lab, THR))
    fig.suptitle("CARLA frames from the held-out capture (camera pose and seeds not used for training or checkpoint selection). Red = ambulance, blue = police, orange = firetruck, green = car (unlabelled)", fontsize=9)
    fig.supxlabel("CARLA simulation frames; not real-world data", fontsize=7, color="#666", x=0.99, ha="right")
    fig.savefig(os.path.join(OUT, "fig_sim_heldout_frames.png"), dpi=110)
    plt.close(fig)
    print("wrote fig_sim_heldout_frames.png for", [w for w, _ in sel])

J = os.path.join(EV, "carla_detection_eval.json")
if os.path.exists(J):
    R = json.load(open(J))["models"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for panel, key, title in ((0, "emergency_by_distance@0.5", "Emergency vehicle reported as an emergency class, by camera distance"),
                              (1, "emergency_by_weather@0.5", "... by weather")):
        names = None
        for ti, (t, lab) in enumerate(TAGS):
            if t not in R:
                continue
            D = R[t][key]
            names = list(D)
            x = np.arange(len(names)) + (ti - 0.5) * 0.38
            v = [D[n]["emerg_any"] if D[n]["emerg_any"] is not None else 0.0 for n in names]
            axs[panel].bar(x, v, 0.36, label=lab, color="C%d" % ti, alpha=0.8)
            for xi, n in zip(x, names):
                axs[panel].text(xi, 1.02, "n=%d" % D[n]["n"], ha="center", fontsize=6, rotation=90)
        if names:
            axs[panel].set_xticks(range(len(names)))
            axs[panel].set_xticklabels(names, fontsize=8)
        axs[panel].set_ylim(0, 1.25)
        axs[panel].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[panel].set_title(title, fontsize=8)
        axs[panel].grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Held-out CARLA capture, conf>=0.5 (pooled over seeds; frames are not independent)", fontsize=9)
    fig.supxlabel("CARLA simulation frames; not real-world data", fontsize=7, color="#666", x=0.99, ha="right")
    fig.savefig(os.path.join(OUT, "fig_heldout_by_condition.png"), dpi=150)
    plt.close(fig)
    print("wrote fig_heldout_by_condition.png")
