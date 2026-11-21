#!/usr/bin/env python3
"""Paper figures (vector PDF + 300 dpi PNG) from result files only. Nothing typed by hand.
Usage (repo root, traffic_env):  python figures_pipeline/make_paper_figures.py
Inputs : results_v4_frcnn60/{stack_cv,holdout_final}.json, results_coco/{coco_final,coco_extra}.json,
         training results.csv files (see CSV_DIRS below).  Output: docs/paper_figures/
Labels : CV = DEV; holdout = POST-HOC; COCO = FROZEN-ONCE (extra comparisons POST-HOC)."""
import csv, json, glob, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "results_v4_frcnn60"
COCO = ROOT / "results_coco"
OUT = ROOT / "docs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
# where the per-run results.csv files live (first existing wins); override with env PAPER_CSV_ROOT
CSV_ROOT = Path(os.environ.get("PAPER_CSV_ROOT", Path.home() / "Portfolio/Smart_Traffic_System"))
FLAT = Path(os.environ.get("PAPER_CSV_FLAT", ""))   # optional flat folder: results_v3__yolov8n_fold0__results.csv ...

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E6E6E6", "axes.axisbelow": True, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8})
COL = {"single_yolov8n": "#4C78A8", "single_rtdetr_l": "#54A24B", "single_fasterrcnn": "#8C8C8C", "wbf": "#B279A2",
       "softavg": "#B279A2", "meta_LR": "#E4572E", "cascade_frozen": "#F2A33A", "nms_union": "#B279A2"}
NAME = {"single_yolov8n": "YOLOv8n", "single_rtdetr_l": "RT-DETR-l", "single_fasterrcnn": "Faster R-CNN", "wbf": "WBF",
        "softavg": "Soft avg", "meta_LR": "Stacking (LR)", "cascade_frozen": "Cascade", "nms_union": "NMS union"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): v for k, v in r.items()} for r in rows]


def find_csv(rel_dir, name):
    """rel_dir like results_v3/yolov8n_fold0"""
    p = CSV_ROOT / rel_dir / "results.csv"
    if p.exists():
        return p
    if str(FLAT) != ".":
        q = FLAT / (rel_dir.replace("/", "__") + "__results.csv")
        if q.exists():
            return q
    return None


# ---------- 1. training curves (val mAP50-95 per epoch, one line per fold) ----------
def fig_curves():
    spec = [("YOLOv8n", "results_v3/yolov8n_fold%d", "metrics/mAP50-95(B)", COL["single_yolov8n"]),
            ("RT-DETR-l", "results_v3/rtdetr_l_fold%d", "metrics/mAP50-95(B)", COL["single_rtdetr_l"]),
            ("Faster R-CNN", "results_v3_frcnn60/fasterrcnn_fold%d", "map50_95", COL["single_fasterrcnn"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, (lab, pat, key, c) in zip(axes, spec):
        n = 0
        for k in range(5):
            p = find_csv(pat % k, lab)
            if p is None:
                continue
            rows = read_csv(p)
            y = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            ax.plot(range(1, len(y) + 1), y, color=c, lw=1, alpha=0.8)
            b = int(np.argmax(y))
            ax.plot(b + 1, y[b], "o", ms=3, color=c)
            n += 1
        ax.set_title(f"{lab} ({n} folds)", fontsize=9)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("validation mAP50-95")
    fig.suptitle("Training curves on each fold's validation images (DEV); dot = best epoch", fontsize=9, y=1.04)
    save(fig, "fig_training_curves")


# ---------- 2. per-class AP50, CV (DEV) ----------
CLS = ["ambulance", "police", "firetruck"]
METH = ["single_yolov8n", "single_rtdetr_l", "single_fasterrcnn", "wbf", "softavg", "meta_LR"]


def fig_perclass():
    st = jload(V4 / "stack_cv.json")
    ho = jload(V4 / "holdout_final.json")["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    w = 0.13
    for ax, title in zip(axes, ["5-fold CV (DEV): mean ± std over folds", "Holdout (POST-HOC): mean over 5 fold-models"]):
        for i, m in enumerate(METH):
            vals, errs = [], []
            for c in CLS:
                if ax is axes[0]:
                    v = np.array([st[str(f)][m]["ap50_" + c] for f in range(5)])
                    vals.append(v.mean()); errs.append(v.std(ddof=1))
                else:
                    d = ho[m]["ap50_" + c]
                    vals.append(d["mean"]); errs.append(d.get("std_over_training_runs", 0))
            x = np.arange(len(CLS)) + (i - 2.5) * w
            ax.bar(x, vals, w, yerr=errs, color=COL[m], alpha=1 - 0.12 * (i % 2), capsize=1.5,
                   error_kw={"lw": 0.7}, label=NAME[m])
        ax.set_xticks(range(len(CLS))); ax.set_xticklabels(CLS)
        lo = min(p.get_height() - (e or 0) for p, e in zip(ax.patches, [0] * len(ax.patches)))
        ax.set_ylim(max(0.0, np.floor((lo - 0.03) * 20) / 20), 1.01); ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("AP50")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.text(0.5, -0.16, "Holdout error bars: std over the 5 training runs, not a confidence interval. Police has 15 holdout boxes. "
             "Class 'car' omitted: labels incomplete.", ha="center", fontsize=6.5)
    save(fig, "fig_per_class_ap50")


# ---------- 3. COCO: mAP50 with CI + paired vs RT-DETR + cascade frontier ----------
def fig_coco():
    c = jload(COCO / "coco_final.json")
    e = jload(COCO / "coco_extra.json")
    ci = dict(c["map50_ci95"]); ci.update(e.get("map50_ci95", {}))
    pt = {k: (v["map50"] if isinstance(v, dict) else v) for k, v in c["point"].items()}
    pt.update({k: (v["map50"] if isinstance(v, dict) else v) for k, v in e.get("point", {}).items()})
    order = ["single_yolov8n", "single_rtdetr_l", "single_fasterrcnn", "nms_union", "wbf", "softavg", "meta_LR"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1.0]})
    ax = axes[0]
    for i, m in enumerate(order):
        y = len(order) - i
        ax.plot(pt[m], y, "o", color=COL.get(m, "#B279A2"), ms=4)
        if m in ci:
            ax.plot(ci[m], [y, y], color=COL.get(m, "#B279A2"), lw=1.3)
    ax.set_yticks(range(len(order), 0, -1)); ax.set_yticklabels([NAME[m] for m in order])
    ax.set_xlabel("mAP50 (95% bootstrap)"); ax.set_title("COCO test half, 2500 images", fontsize=8.5)
    ax = axes[1]
    pairs = [("meta_LR-single_rtdetr_l", "Stacking − RT-DETR", c), ("softavg-single_rtdetr_l", "Soft avg − RT-DETR", c),
             ("wbf-single_rtdetr_l", "WBF − RT-DETR", c), ("meta_LR-wbf", "Stacking − WBF", c),
             ("cascade_band0.7_transfer-single_rtdetr_l", "Cascade* − RT-DETR", c)]
    for i, (k, lab, src) in enumerate(pairs):
        d = src["paired"][k]; y = len(pairs) - i
        sig = not (d["ci95"][0] <= 0 <= d["ci95"][1])
        ax.plot(d["mean"], y, "o", color="#E4572E" if sig else "#777777", ms=4)
        ax.plot(d["ci95"], [y, y], color="#E4572E" if sig else "#777777", lw=1.3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(pairs), 0, -1)); ax.set_yticklabels([p[1] for p in pairs], fontsize=6.5)
    ax.set_xlabel("paired mAP50 difference"); ax.set_title("red: interval excludes 0", fontsize=8.5)
    ax = axes[2]
    xs, ys = [], []
    for r in ("0.333", "0.500", "0.667"):
        esc = c["esc_rate"][r]; m = pt["cascade_r" + r]
        xs.append(esc * 100); ys.append(m)
        ax.annotate("ρ=" + r, (esc * 100, m), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.plot(xs, ys, "o-", color=COL["cascade_frozen"], ms=4, label="risk-controlled cascade")
    ax.axhline(pt["single_rtdetr_l"], color=COL["single_rtdetr_l"], ls="--", lw=1, label="RT-DETR-l")
    ax.axhline(pt["single_yolov8n"], color=COL["single_yolov8n"], ls="--", lw=1, label="YOLOv8n")
    ax.set_ylim(0.55, 0.86); ax.set_xlabel("images escalated (%)"); ax.set_ylabel("mAP50"); ax.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=1)
    ax.set_title("Cascade on COCO", fontsize=8.5)
    fig.text(0.5, -0.12, "*Cascade = band<0.7 without recalibration. FROZEN-ONCE on the COCO test half; FRCNN/NMS comparisons come from coco_extra.json (POST-HOC). "
             "torchmetrics mAP: not comparable with the COCO leaderboard.", ha="center", fontsize=6.5)
    fig.subplots_adjust(wspace=0.85)
    save(fig, "fig_coco")


# ---------- 4. seed variability: YOLOv8n best-epoch val mAP50 per fold, 3 seeds ----------
def fig_seeds():
    pats = {"seed 0": "results_v3/yolov8n_fold%d", "seed 1": "results_v3/yolov8n_s1_fold%d", "seed 2": "results_v3/yolov8n_s2_fold%d"}
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    got = False
    for j, (lab, pat) in enumerate(pats.items()):
        xs, ys = [], []
        for k in range(5):
            p = find_csv(pat % k, lab)
            if p is None:
                continue
            rows = read_csv(p)
            fit = [float(r["metrics/mAP50-95(B)"]) for r in rows]
            b = int(np.argmax(fit))                       # best-checkpoint rule: mAP50-95
            xs.append(k + (j - 1) * 0.15); ys.append(float(rows[b]["metrics/mAP50(B)"]))
        if ys:
            got = True
            ax.plot(xs, ys, "o", ms=4, label=lab)
    if not got:
        plt.close(fig); print("skip seeds (no csv)"); return
    ax.set_xticks(range(5)); ax.set_xlabel("fold"); ax.set_ylabel("val mAP50 at best epoch")
    ax.set_title("YOLOv8n: seed-to-seed spread per fold (DEV)", fontsize=8.5); ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_seed_variability")


if __name__ == "__main__":
    for f in (fig_curves, fig_perclass, fig_coco, fig_seeds):
        try:
            f()
        except Exception as ex:                       # a missing input must not hide the other figures
            print("FAILED", f.__name__, repr(ex))
    print("PAPER_FIGS_DONE", sorted(p.name for p in OUT.glob("*.pdf")))
