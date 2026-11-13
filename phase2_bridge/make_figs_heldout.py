#!/usr/bin/env python
"""Figures from the CARLA result files (mean + 95% percentile bootstrap over seeds + per-seed points). Run in WSL (matplotlib).
Every figure carries the watermark 'CARLA simulation - single junction, Town05_Opt'.
Usage: python make_figs_carla.py --dir /mnt/c/Users/Admin/Documents/phase2_bridge --out figs_carla"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WM = "CARLA simulation - single junction (Town05_Opt); number of seeds per group (n) in x labels; not real-world data"
COL = {"fixed": "#888888", "preempt": "#d98e04", "maxpressure": "#2a7ab0", "fuzzy": "#2e8b57"}
LAB = {"amb_stopped_s": "Ambulance stopped time (s)", "norm_stopped_s_per_veh": "Stopped time per ordinary vehicle (s)",
       "max_wait_s": "Max continuous wait (s)", "switches_per_h": "Phase switches per hour",
       "false_preempt_s": "Emergency action without ambulance (s)"}


def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            out.setdefault(r["policy"], {})[r["seed"]] = r
    return out


def ci(v, n=4000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def panel(ax, groups, policies, metric):
    w = 0.8 / len(policies)
    for gi, (gname, byp) in enumerate(groups):
        for pi, p in enumerate(policies):
            rows = byp.get(p, {})
            v = [r[metric] for r in rows.values() if r.get(metric) is not None]
            if not v:
                continue
            m, lo, hi = ci(v)
            x = gi + (pi - (len(policies) - 1) / 2) * w
            ax.bar(x, m, w * 0.9, color=COL.get(p, "#555"), alpha=0.75, label=p if gi == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="k", capsize=3, lw=1)
            ax.scatter(np.full(len(v), x) + np.linspace(-w * 0.25, w * 0.25, len(v)), v, s=8, color="k", alpha=0.45, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups], fontsize=8 if len(groups) <= 4 else 7, rotation=0 if len(groups) <= 4 else 40, ha="center" if len(groups) <= 4 else "right")
    ax.set_title(LAB.get(metric, metric), fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def figure(path, groups, policies, metrics, title):
    groups = [("%s (n=%d)" % (g.replace("_", " "), max(len(r) for r in b.values())), b) for g, b in groups if b]
    if not groups:
        print("skip (no data):", path)
        return
    fig, axs = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, metrics):
        panel(ax, groups, policies, m)
    axs[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.supxlabel(WM, fontsize=7, color="#666", x=0.99, ha="right")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="figs_carla")
    ap.add_argument("--eval-json", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = a.dir
    clean = load(os.path.join(d, "results_lowflow.jsonl"))
    figure(os.path.join(a.out, "fig_carla_clean.png"), [("no perception error", clean)],
           ["fixed", "preempt", "maxpressure", "fuzzy"],
           ["amb_stopped_s", "norm_stopped_s_per_veh", "max_wait_s", "switches_per_h"],
           "Policies on one CARLA junction (all-red 2 s, inflow 0.25 veh/s)")
    figure(os.path.join(a.out, "fig_carla_swept_noise.png"),
           [("none", clean), ("mild (swept)", load(os.path.join(d, "results_lowflow_noise_mild.jsonl"))),
            ("severe (swept)", load(os.path.join(d, "results_lowflow_noise_severe.jsonl")))],
           ["preempt", "fuzzy"], ["amb_stopped_s", "false_preempt_s", "norm_stopped_s_per_veh"],
           "Injected (swept) perception error: parameters are NOT measured rates")
    groups = []
    for f in sorted(glob.glob(os.path.join(d, "results_v10_*.jsonl"))):
        groups.append((os.path.basename(f)[len("results_v10_"):-len(".jsonl")], load(f)))
    figure(os.path.join(a.out, "fig_carla_measured_noise.png"), groups, ["preempt", "fuzzy"],
           ["amb_stopped_s", "false_preempt_s", "norm_stopped_s_per_veh"],
           "Perception error injected at rates MEASURED on CARLA frames, per weather (ft = CARLA-fine-tuned detector, optimistic; p1 = Phase-1 detector)")
    ej = a.eval_json or os.path.join(d, "carla_eval", "carla_detection_eval.json")
    if os.path.exists(ej):
        J = json.load(open(ej))
        thr = "conf>=0.50"
        tags = sorted(J["models"])
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        cls = ["ambulance", "police", "firetruck", "car"]
        w = 0.8 / max(len(tags), 1)
        for ti, t in enumerate(tags):
            S = J["models"][t][thr]
            for ci_, c in enumerate(cls):
                dd = S[c]
                if not dd["n_obs"]:
                    continue
                x = ci_ + (ti - (len(tags) - 1) / 2) * w
                f = dd["found"]
                lo, hi = (dd["found_ci"] or (f, f))
                axs[0].bar(x, f, w * 0.9, color="C%d" % ti, alpha=0.75, label=t if ci_ == 0 else None)
                axs[0].errorbar(x, f, yerr=[[f - lo], [hi - f]], color="k", capsize=3, lw=1)
                axs[0].text(x, 1.04, "n=%d" % dd["n_obs"], ha="center", va="bottom", fontsize=6, rotation=90, color="k")
            fpf = S["false_positives_per_frame"]
            axs[1].bar(np.arange(len(cls)) + (ti - (len(tags) - 1) / 2) * w, [fpf[c] for c in cls], w * 0.9, color="C%d" % ti, alpha=0.75)
        axs[0].set_xticks(range(len(cls)))
        axs[0].set_xticklabels(cls)
        axs[0].set_ylim(0, 1.4)
        axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[0].set_title("Found (IoU>=0.5) at conf>=0.5, per GT class", fontsize=9)
        axs[0].legend(fontsize=8, loc="upper center", ncol=2)
        axs[1].set_xticks(range(len(cls)))
        axs[1].set_xticklabels(cls)
        axs[1].set_title("False positives per frame (no GT actor, IoU<0.3); parked props have no GT", fontsize=9)
        fig.suptitle("Phase-1 detector vs CARLA-fine-tuned detector on CARLA frames\ncarla_ft: checkpoint NOT selected on this set: held-out camera pose and seeds; yolov8n_fold0: trained on real images only", fontsize=9)
        fig.supxlabel("CARLA simulation frames; cluster-bootstrap CI over vehicle tracks; not real-world data", fontsize=7, color="#666", x=0.99, ha="right")
        fig.savefig(os.path.join(a.out, "fig_carla_detection.png"), dpi=200)
        plt.close(fig)
        print("wrote", os.path.join(a.out, "fig_carla_detection.png"))


if __name__ == "__main__":
    main()
