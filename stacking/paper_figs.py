import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path.home() / "Portfolio/Smart_Traffic_System"; R, C = H / "results_v3", H / "results_coco"
O = H / "paper_pack/figs"; O.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.constrained_layout.use": True, "font.size": 9})


def save(fig, n):
    fig.savefig(O / f"{n}.png", dpi=200); fig.savefig(O / f"{n}.pdf"); plt.close(fig); print("saved", n)


P = json.load(open(R / "image_cascade_protocol_frozen.json")); F = sorted(P["frontier"], key=lambda r: r["esc"])
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot([r["esc"] for r in F], [r["risk"] for r in F], c="C0", label="loss rate (validation)")
ax.plot([r["esc"] for r in F], [r["ucb"] for r in F], c="C1", ls="--", label="90% upper bound (exact binomial)")
lo = P["light_only_risk"]; ax.axhline(lo["risk"], c="gray", lw=.8, ls=":", label="light model only")
for a, h in P["hi"].items():
    r = next((x for x in F if h is not None and abs(x["hi"] - h) < 1e-9), None)
    if r:
        ax.scatter([r["esc"]], [r["ucb"]], c="k", zorder=5); ax.annotate(f"alpha={a}", (r["esc"], r["ucb"]), textcoords="offset points", xytext=(6, 6), fontsize=7)
ax.set_xlabel("share of images sent to the heavy model"); ax.set_ylabel("share of image groups where a heavy-model object is lost")
ax.set_title("Risk-controlled cascade on validation data"); ax.legend(fontsize=7); ax.grid(alpha=.3)
save(fig, "fig_risk_frontier")

Cj = json.load(open(C / "coco_final.json")); pt, ci, es = Cj["point"], Cj["map50_ci95"], Cj["esc_rate"]


def yerr(keys):
    p = np.array([pt[k]["map50"] for k in keys]); b = np.array([ci[k] for k in keys])
    return [p - b[:, 0], b[:, 1] - p]


fig, ax = plt.subplots(figsize=(6.4, 4.6))
for k, x, m, c, lab in (("single_yolov8n", 0, "o", "C0", "light model only"), ("single_rtdetr_l", 1, "s", "C3", "heavy model only")):
    ax.errorbar([x], [pt[k]["map50"]], yerr=yerr([k]), fmt=m, c=c, capsize=3, label=lab)
ks = ("0.333", "0.500", "0.667")
ax.errorbar([es[k] for k in ks], [pt[f"cascade_r{k}"]["map50"] for k in ks], yerr=yerr([f"cascade_r{k}" for k in ks]), fmt="^-", c="C2", capsize=3, label="cascade (calibrated threshold)")
ax.errorbar([es[k] for k in ks], [pt[f"random_r{k}"]["map50"] for k in ks], yerr=yerr([f"random_r{k}" for k in ks]), fmt="v--", c="C1", capsize=3, label="random routing, same rate")
ax.errorbar([es["band0.7"]], [pt["cascade_band0.7_transfer"]["map50"]], yerr=yerr(["cascade_band0.7_transfer"]), fmt="D", c="C4", capsize=3, label="band<0.7 (not recalibrated)")
for k, lab, c_ in (("softavg", "soft average", "C5"), ("meta_LR", "stacking (LR)", "C6"), ("single_fasterrcnn", "Faster R-CNN", "C7")):
    ax.axhline(pt[k]["map50"], lw=1.0, c=c_, ls="--", label=f"{lab} ({pt[k]['map50']:.3f})")
ax.set_xlabel("share of images sent to the heavy model"); ax.set_ylabel("mAP50 (car, motorcycle, bus, truck)")
ax.set_title("COCO val2017 test half (2500 images), 95% bootstrap intervals"); ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.3)
save(fig, "fig_coco")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.errorbar([0], [pt["single_yolov8n"]["map50"]], yerr=yerr(["single_yolov8n"]), fmt="o", c="C0", capsize=3, label="light model only")
ax.errorbar([es[k] for k in ks], [pt[f"cascade_r{k}"]["map50"] for k in ks], yerr=yerr([f"cascade_r{k}" for k in ks]), fmt="^-", c="C2", capsize=3, label="cascade (calibrated threshold)")
ax.errorbar([es[k] for k in ks], [pt[f"random_r{k}"]["map50"] for k in ks], yerr=yerr([f"random_r{k}" for k in ks]), fmt="v--", c="C1", capsize=3, label="random routing, same rate")
ax.errorbar([es["band0.7"]], [pt["cascade_band0.7_transfer"]["map50"]], yerr=yerr(["cascade_band0.7_transfer"]), fmt="D", c="C4", capsize=3, label="band<0.7 (not recalibrated)")
ax.axhline(pt["single_rtdetr_l"]["map50"], c="C3", ls="--", label=f"heavy model only, 100% sent ({pt['single_rtdetr_l']['map50']:.3f})")
ax.set_xlim(-0.01, 0.14); ax.set_xlabel("share of images sent to the heavy model"); ax.set_ylabel("mAP50 (car, motorcycle, bus, truck)")
ax.set_title("COCO val2017 test half, zoom on low escalation rates"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3)
save(fig, "fig_coco_zoom")

data = {"VAL": {}, "HOLD": {}}; sec = cur = None
for ln in open(R / "post_logs/strict_eval.log").read().splitlines():
    if ln.startswith("VALIDATION"):
        sec = "VAL"
    elif ln.startswith("HOLDOUT"):
        sec = "HOLD"
    m = re.match(r"d>(-?\d+): images=(\d+)", ln)
    if m and sec:
        cur = int(m.group(1)); data[sec][cur] = {"n": int(m.group(2))}; continue
    m = re.match(r"\s+(light|heavy|cascade)\s+\{(.*)\}", ln)
    if m and sec and cur in data[sec]:
        e = re.search(r"'emerg_map50': np\.float64\(([0-9.eE+-]+)\)", m.group(2))
        if e:
            data[sec][cur][m.group(1)] = float(e.group(1))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, sec_, ttl in ((axs[0], "VAL", "validation (out-of-fold)"), (axs[1], "HOLD", "held-out split (post hoc)")):
    Ts = [t for t in sorted(data[sec_]) if data[sec_][t]["n"] >= 30 and "light" in data[sec_][t]]; n0 = data[sec_][-1]["n"]
    for s, c in (("light", "C0"), ("heavy", "C3"), ("cascade", "C2")):
        ax.plot(range(len(Ts)), [data[sec_][t][s] for t in Ts], "o-", c=c, label=s)
    ax.set_xticks(range(len(Ts))); ax.set_xticklabels([("all" if t < 0 else f"d>{t}") + f"\n{data[sec_][t]['n'] / n0:.0%}" for t in Ts])
    ax.set_title(ttl); ax.set_xlabel("min. pHash distance to training images\n(share of images kept)"); ax.grid(alpha=.3)
    for j_, t in enumerate(Ts):
        if data[sec_][t]["n"] < 150:
            ax.annotate(f"only {data[sec_][t]['n']} images:\nnot reliable", (j_ + 0.3, 0.06), xycoords=("data", "axes fraction"), ha="right", fontsize=7, color="gray")
axs[0].set_ylabel("emergency-mAP50"); axs[0].legend(fontsize=7)
fig.suptitle("Metrics on images without near neighbours in training", fontsize=10)
save(fig, "fig_similarity")
