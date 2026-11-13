import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path.home() / "Portfolio/Smart_Traffic_System"; R = H / "results_v3"
B = json.load(open(R / "image_cascade_bench2.json")); D = json.load(open(R / "dev_paired_ci.json"))["point"]
items = [("light", "light", "YOLOv8n", "C0"), ("cascade_a0.02", "cascade", "Cascade (alpha 0.02)", "C2"), ("heavy", "heavy", "RT-DETR-l", "C3")]
fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, (key, ttl) in zip(axs, (("map50", "mAP50 (four classes)"), ("emerg", "emergency-class mAP50"))):
    for bk, dk, lab, c in items:
        x, p95, y = B[bk]["ms_mean_median_over_rounds"], B[bk]["ms_p95_median"], D[dk][key]
        ax.plot([x, p95], [y, y], c=c, lw=1, alpha=.6)
        ax.plot([x], [y], "o", c=c, label=f"{lab}: mean {x:.1f} ms, p95 {p95:.1f} ms")
        ax.plot([p95], [y], "D", mfc="white", c=c)
    ax.set_xlabel("latency per image (ms)\nfilled = mean, hollow = 95th percentile"); ax.set_ylabel(ttl); ax.grid(alpha=.3)
axs[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Accuracy against latency, pooled out-of-fold validation data (laptop GPU, batch 1)", fontsize=9)
out = H / "paper_pack/figs"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_latency_tradeoff.png", dpi=200); fig.savefig(out / "fig_latency_tradeoff.pdf"); print("saved fig_latency_tradeoff")
