import json

from common import *

setup_style()
p = RES / "cascade_cv.json"
if not p.exists():
    skip("cascade_cv.json missing")
cc = json.load(open(p))
res, light = cc["res"], cc["light"]
bj = RES / "bench.json"
bench = json.load(open(bj)) if bj.exists() else None
has_bench = bool(bench) and all(k in bench for k in [light] + list(res))
ncols = 2 if has_bench else 1
fig, axes = plt.subplots(len(res), ncols, figsize=(5.2 * ncols, 3.6 * len(res)), squeeze=False)
for r, (heavy, tbl) in enumerate(res.items()):
    pts = []
    for name, v in tbl.items():
        mp_ = np.array([e["map50"] for e in v])
        pts.append((name, float(np.mean([e["esc"] for e in v])), float(mp_.mean()),
                    float(mp_.std(ddof=1)) if len(mp_) > 1 else 0.0))
    pts.sort(key=lambda t: t[1])
    for c in range(ncols):
        ax = axes[r, c]
        if c == 0:
            xs = [q[1] for q in pts]
            xl = "escalation rate (share of images sent to the heavy model)"
        else:
            lm, hm = bench[light]["ms_median"], bench[heavy]["ms_median"]
            xs = [lm if n == "light_only" else hm if n == "heavy_only" else lm + e * hm for n, e, _, _ in pts]
            xl = "expected latency per image (ms, batch 1)"
        ax.errorbar(xs, [q[2] for q in pts], yerr=[q[3] for q in pts], fmt="o-",
                    color=MODEL_COL.get(heavy, PAL["blue"]), capsize=2, lw=1)
        for q, x in zip(pts, xs):
            ax.annotate(q[0].replace("band<", "<"), (x, q[2]), textcoords="offset points", xytext=(3, 4), fontsize=6.5)
        ax.set_xlabel(xl)
        ax.set_ylabel("mAP50")
        ax.set_title(f"cascade: {MODEL_LABEL.get(light, light)} to {MODEL_LABEL.get(heavy, heavy)} "
                     f"({len(cc['folds'])} folds)")
save(fig, "p1_cascade_pareto")
