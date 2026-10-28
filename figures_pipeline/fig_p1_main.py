import json

from common import *

setup_style()
p = RES / "stack_cv.json"
if not p.exists():
    skip("stack_cv.json missing (run post_train.sh)")
cv = json.load(open(p))
folds = sorted(cv, key=int)
methods = list(cv[folds[0]])
KEYS = ("map50", "map", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car")


def vals(m, k):
    v = np.array([cv[f][m].get(k, np.nan) for f in folds], float)
    return v[(~np.isnan(v)) & (v >= 0)]


def ms(v):
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (np.nan, np.nan)


def label(m):
    if m.startswith("single_"):
        return MODEL_LABEL.get(m[7:], m[7:]) + " (single)"
    return {"nms_union": "NMS union", "top1_ensemblenet_style": "Top-1 (EnsembleNet-style)",
            "softavg": "Soft average"}.get(m, m.replace("meta_", "Stacking: "))


def col_of(m):
    if m.startswith("single_"):
        return MODEL_COL.get(m[7:], PAL["grey"])
    return PAL["purple"] if m.startswith("meta_") else PAL["grey"]


lines = ["method," + ",".join(KEYS)]
tex = ["\\begin{tabular}{l" + "c" * len(KEYS) + "}", "\\toprule",
       "Method & mAP50 & mAP50-95 & AP50 amb. & AP50 pol. & AP50 fire. & AP50 car \\\\", "\\midrule"]
for m in methods:
    cells = []
    for k in KEYS:
        mu, sd = ms(vals(m, k))
        cells.append("n/a" if np.isnan(mu) else f"{mu:.3f}$\\pm${sd:.3f}")
    lines.append(label(m).replace(",", ";") + "," + ",".join(c.replace("$\\pm$", "+-") for c in cells))
    tex.append(label(m).replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(P1_OUT / "main_table.csv").write_text("\n".join(lines))
(P1_OUT / "main_table.tex").write_text("\n".join(tex))

order = sorted(methods, key=lambda m: ms(vals(m, "map50"))[0] if len(vals(m, "map50")) else -1)
fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(order) + 1.2))
for i, m in enumerate(order):
    v = vals(m, "map50")
    mu, sd = ms(v)
    ax.barh(i, mu, xerr=sd, color=col_of(m), alpha=0.85, error_kw=dict(lw=1))
    ax.scatter(v, [i] * len(v), s=10, color="black", zorder=3)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([label(m) for m in order])
ax.set_xlim(max(0.0, np.nanmin([ms(vals(m, "map50"))[0] for m in order]) - 0.08), 1.0)
ax.set_xlabel("mAP50 (mean ± std over folds; dots = folds)")
ax.set_title(f"Detection fusion comparison, {len(folds)} folds (val)")
save(fig, "p1_methods_map50")

singles = [f"single_{m}" for m in MODELS if f"single_{m}" in methods]
metas = [m for m in methods if m.startswith("meta_") and len(vals(m, "map50"))]
best_meta = max(metas, key=lambda m: ms(vals(m, "map50"))[0]) if metas else None
sel = singles + (["softavg"] if "softavg" in methods else []) + ([best_meta] if best_meta else [])
fig, ax = plt.subplots(figsize=(7, 3.4))
w = 0.8 / len(sel)
for i, m in enumerate(sel):
    mus = [ms(vals(m, f"ap50_{c}"))[0] for c in CLASSES]
    sds = [ms(vals(m, f"ap50_{c}"))[1] for c in CLASSES]
    ax.bar(np.arange(4) + (i - (len(sel) - 1) / 2) * w, mus, w, yerr=sds, color=col_of(m), alpha=0.9,
           label=label(m), error_kw=dict(lw=0.8))
ax.set_xticks(range(4))
ax.set_xticklabels(CLASSES)
ax.set_ylabel("AP50")
ax.set_ylim(0.4, 1.02)
ax.legend(ncol=2, loc="lower left")
ax.set_title("Per-class AP50 (val)")
fig.text(0.5, -0.04, "car AP: val car labels (UA-DETRAC) are incomplete; police: about 27 boxes per fold, wide intervals",
         ha="center", fontsize=7)
save(fig, "p1_per_class_ap")
