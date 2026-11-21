#!/usr/bin/env python3
"""Build the root README.md and its figures from results_v4_frcnn60/final_report.md (main run),
results_v3/final_report.md (15-epoch Faster R-CNN appendix), results_coco/coco_final.json and the leakage-audit log.

Every number in the figures and tables is parsed from those files; nothing is typed in by hand.
Usage (from the repository root):  python figures_pipeline/build_root_readme.py
"""
import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results_v4_frcnn60" / "final_report.md"        # main run (retrained Faster R-CNN, 60 epochs)
REPORT_V3 = ROOT / "results_v3" / "final_report.md"             # appendix: 15-epoch Faster R-CNN
COCO = ROOT / "results_coco" / "coco_final.json"
CASC_EXT = ROOT / "results_v4_frcnn60" / "cascade_ext_report.md"
AUDIT = ROOT / "docs" / "leakage_audit" / "audit_holdout_v3.txt"
TEMPLATE = Path(__file__).with_name("root_readme_template.md")
FIGDIR = ROOT / "docs" / "readme_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

NAMES = {
    "single_yolov8n": "YOLOv8n", "single_rtdetr_l": "RT-DETR-l", "single_fasterrcnn": "Faster R-CNN (60 ep)",
    "nms_union": "NMS union", "wbf": "WBF", "top1_ensemblenet_style": "Top-1", "softavg": "Soft average",
    "meta_LR": "Stacking: LR", "meta_HGB": "Stacking: HGB", "meta_RF": "Stacking: RF", "meta_SVM": "Stacking: SVM",
    "meta_MLP": "Stacking: MLP", "meta_KNN": "Stacking: KNN", "cascade_frozen": "Cascade (frozen policy)",
}
PAIRS = {
    "cascade_minus_heavy": "Cascade - RT-DETR-l", "cascade_minus_light": "Cascade - YOLOv8n",
    "meta_minus_heavy": "Stacking (LR) - RT-DETR-l", "softavg_minus_heavy": "Soft average - RT-DETR-l",
    "wbf_minus_heavy": "WBF - RT-DETR-l", "meta_minus_wbf": "Stacking (LR) - WBF",
}
C_YOLO, C_RTDETR, C_FRCNN, C_ENS, C_META, C_CASC = "#4C78A8", "#54A24B", "#A0A0A0", "#B279A2", "#E4572E", "#F2A33A"


def color_of(key):
    if key == "single_yolov8n": return C_YOLO
    if key == "single_rtdetr_l": return C_RTDETR
    if key == "single_fasterrcnn": return C_FRCNN
    if key.startswith("meta_"): return C_META
    if key == "cascade_frozen": return C_CASC
    return C_ENS


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight", "axes.titleweight": "bold",
    })


def sections(text):
    out, cur = {}, None
    for ln in text.splitlines():
        if ln.startswith("## "):
            cur = ln[3:].strip(); out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return out


def find(secs, prefix):
    for k, v in secs.items():
        if k.startswith(prefix):
            return k, v
    raise KeyError(prefix)


def table(lines):
    raw = [l for l in lines if l.startswith("|")]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in raw]
    return rows[0], rows[2:], raw


def num(s):
    m = re.match(r"\s*([-+]?\d*\.?\d+)\s*(?:±\s*([\d.]+))?", s)
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


def ci(s):
    m = re.match(r"\[\s*([-+\d.]+),\s*([-+\d.]+)\s*\]", s)
    return (float(m.group(1)), float(m.group(2))) if m else None


def savefig(fig, name):
    fig.savefig(FIGDIR / name, facecolor="white")
    plt.close(fig)


def fig_cv(rows):
    rows = list(rows)
    fig, ax = plt.subplots(figsize=(8.6, 5.9))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        m, sd = num(r[1])
        ax.barh(y, m, xerr=sd, color=color_of(r[0]), edgecolor="white", height=0.68,
                error_kw=dict(ecolor="#555555", capsize=2.5, lw=0.9, alpha=0.65))
        ax.text(m - 0.002, y, f"{m:.4f}", va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES.get(r[0], r[0]) for r in rows])
    ax.set_xlim(0.80, 0.94)
    ax.set_xlabel("mAP50, 5-fold cross-validation (bar = mean, whisker = std over folds); axis starts at 0.80")
    ax.set_title("Phase 1 - detectors, ensembles and stacking (DEV)")
    ax.legend(handles=[Patch(color=C_YOLO, label="YOLOv8n"), Patch(color=C_RTDETR, label="RT-DETR-l"),
                       Patch(color=C_FRCNN, label="Faster R-CNN (60 ep)"),
                       Patch(color=C_ENS, label="Ensembles"), Patch(color=C_META, label="Stacking")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False, fontsize=8.5, columnspacing=1.2)
    savefig(fig, "fig_cv_map50.png")


def fig_cascade(rows):
    pts = []
    for r in rows:
        m, sd = num(r[2])
        pts.append((r[0], float(r[1]), m, sd, float(r[3])))
    pts.sort(key=lambda p: p[4])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot([p[4] for p in pts], [p[2] for p in pts], color="#BBBBBB", lw=1.6, zorder=1)
    flip = 1
    for name, esc, m, sd, ms in pts:
        frozen = name == "band<0.7"
        col = C_YOLO if name == "light_only" else C_RTDETR if name == "heavy_only" else (C_CASC if frozen else "#7F7F7F")
        ax.plot(ms, m, "*" if frozen else "o", ms=17 if frozen else 8, color=col, zorder=3)
        if name == "light_only":
            ax.annotate(f"YOLOv8n only\n{m:.4f}", (ms, m), textcoords="offset points", xytext=(8, 10), fontsize=8.5)
        elif name == "heavy_only":
            ax.annotate(f"RT-DETR-l only\n{m:.4f}", (ms, m), textcoords="offset points", xytext=(-96, -34), fontsize=8.5)
        elif frozen:
            ax.annotate(f"frozen policy {name}\n{esc:.0%} escalated, {m:.4f}", (ms, m), textcoords="offset points",
                        xytext=(22, -48), fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#888888"))
        else:
            ax.annotate(f"{esc:.0%}", (ms, m), textcoords="offset points", xytext=(0, 9 * flip), ha="center", fontsize=7.8,
                        color="#555555")
            flip *= -1
    ax.set_xlabel("expected inference time per image (ms; laptop RTX 5060, timing is unstable)")
    ax.set_ylabel("mAP50 (CV, DEV)")
    ax.set_ylim(0.878, 0.912)
    ax.set_title("Cascade: accuracy versus cost (labels: share of images escalated)")
    savefig(fig, "fig_cascade_tradeoff.png")


def fig_holdout(rows):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        m, _ = num(r[1]); c = ci(r[2])
        ax.barh(y, m, color=color_of(r[0]), edgecolor="white", height=0.68)
        if c:
            ax.errorbar(m, y, xerr=[[m - c[0]], [c[1] - m]], fmt="none", ecolor="#222222", capsize=3, lw=1.2)
        ax.text((c[1] if c else m) + 0.004, y, f"{m:.4f}" + ("" if c else "  (no interval)"), va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES.get(r[0], r[0]) for r in rows])
    ax.set_xlim(0.75, 1.02)
    ax.set_xlabel("mAP50 on the holdout (95% group-bootstrap interval); axis starts at 0.75")
    ax.set_title("Phase 1 - holdout (POST-HOC: consumed earlier, evaluated once under the v4 frozen protocol)", fontsize=9.5)
    savefig(fig, "fig_holdout_map50.png")


def fig_paired(items):
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ys = list(range(len(items)))[::-1]
    for y, (key, d, lo, hi) in zip(ys, items):
        sig = lo > 0 or hi < 0
        col = C_META if sig else "#7F7F7F"
        ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round")
        ax.plot(d, y, "o", color="white", mec=col, mew=2, ms=8)
        ax.text(hi + 0.002, y, f"{d:+.4f}  [{lo:+.4f}, {hi:+.4f}]", va="center", fontsize=8.5)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([PAIRS.get(k, k) for k, *_ in items])
    ax.set_xlim(-0.06, 0.085)
    ax.set_xlabel("paired mAP50 difference on the holdout (95% bootstrap interval); red = interval excludes 0")
    ax.set_title("Phase 1 - paired differences (POST-HOC)")
    savefig(fig, "fig_holdout_paired.png")


def fig_cost(rows):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    cols = {"yolov8n": C_YOLO, "rtdetr_l": C_RTDETR, "fasterrcnn": C_FRCNN}
    labels = {"yolov8n": "YOLOv8n", "rtdetr_l": "RT-DETR-l", "fasterrcnn": "Faster R-CNN"}
    for ax, (idx, title) in zip(axes, [(1, "parameters (M)"), (2, "GFLOPs"), (3, "ms per image")]):
        for i, r in enumerate(rows):
            v = float(r[idx])
            ax.bar(i, v, color=cols[r[0]], width=0.62)
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([labels[r[0]] for r in rows], rotation=18, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(float(r[idx]) for r in rows) * 1.18)
    fig.suptitle("Model cost (laptop RTX 5060, 100 images, fold 0; timing is unstable)", fontweight="bold", y=1.03)
    savefig(fig, "fig_model_cost.png")


def parse_report(path):
    text = path.read_text(encoding="utf-8")
    secs = sections(text)
    _, cv_lines = find(secs, "5-fold CV")
    _, cas_lines = find(secs, "Cascade")
    _, cost_lines = find(secs, "Cost")
    k_ho, ho_lines = find(secs, "Holdout")
    _, cv_rows, cv_raw = table(cv_lines)
    _, cas_rows, cas_raw = table(cas_lines)
    _, cost_rows, cost_raw = table(cost_lines)
    _, ho_rows, ho_raw = table(ho_lines)
    paired = []
    for ln in ho_lines:
        m = re.match(r"- (\w+): ([+-][\d.]+) CI95 \[([-+\d.e]+), ([-+\d.e]+)\]", ln.strip())
        if m:
            paired.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return dict(proto=text.splitlines()[2].strip(), k_ho=k_ho, cv_rows=cv_rows, cv_raw=cv_raw, cas_rows=cas_rows,
                cas_raw=cas_raw, cost_rows=cost_rows, cost_raw=cost_raw, ho_rows=ho_rows, ho_raw=ho_raw, paired=paired)


def coco_tokens():
    import json
    d = json.loads(COCO.read_text(encoding="utf-8"))
    pt, ci_, pr, esc = d["point"], d["map50_ci95"], d["paired"], d["esc_rate"]
    names = [("single_yolov8n", "YOLOv8n"), ("single_rtdetr_l", "RT-DETR-l"), ("single_fasterrcnn", "Faster R-CNN (COCO-pretrained)"),
             ("nms_union", "NMS union"), ("wbf", "WBF"), ("top1_ensemblenet_style", "Top-1"), ("softavg", "Soft average"),
             ("meta_LR", "Stacking: LR")]
    rows = ["| method | mAP50 | 95% interval |", "|---|---|---|"]
    for k, lab in names:
        c = ci_.get(k)
        rows.append(f"| {lab} | {pt[k]['map50']:.4f} | " + (f"[{c[0]:.3f}, {c[1]:.3f}]" if c else "-") + " |")
    keys = [("meta_LR-single_rtdetr_l", "Stacking (LR) - RT-DETR-l"), ("softavg-single_rtdetr_l", "Soft average - RT-DETR-l"),
            ("wbf-single_rtdetr_l", "WBF - RT-DETR-l"), ("meta_LR-wbf", "Stacking (LR) - WBF"),
            ("cascade_band0.7_transfer-single_rtdetr_l", "Cascade (band<0.7, no recalibration) - RT-DETR-l")]
    pl = [f"- {lab}: **{pr[k]['mean']:+.4f}**, 95% interval [{pr[k]['ci95'][0]:+.4f}, {pr[k]['ci95'][1]:+.4f}]" for k, lab in keys]
    crow = ["| risk level rho | escalated | mAP50 | cascade - RT-DETR-l | share of light-to-heavy gap closed (median, 95%) | random routing, same escalation |", "|---|---|---|---|---|---|"]
    for r in ("0.333", "0.500", "0.667"):
        g = pr[f"share of light->heavy gap closed by cascade_r{r}"]; gr = pr[f"share of light->heavy gap closed by random_r{r}"]
        dh = pr[f"cascade_r{r}-single_rtdetr_l"]
        crow.append(f"| {r} | {esc[r]:.1%} | {pt['cascade_r' + r]['map50']:.4f} | {dh['mean']:+.4f} [{dh['ci95'][0]:+.4f}, {dh['ci95'][1]:+.4f}] | "
                    f"{g['median']:.2f} [{g['ci95'][0]:.2f}, {g['ci95'][1]:.2f}] | {gr['median']:+.2f} [{gr['ci95'][0]:+.2f}, {gr['ci95'][1]:+.2f}] |")
    return {"COCO_N": str(d["n_test_images"]), "COCO_TABLE": "\n".join(rows), "COCO_PAIRED": "\n".join(pl), "COCO_CASCADE_TABLE": "\n".join(crow)}


def audit_tokens():
    t = AUDIT.read_text(encoding="utf-8-sig", errors="replace")
    m = re.findall(r"## A = (\S+): (\d+) unique-location images hashed\s+((?:\s+vs [^\n]+\n\s+A images with a near-duplicate in B: \d+ of \d+[^\n]*\n)+)", t)
    rows = ["| A (holdout) | B (training folds) | near-duplicates of A found in B |", "|---|---|---|"]
    for a, n, body in m:
        for b, nb, hit, tot in re.findall(r"vs (\S+): (\d+) images\s+A images with a near-duplicate in B: (\d+) of (\d+)", body):
            rows.append(f"| `{Path(a).parent.name}` | `{Path(b).parent.parent.name}/{Path(b).parent.name}` ({nb} images) | {hit} of {tot} |")
    if len(rows) < 3:
        raise SystemExit("leakage audit log not parsed")
    return {"AUDIT_TABLE": "\n".join(rows)}


def main():
    A, B = parse_report(REPORT), parse_report(REPORT_V3)
    style()
    fig_cv(A["cv_rows"]); fig_cascade(A["cas_rows"]); fig_holdout(list(A["ho_rows"])); fig_paired(A["paired"]); fig_cost(A["cost_rows"])

    cv = {r[0]: r for r in A["cv_rows"]}; ho = {r[0]: r for r in A["ho_rows"]}; cas = {r[0]: r for r in A["cas_rows"]}
    cv3 = {r[0]: r for r in B["cv_rows"]}
    n_img = re.search(r"(\d+) images", A["k_ho"]).group(1)
    boxes = ast.literal_eval(re.search(r"boxes (\{.*?\})", A["k_ho"]).group(1))
    f_new, f_old = num(cv["single_fasterrcnn"][1])[0], num(cv3["single_fasterrcnn"][1])[0]
    tokens = {
        "PROTO_LINE": A["proto"], "V3_PROTO_LINE": B["proto"],
        "CV_YOLO": f"{num(cv['single_yolov8n'][1])[0]:.4f}", "CV_RTDETR": f"{num(cv['single_rtdetr_l'][1])[0]:.4f}",
        "CV_FRCNN": f"{f_new:.4f}", "V3_FRCNN": f"{f_old:.4f}", "FRCNN_DELTA": f"{f_new - f_old:+.4f}",
        "C07_MAP": f"{num(cas['band<0.7'][2])[0]:.4f}", "C07_MS": cas["band<0.7"][3], "C07_ESC": f"{float(cas['band<0.7'][1]):.0%}",
        "LIGHT_MS": cas["light_only"][3], "HEAVY_MS": cas["heavy_only"][3],
        "HO_N": n_img, "HO_BOXES": ", ".join(f"{k} {v}" for k, v in boxes.items()),
        "HO_YOLO": f"{num(ho['single_yolov8n'][1])[0]:.4f}", "HO_RTDETR": f"{num(ho['single_rtdetr_l'][1])[0]:.4f}",
        "HO_FRCNN": f"{num(ho['single_fasterrcnn'][1])[0]:.4f}", "HO_META": f"{num(ho['meta_LR'][1])[0]:.4f}",
        "HO_CASC": f"{num(ho['cascade_frozen'][1])[0]:.4f}",
        "CV_TABLE": "\n".join(A["cv_raw"]), "CASCADE_TABLE": "\n".join(A["cas_raw"]), "COST_TABLE": "\n".join(A["cost_raw"]),
        "HOLDOUT_TABLE": "\n".join(A["ho_raw"]),
        "V3_CV_TABLE": "\n".join(B["cv_raw"]), "V3_HOLDOUT_TABLE": "\n".join(B["ho_raw"]),
        "V3_HOLDOUT_TITLE": B["k_ho"],
        "PAIRED_LIST": "\n".join(f"- {PAIRS.get(k, k)}: **{d:+.4f}**, 95% interval [{lo:+.4f}, {hi:+.4f}]" for k, d, lo, hi in A["paired"]),
        "V3_PAIRED_LIST": "\n".join(f"- {PAIRS.get(k, k)}: **{d:+.4f}**, 95% interval [{lo:+.4f}, {hi:+.4f}]" for k, d, lo, hi in B["paired"]),
    }
    for k, d, lo, hi in A["paired"]:
        tokens["P_" + k.upper()] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    ext = CASC_EXT.read_text(encoding="utf-8")
    tokens["CASCADE_EXT_REPORT"] = re.sub(r"^#+\s*(.*)$", r"**\1**", ext, flags=re.M).strip()
    tokens.update(coco_tokens()); tokens.update(audit_tokens())
    out = TEMPLATE.read_text(encoding="utf-8")
    for k, v in tokens.items():
        out = out.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", out)
    if left:
        raise SystemExit(f"unresolved tokens: {sorted(set(left))}")
    missing = [p for p in re.findall(r'src="([^"]+)"', out) if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"missing images: {missing}")
    (ROOT / "README.md").write_text(out, encoding="utf-8", newline="\n")
    print("README.md written:", len(out.splitlines()), "lines; figures:", sorted(p.name for p in FIGDIR.glob("*.png")))


if __name__ == "__main__":
    main()
