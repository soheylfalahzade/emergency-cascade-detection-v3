import json
from pathlib import Path
import numpy as np

RES = Path.home() / "Portfolio/Smart_Traffic_System/results_v3"
cv = json.load(open(RES / "stack_cv.json")); cc = json.load(open(RES / "cascade_cv.json"))
P = json.load(open(RES / "protocol_frozen.json")); B = json.load(open(RES / "bench.json"))
H = json.load(open(RES / "holdout_final.json")) if (RES / "holdout_final.json").exists() else None
ms = lambda v: f"{np.mean(v):.4f}±{(np.std(v, ddof=1) if len(v) > 1 else 0):.4f}"
L = ["# Final results (auto-generated)", "",
     f"Frozen protocol ({P['time']}): meta={P['meta']}, heavy={P['heavy']}, cascade={P['policy']} (fallback={P['fallback']})", ""]
folds = sorted(cv); methods = list(cv[folds[0]])
L += [f"## {len(folds)}-fold CV on val images (leave-one-fold-out meta), mean±std over folds", "",
      "| method | mAP50 | mAP50-95 | AP50 police | AP50 ambulance |", "|---|---|---|---|---|"]
for m in methods:
    cells = []
    for k in ("map50", "map", "ap50_police", "ap50_ambulance"):
        v = [cv[f][m][k] for f in folds if cv[f][m].get(k, -1) >= 0]
        cells.append(ms(v) if v else "n/a")
    L.append(f"| {m} | " + " | ".join(cells) + " |")
lm, hm = B[P["light"]]["ms_median"], B[P["heavy"]]["ms_median"]
L += ["", f"## Cascade (light={P['light']}, heavy={P['heavy']}), CV; image-level escalation, full-image heavy model", "",
      "| policy | escalation | mAP50 | expected ms/img |", "|---|---|---|---|"]
for name, v in cc["res"][P["heavy"]].items():
    e = float(np.mean([x["esc"] for x in v]))
    t = lm if name == "light_only" else hm if name == "heavy_only" else lm + e * hm
    L.append(f"| {name} | {e:.2f} | {ms([x['map50'] for x in v])} | {t:.1f} |")
L += ["", f"## Cost (GPU {B['gpu']}, {B['n_images']} images, fold{B['fold']})", "", "| model | params (M) | GFLOPs | ms/img median | FPS |", "|---|---|---|---|---|"]
for m in ("yolov8n", "rtdetr_l", "fasterrcnn"):
    b = B[m]
    L.append(f"| {m} | {b['params_M']:.1f} | {('%.1f' % b['flops_G']) if b['flops_G'] else 'n/a'} | {b['ms_median']:.1f} | {b['fps_median']:.1f} |")
if H:
    L += ["", f"## Holdout (evaluated once; {H['n_images']} images, boxes {H['boxes']}); mean over 5 fold-models", "",
          "| method | mAP50 | CI95 (group bootstrap) | AP50 police | police CI95 |", "|---|---|---|---|---|"]
    for m, v in H["summary"].items():
        c = v.get("map50_ci95_group_bootstrap"); pc = v.get("police_ci95_group_bootstrap")
        pm = v["ap50_police"]["mean"]
        L.append(f"| {m} | {v['map50']['mean']:.4f} | {('[%.3f, %.3f]' % tuple(c)) if c and c[0] is not None else '-'} | "
                 f"{('%.3f' % pm) if pm is not None else 'n/a'} | {('[%.3f, %.3f]' % tuple(pc)) if pc and pc[0] is not None else '-'} |")
    L += ["", "Paired differences (bootstrap):"] + [f"- {n}: {d['mean']:+.4f} CI95 {d['ci95']}" for n, d in H["paired_differences"].items()]
L += ["", "## Caveats (fixed text)",
      "- best.pt/early-stop for YOLO and RT-DETR were selected on each fold's val; Faster R-CNN uses the last epoch.",
      "- Metrics: torchmetrics mAP, conf>=0.02, top-100 per image, same code for all methods.",
      "- Police has very few boxes; any police number needs its interval.",
      "- Val car labels (UA-DETRAC) are incomplete; car AP is not a clean metric.",
      "- Cascade escalation is image-level with the full image sent to the heavy model.",
      "- No emergency-vehicle video ground truth exists; video claims are not supported."]
(RES / "final_report.md").write_text("\n".join(L))
print("\n".join(L))
