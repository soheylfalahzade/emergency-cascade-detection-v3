import json, hashlib
from pathlib import Path

H = Path.home() / "Portfolio/Smart_Traffic_System"; R, C = H / "results_v3", H / "results_coco"
OUT = H / "paper_pack"; OUT.mkdir(exist_ok=True)
L = []


def ld(p):
    return json.load(open(p)) if Path(p).exists() else None


def f(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v) for v in x) + "]"
    return "" if x is None else str(x)


def table(rows, cols):
    L.append("| " + " | ".join(cols) + " |"); L.append("|" + "---|" * len(cols))
    for r in rows:
        L.append("| " + " | ".join(f(r.get(c)) for c in cols) + " |")
    L.append("")


def paired(d):
    rows = []
    for n, v in d.items():
        if "ci95" in v:
            rows.append(dict(comparison=n, stat=v.get("mean", v.get("median")), ci95=v["ci95"]))
        elif "emerg_mean" in v:
            rows.append(dict(comparison=n + " [emerg]", stat=v["emerg_mean"], ci95=v["emerg_ci95"]))
            rows.append(dict(comparison=n + " [mAP50]", stat=v["map50_mean"], ci95=v["map50_ci95"]))
        else:
            for q, w in v.items():
                if isinstance(w, dict):
                    rows.append(dict(comparison=f"{n} [{q}]", stat=w.get("mean", w.get("median")), ci95=w.get("ci95")))
    return rows


L += ["# Phase-1 results summary (auto-generated)", "",
      "Provenance labels: DEV = cross-fitted validation data on which choices were made (optimistic for anything selected on it). "
      "FROZEN-ONCE = protocol hash-frozen before the split was read, evaluated once. POST-HOC = defined or computed after results were seen.", ""]
if (R / "final_report.md").exists():
    L += ["## 1. Frozen protocol: CV (DEV) and held-out split (FROZEN-ONCE; band<0.7 cascade, stacking, WBF, soft average)", "",
          (R / "final_report.md").read_text(), ""]

P = ld(R / "image_cascade_protocol_frozen.json")
if P:
    L += ["## 2. Risk-controlled image-level cascade: frozen parameters (derived on DEV)", "",
          f"lo={P['lo']:.4f} tau={P['tau']} delta={P['delta']} alphas={P['alphas']} selected hi={P['hi']}",
          f"light-only loss rate: {P['light_only_risk']}", ""]
    fr = {round(r["hi"], 2): r for r in P["frontier"]}
    table([dict(alpha=a, hi=h, **{k: fr[round(h, 2)][k] for k in ("esc", "lost_groups", "n_groups", "risk", "ucb")})
           for a, h in P["hi"].items() if h is not None and round(h, 2) in fr], ["alpha", "hi", "esc", "lost_groups", "n_groups", "risk", "ucb"])
    table([dict(system=k, **v) for k, v in P["dev"].items()], ["system", "map50", "emerg_map50", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car", "emerg_recall"])

H2 = ld(R / "image_cascade_holdout_final.json")
if H2:
    L += ["## 3. Held-out split, risk-controlled cascade (parameters FROZEN-ONCE; emergency-mAP50 POST-HOC)", "", H2["note"], ""]
    table([dict(method=k, emerg_mAP50=v["emerg_map50"][0], mAP50=v["map50"][0], firetruck=v["ap50_firetruck"][0], emerg_recall=v["emerg_recall"][0],
                loss_rate=v["loss_rate_holdout"], esc=v["esc_rate"]) for k, v in H2["summary"].items()],
          ["method", "emerg_mAP50", "mAP50", "firetruck", "emerg_recall", "loss_rate", "esc"])
S = ld(R / "image_cascade_holdout_ci_stratified.json")
if S:
    L += ["### 3b. Stratified group bootstrap (POST-HOC). Caveat: one group holds 191 of 221 firetruck boxes, so firetruck/emergency intervals are unreliable (upper limits pile up at 1.0)", ""]
    table([dict(method=k, emerg=v["emerg"]["point"], emerg_ci=v["emerg"]["ci95"], mAP50=v["map50"]["point"], mAP50_ci=v["map50"]["ci95"],
                firetruck=v["ap50_firetruck"]["point"]) for k, v in S["summary"].items()], ["method", "emerg", "emerg_ci", "mAP50", "mAP50_ci", "firetruck"])
    table(paired(S["paired"]), ["comparison", "stat", "ci95"])

D = ld(R / "dev_paired_ci.json")
if D:
    L += ["## 4. Pooled out-of-fold validation, cascade (alpha 0.02) vs random routing at the same rate (DEV; stratified group bootstrap)", "", D["note"], f"hi={D['hi']} esc_rate={D['esc_rate']}", ""]
    table([dict(system=k, **v) for k, v in D["point"].items()], ["system", "map50", "emerg", "ap50_firetruck", "ap50_ambulance", "ap50_police"])
    table(paired(D["paired"]), ["comparison", "stat", "ci95"])

Cj, Ce = ld(C / "coco_final.json"), ld(C / "coco_extra.json")
if Cj:
    L += ["## 5. COCO val2017 benchmark: car, motorcycle, bus, truck; COCO-pretrained detectors, no training by us (FROZEN-ONCE on the test half)", "",
          f"test images={Cj['n_test_images']} bootstrap={Cj['bootstrap']} esc_rate={Cj['esc_rate']}", ""]
    table([dict(method=k, mAP50=v["map50"], ci=Cj["map50_ci95"].get(k), mAP50_95=v["map"], car=v["ap50_car"], motorcycle=v["ap50_motorcycle"], bus=v["ap50_bus"],
                truck=v["ap50_truck"]) for k, v in Cj["point"].items()], ["method", "mAP50", "ci", "mAP50_95", "car", "motorcycle", "bus", "truck"])
    table(paired(Cj["paired"]), ["comparison", "stat", "ci95"])
if Ce:
    L += ["### 5b. Against the strongest single model (POST-HOC)", ""]
    table(paired({k: v for k, v in Ce.items() if isinstance(v, dict) and "ci95" in v}), ["comparison", "stat", "ci95"])

B = ld(R / "image_cascade_bench2.json")
if B:
    L += ["## 6. Latency (this laptop only; five rounds in randomised order; absolute values are specific to this machine and power state)", "", B["note"], f"GPU clocks per round: {B['gpu_clocks_per_round']}", ""]
    table([dict(system=k, **v) for k, v in B.items() if isinstance(v, dict)], ["system", "ms_mean_median_over_rounds", "ms_mean_range", "ms_p95_median", "escalation_rate"])

lg = R / "post_logs/strict_eval.log"
if lg.exists():
    L += ["## 7. Similarity-stratified evaluation (EXPLORATORY, POST-HOC): metrics on images without a near neighbour (pHash distance) in training", "", "```"]
    L += [x for x in lg.read_text().splitlines() if x.strip() and "nohup" not in x and "Warning" not in x and "warnings" not in x] + ["```", ""]

srcs = ["image_cascade_protocol_frozen.json", "image_cascade_holdout_final.json", "image_cascade_holdout_ci_stratified.json", "dev_paired_ci.json",
        "image_cascade_bench2.json", "holdout_final.json", "stack_cv.json", "cascade_cv.json", "protocol_frozen.json"]
L += ["## Source files (sha256, first 12)", ""] + [f"- {n}: {hashlib.sha256(open(R / n, 'rb').read()).hexdigest()[:12]}" for n in srcs if (R / n).exists()]
L += [f"- coco_final.json: {hashlib.sha256(open(C / 'coco_final.json', 'rb').read()).hexdigest()[:12]}"] if Cj else []
(OUT / "results_summary.md").write_text("\n".join(L)); print("wrote", OUT / "results_summary.md", len(L), "lines")
