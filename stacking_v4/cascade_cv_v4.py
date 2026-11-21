"""Extended cascade study on cross-validation val dets (DEV only; never touches the holdout).

Variants (light = YOLOv8n; heavy models = RT-DETR-l and the retrained Faster R-CNN):
  light_only, rtdetr_only, frcnn_only, wbf_heavy_only          reference points
  r|b1<x      image escalated by the light model's uncertainty band -> RT-DETR-l
  f|b1<x      same, -> Faster R-CNN
  fuse|b1<x   same, -> WBF fusion of RT-DETR-l and Faster R-CNN (both run on escalated images)
  tier2|b1<x|b2<y  light band -> RT-DETR-l; if RT-DETR-l is still uncertain (band y) -> WBF(RT-DETR-l, Faster R-CNN)
Expected cost per image uses the median ms of bench.json. Selection rule (fixed here, before any holdout use):
  REF = max(mean CV mAP50 of rtdetr_only, frcnn_only); candidates = cascades with mean CV mAP50 >= REF - 0.01;
  selected = candidate with the lowest mean expected ms (ties: higher mAP50). No candidate -> selected = None.
"""
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
from stackcore import evaluate_map

R = Path.home() / "Portfolio/Smart_Traffic_System"
RES = R / "results_v4_frcnn60"
D = RES / "dets"
if (RES / "holdout_done.flag").exists():
    sys.exit("holdout already consumed in this run; extended protocol cannot change")
LIGHT, H1, H2 = "yolov8n", "rtdetr_l", "fasterrcnn"
MINC, FLOOR, TOL = 0.02, 0.25, 0.01
HI1 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HI2 = [0.5, 0.7, 0.9]


def load(m, f):
    return pickle.load(open(D / f"{m}_fold{f}_val.pkl", "rb"))


def dets(r):
    k = r["conf"] >= MINC
    b, s, l = r["boxes"][k], r["conf"][k], r["cls"][k]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


def wbf2(ra, rb, iou_thr=0.55):
    from ensemble_boxes import weighted_boxes_fusion
    bl, sl, ll = [], [], []
    for r in (ra, rb):
        k = r["conf"] >= MINC
        bl.append(np.asarray(r["boxes"], np.float64).reshape(-1, 4)[k])
        sl.append(np.asarray(r["conf"], np.float64)[k])
        ll.append(np.asarray(r["cls"])[k])
    mx = max([b.max() for b in bl if len(b)] + [1e-9])
    if mx == 1e-9:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    b, s_, l = weighted_boxes_fusion([(x / mx).clip(0, 1).tolist() for x in bl], [x.tolist() for x in sl],
                                     [x.tolist() for x in ll], iou_thr=iou_thr, skip_box_thr=MINC)
    b = (np.asarray(b) * mx).astype(np.float32); s_ = np.asarray(s_).astype(np.float32); l = np.asarray(l).astype(np.int64)
    o = np.argsort(-s_)[:100]
    return b[o], s_[o], l[o]


def band(confs, hi):
    return np.array([bool((c < hi).any()) for c in confs])


bench = json.load(open(RES / "bench.json"))
ML, MR, MF = (bench[m]["ms_median"] for m in (LIGHT, H1, H2))
folds = [f for f in range(5) if all((D / f"{m}_fold{f}_val.pkl").exists() for m in (LIGHT, H1, H2))]
assert len(folds) == 5, f"need 5 folds of val dets for all three models, got {folds}"

P = {}  # policy -> list of per-fold dicts


def add(name, fold, map50, e1, e2, cost):
    P.setdefault(name, []).append(dict(fold=fold, map50=float(map50), e1=float(e1), e2=float(e2), cost_ms=float(cost)))


for f in folds:
    L, A, B = load(LIGHT, f), load(H1, f), load(H2, f)
    names = [r["image"] for r in L]
    assert names == [r["image"] for r in A] == [r["image"] for r in B], "image order mismatch"
    gts = [(r["gt_boxes"], r["gt_cls"]) for r in L]
    n = len(L)
    dl, da, db = [dets(r) for r in L], [dets(r) for r in A], [dets(r) for r in B]
    wb = [wbf2(A[i], B[i]) for i in range(n)]
    cl = [r["conf"][r["conf"] >= FLOOR] for r in L]
    ca = [r["conf"][r["conf"] >= FLOOR] for r in A]
    add("light_only", f, evaluate_map(dl, gts)["map50"], 0, 0, ML)
    add("rtdetr_only", f, evaluate_map(da, gts)["map50"], 1, 0, MR)
    add("frcnn_only", f, evaluate_map(db, gts)["map50"], 1, 0, MF)
    add("wbf_heavy_only", f, evaluate_map(wb, gts)["map50"], 1, 1, MR + MF)
    for h1 in HI1:
        e1 = band(cl, h1)
        r1 = e1.mean()
        add(f"r|b1<{h1}", f, evaluate_map([da[i] if e1[i] else dl[i] for i in range(n)], gts)["map50"], r1, 0, ML + r1 * MR)
        add(f"f|b1<{h1}", f, evaluate_map([db[i] if e1[i] else dl[i] for i in range(n)], gts)["map50"], r1, 0, ML + r1 * MF)
        add(f"fuse|b1<{h1}", f, evaluate_map([wb[i] if e1[i] else dl[i] for i in range(n)], gts)["map50"], r1, r1, ML + r1 * (MR + MF))
        for h2 in HI2:
            e2 = e1 & band(ca, h2)
            r2 = e2.mean()
            fin = [wb[i] if e2[i] else (da[i] if e1[i] else dl[i]) for i in range(n)]
            add(f"tier2|b1<{h1}|b2<{h2}", f, evaluate_map(fin, gts)["map50"], r1, r2, ML + r1 * MR + r2 * MF)
    print(f"fold{f} done", flush=True)

summary = {}
for k, v in P.items():
    m = [e["map50"] for e in v]
    summary[k] = dict(mean_map50=float(np.mean(m)), std_map50=float(np.std(m, ddof=1)), per_fold_map50=m,
                      mean_e1=float(np.mean([e["e1"] for e in v])), mean_e2=float(np.mean([e["e2"] for e in v])),
                      mean_cost_ms=float(np.mean([e["cost_ms"] for e in v])))
REF = max(summary["rtdetr_only"]["mean_map50"], summary["frcnn_only"]["mean_map50"])
TARGET = REF - TOL
cascades = [k for k in summary if k.split("|")[0] in ("r", "f", "fuse", "tier2")]
cands = [k for k in cascades if summary[k]["mean_map50"] >= TARGET]
selected = min(cands, key=lambda k: (summary[k]["mean_cost_ms"], -summary[k]["mean_map50"])) if cands else None
paired = {}
for k in ("wbf_heavy_only", "rtdetr_only", "frcnn_only"):
    best_single = [max(a, b) for a, b in zip(summary["rtdetr_only"]["per_fold_map50"], summary["frcnn_only"]["per_fold_map50"])]
    d = np.array(summary[k]["per_fold_map50"]) - np.array(best_single)
    paired[k + "_minus_best_single_per_fold"] = dict(per_fold=d.tolist(), mean=float(d.mean()), std=float(d.std(ddof=1)))
out = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), light=LIGHT, heavy1=H1, heavy2=H2, tol=TOL, floor=FLOOR, minc=MINC,
           hi1=HI1, hi2=HI2, bench_ms=dict(light=ML, rtdetr=MR, frcnn=MF), ref_map50=REF, target_map50=TARGET,
           selected=selected, candidates=cands, summary=summary, paired_vs_best_single=paired,
           label="DEV (cross-validation); not holdout")
json.dump(out, open(RES / "cascade_cv_v4.json", "w"), indent=1)
S = Path(__file__).resolve().parent
proto = dict(time=out["time"], rule=__doc__.strip().split("\n"), ref_map50=REF, target_map50=TARGET, selected=selected,
             selected_stats=summary.get(selected) if selected else None,
             sha256={p.name: hashlib.sha256(open(p, "rb").read()).hexdigest()
                     for p in (S / "cascade_cv_v4.py", S / "stackcore.py", S / "dump_dets.py") if p.exists()})
json.dump(proto, open(RES / "protocol_frozen_ext.json", "w"), indent=1)

L_ = ["# Extended cascade (CV, DEV)", "",
      f"Reference single-model mAP50 (best of RT-DETR-l, Faster R-CNN): {REF:.4f}; target = {TARGET:.4f}; selected = {selected}", "",
      "| policy | esc to heavy1 | esc to heavy2 | mAP50 mean±std | expected ms/img |", "|---|---|---|---|---|"]
for k, v in sorted(summary.items(), key=lambda kv: kv[1]["mean_cost_ms"]):
    L_.append(f"| {k} | {v['mean_e1']:.2f} | {v['mean_e2']:.2f} | {v['mean_map50']:.4f}±{v['std_map50']:.4f} | {v['mean_cost_ms']:.1f} |")
L_ += ["", "Paired per-fold differences versus the best single heavy model (DEV):"]
for k, v in paired.items():
    L_.append(f"- {k}: mean {v['mean']:+.4f}, std {v['std']:.4f}, per fold {[round(x, 4) for x in v['per_fold']]}")
(RES / "cascade_ext_report.md").write_text("\n".join(L_))
print("\n".join(L_))