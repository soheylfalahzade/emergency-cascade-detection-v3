import sys, os
for _p in (os.path.expanduser("~/eb_local"), os.path.expanduser("~/eb_local/numba_stub")):
    if _p not in sys.path: sys.path.append(_p)
import sys, json, pickle, argparse
from pathlib import Path
import numpy as np, torch, torchvision
from stackcore import *

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
DETS = ROOT / "results_v3/dets"


def load(m, f, split):
    p = DETS / f"{m}_fold{f}_{split}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def aligned(models, f, split):
    L = [load(m, f, split) for m in models]
    if any(x is None for x in L):
        return None
    ref = [r["image"] for r in L[0]]
    for x in L[1:]:
        assert [r["image"] for r in x] == ref, f"image order mismatch fold{f} {split}"
    return dict(zip(models, L))


def cap(d, n=100):
    out = []
    for b, s, l in d:
        o = np.argsort(-np.asarray(s))[:n]
        out.append((np.asarray(b)[o], np.asarray(s)[o], np.asarray(l)[o]))
    return out


def wbf_fuse(pm, models, i, min_conf, iou_thr=0.55):
    from ensemble_boxes import weighted_boxes_fusion
    bl, sl, ll = [], [], []
    for m in models:
        r = pm[m][i]; k = r["conf"] >= min_conf
        bl.append(np.asarray(r["boxes"], np.float64).reshape(-1, 4)[k]); sl.append(np.asarray(r["conf"], np.float64)[k]); ll.append(np.asarray(r["cls"])[k])
    mx = max([b.max() for b in bl if len(b)] + [1e-9])
    if mx == 1e-9:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
    b, s_, l = weighted_boxes_fusion([(x / mx).clip(0, 1).tolist() for x in bl], [x.tolist() for x in sl], [x.tolist() for x in ll],
                                     iou_thr=iou_thr, skip_box_thr=min_conf)
    return (b * mx).astype(np.float32), s_.astype(np.float32), l.astype(np.int64)


def methods_eval(models, pm, built, clfs, min_conf):
    ref = pm[models[0]]; n = len(ref)
    gts = [(r["gt_boxes"], r["gt_cls"]) for r in ref]
    res = {}
    for m in models:
        d = []
        for r in pm[m]:
            k = r["conf"] >= min_conf
            d.append((r["boxes"][k], r["conf"][k], r["cls"][k]))
        res[f"single_{m}"] = evaluate_map(cap(d), gts)
    d = []
    for i in range(n):
        b = np.concatenate([pm[m][i]["boxes"] for m in models]); s = np.concatenate([pm[m][i]["conf"] for m in models])
        l = np.concatenate([pm[m][i]["cls"] for m in models])
        k = s >= min_conf; b, s, l = b[k], s[k], l[k]
        if len(b):
            keep = torchvision.ops.batched_nms(torch.from_numpy(b.astype(np.float32)), torch.from_numpy(s.astype(np.float32)),
                                               torch.from_numpy(l.astype(np.int64)), 0.5).numpy()
            b, s, l = b[keep], s[keep], l[keep]
        d.append((b, s, l))
    res["nms_union"] = evaluate_map(cap(d), gts)
    res["wbf"] = evaluate_map(cap([wbf_fuse(pm, models, i, min_conf) for i in range(n)]), gts)
    res["top1_ensemblenet_style"] = evaluate_map(cap([fuse_top1(cl) for cl in built["cl"]]), gts)
    res["softavg"] = evaluate_map([fuse_softavg(cl) for cl in built["cl"]], gts)
    for name, clf in clfs.items():
        res[f"meta_{name}"] = evaluate_map(per_image(built, n, proba5(clf, built["X"])), gts)
    return res


def summarize(results, keys=("map50", "map", "ap50_police", "ap50_ambulance")):
    methods = list(next(iter(results.values())))
    print(f"\nn_runs={len(results)}  (mean ± std)")
    print(f"{'method':<28}" + "".join(f"{k:>20}" for k in keys))
    for m in methods:
        cells = []
        for k in keys:
            v = np.array([results[f][m].get(k, np.nan) for f in results], float)
            v = v[(~np.isnan(v)) & (v >= 0)]
            cells.append(f"{v.mean():.4f}±{(v.std(ddof=1) if len(v) > 1 else 0):.4f}" if len(v) else "n/a")
        print(f"{m:<28}" + "".join(f"{c:>20}" for c in cells))


def oof_training(models, use_geom, iou_thr, min_conf, folds=None):
    folds = folds or [f for f in range(5) if aligned(models, f, "val") is not None]
    pms = {f: aligned(models, f, "val") for f in folds}
    built = {f: build_split(pms[f], models, use_geom, iou_thr, min_conf) for f in folds}
    return folds, pms, built


def run_cv(a):
    folds, pms, built = oof_training(a.models, a.geom, a.iou, a.min_conf)
    assert len(folds) >= 2, "need >=2 folds with val dets for all models"
    results = {}
    for f in folds:
        tr = [g for g in folds if g != f]
        X = np.concatenate([built[g]["X"] for g in tr]); y = np.concatenate([built[g]["y"] for g in tr])
        clfs = {n: make_meta(n).fit(X, y) for n in a.metas}
        results[f] = methods_eval(a.models, pms[f], built[f], clfs, a.min_conf)
        print(f"fold{f} done", flush=True)
    summarize(results)
    json.dump(results, open(ROOT / "results_v3/stack_cv.json", "w"), indent=1)


def run_holdout(a):
    if not a.final:
        sys.exit("holdout is evaluated ONCE after freezing models/meta/thresholds. Re-run with --final when frozen.")
    folds, pms, built = oof_training(a.models, a.geom, a.iou, a.min_conf)
    X = np.concatenate([built[g]["X"] for g in folds]); y = np.concatenate([built[g]["y"] for g in folds])
    clf = make_meta(a.meta).fit(X, y)
    ks = [k for k in range(5) if aligned(a.models, k, "holdout") is not None]
    assert ks, "no holdout dets for all models"
    results = {}
    for k in ks:
        pm = aligned(a.models, k, "holdout")
        b = build_split(pm, a.models, a.geom, a.iou, a.min_conf)
        results[k] = methods_eval(a.models, pm, b, {a.meta: clf}, a.min_conf)
        print(f"holdout with fold{k} models done", flush=True)
    summarize(results)
    json.dump(results, open(ROOT / f"results_v3/stack_holdout_{a.meta}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "holdout"])
    ap.add_argument("--models", nargs="+", default=["yolov8n", "rtdetr_l", "fasterrcnn"])
    ap.add_argument("--metas", nargs="+", default=["LR", "HGB"])
    ap.add_argument("--meta", default="HGB")
    ap.add_argument("--geom", action="store_true")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--min_conf", type=float, default=0.02)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    assert len(a.models) >= 2, "stacking needs >=2 base models"
    run_cv(a) if a.mode == "cv" else run_holdout(a)
