import sys, os, json, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision
from stackcore import *
from stack_eval import aligned, cap, oof_training, wbf_fuse

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
RES = ROOT / "results_v3"
FLAG = RES / "holdout_done.flag"
if FLAG.exists():
    sys.exit("holdout was already evaluated once; delete the flag file knowingly to repeat")
P = json.load(open(RES / "protocol_frozen.json"))
MODELS = ["yolov8n", "rtdetr_l", "fasterrcnn"]
LIGHT, HEAVY, META = P["light"], P["heavy"], P["meta"]
HI = float(P["policy"].split("<")[1])
MINC, FLOOR = 0.02, 0.25
BOOT = int(os.environ.get("BOOT", 100))

folds, pms, built = oof_training(MODELS, False, 0.55, MINC)
assert folds == [0, 1, 2, 3, 4], folds
X = np.concatenate([built[g]["X"] for g in folds]); y = np.concatenate([built[g]["y"] for g in folds])
clf = make_meta(META).fit(X, y)
print("meta trained on", len(y), "candidates", flush=True)


def cf(r):
    k = r["conf"] >= MINC
    b, s, l = r["boxes"][k], r["conf"][k], r["cls"][k]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


def nms_union(pm, i):
    b = np.concatenate([pm[m][i]["boxes"] for m in MODELS]); s = np.concatenate([pm[m][i]["conf"] for m in MODELS])
    l = np.concatenate([pm[m][i]["cls"] for m in MODELS])
    k = s >= MINC; b, s, l = b[k], s[k], l[k]
    if len(b):
        keep = torchvision.ops.batched_nms(torch.from_numpy(b.astype(np.float32)), torch.from_numpy(s.astype(np.float32)),
                                           torch.from_numpy(l.astype(np.int64)), 0.5).numpy()
        b, s, l = b[keep], s[keep], l[keep]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


def method_dets(k):
    pm = aligned(MODELS, k, "holdout")
    assert pm is not None, f"missing holdout dets for fold{k}"
    ref = pm[MODELS[0]]; n = len(ref)
    bt = build_split(pm, MODELS, False, 0.55, MINC)
    out = {f"single_{m}": [cf(r) for r in pm[m]] for m in MODELS}
    out["nms_union"] = [nms_union(pm, i) for i in range(n)]
    out["wbf"] = cap([wbf_fuse(pm, MODELS, i, MINC) for i in range(n)])
    out["top1_ensemblenet_style"] = cap([fuse_top1(cl) for cl in bt["cl"]])
    out["softavg"] = [fuse_softavg(cl) for cl in bt["cl"]]
    out[f"meta_{META}"] = per_image(bt, n, proba5(clf, bt["X"]))
    conf = [r["conf"][r["conf"] >= FLOOR] for r in pm[LIGHT]]
    esc = np.array([bool((c < HI).any()) for c in conf])
    out["cascade_frozen"] = [out[f"single_{HEAVY}"][i] if esc[i] else out[f"single_{LIGHT}"][i] for i in range(n)]
    gts = [(r["gt_boxes"], r["gt_cls"]) for r in ref]
    return out, gts, esc, [r["image"] for r in ref]


store, per = {}, {}
for k in folds:
    out, gts, esc, names = method_dets(k)
    store[k] = (out, gts, esc, names)
    per[k] = {m: evaluate_map(d, gts) for m, d in out.items()}
    per[k]["_esc_rate"] = float(esc.mean())
    print(f"holdout point estimates with fold{k} models done", flush=True)
names = store[0][3]
assert all(store[k][3] == names for k in folds), "holdout image order differs between folds"

man = pd.read_csv(ROOT / "datasets/folds_v3_manifest.csv")
gmap = dict(zip(man["id"], man["group"]))
gid = np.array([gmap.get(Path(n).stem, -1 - i) for i, n in enumerate(names)])
print("holdout images:", len(names), "| matched to manifest groups:", int(sum(Path(n).stem in gmap for n in names)))
members = {g: np.where(gid == g)[0] for g in np.unique(gid)}
ug = list(members)
BM = [f"single_{LIGHT}", f"single_{HEAVY}", f"meta_{META}", "cascade_frozen", "softavg", "nms_union", "wbf"]
rng = np.random.RandomState(0)
boot = {m: [] for m in BM}; bootp = {m: [] for m in BM}
for b in range(BOOT):
    idx = np.concatenate([members[ug[j]] for j in rng.randint(0, len(ug), len(ug))])
    vals = {m: [] for m in BM}; pv = {m: [] for m in BM}
    for k in folds:
        out, gts, _, _ = store[k]
        g = [gts[i] for i in idx]
        for m in BM:
            r = evaluate_map([out[m][i] for i in idx], g)
            vals[m].append(r["map50"])
            if r.get("ap50_police", -1) >= 0:
                pv[m].append(r["ap50_police"])
    for m in BM:
        boot[m].append(float(np.mean(vals[m])))
        bootp[m].append(float(np.mean(pv[m])) if pv[m] else float("nan"))
    if b % 10 == 0:
        print(f"bootstrap {b}/{BOOT}", flush=True)


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def mval(m, key):
    v = np.array([per[k][m].get(key, -1) for k in folds], float)
    v = v[v >= 0]
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (None, None)


summary = {}
for m in per[folds[0]]:
    if m.startswith("_"):
        continue
    summary[m] = {key: dict(zip(("mean", "std_over_training_runs"), mval(m, key)))
                  for key in ("map50", "map", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car")}
    if m in BM:
        summary[m]["map50_ci95_group_bootstrap"] = ci(boot[m]); summary[m]["police_ci95_group_bootstrap"] = ci(bootp[m])
pairs = {"cascade_minus_heavy": ("cascade_frozen", f"single_{HEAVY}"), "cascade_minus_light": ("cascade_frozen", f"single_{LIGHT}"),
         f"meta_minus_heavy": (f"meta_{META}", f"single_{HEAVY}"), "softavg_minus_heavy": ("softavg", f"single_{HEAVY}"),
         "wbf_minus_heavy": ("wbf", f"single_{HEAVY}"), "meta_minus_wbf": (f"meta_{META}", "wbf")}
diffs = {n: dict(mean=float(np.mean(np.array(boot[a]) - np.array(boot[b]))), ci95=ci(np.array(boot[a]) - np.array(boot[b])))
         for n, (a, b) in pairs.items()}
cnt = Counter(int(c) for _, gc in store[0][1] for c in gc)
res = dict(protocol=P, n_images=len(names), boxes={CLASSES[c]: v for c, v in sorted(cnt.items())}, bootstrap=BOOT,
           esc_rate_mean=float(np.mean([per[k]["_esc_rate"] for k in folds])), summary=summary, paired_differences=diffs,
           per_training_run={str(k): {m: v for m, v in per[k].items()} for k in folds})
json.dump(res, open(RES / "holdout_final.json", "w"), indent=1)
FLAG.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
print("\nHOLDOUT (mean over 5 fold-models; CI = group bootstrap over images)")
for m, v in summary.items():
    print(f"{m:<26} mAP50={v['map50']['mean']:.4f} (std over runs {v['map50']['std_over_training_runs']:.4f})",
          f"CI95={v.get('map50_ci95_group_bootstrap')}" if "map50_ci95_group_bootstrap" in v else "")
print("paired:", json.dumps(diffs, indent=1))
print("saved holdout_final.json; flag written")
