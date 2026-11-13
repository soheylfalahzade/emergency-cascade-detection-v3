"""Stratified group bootstrap for the held-out split (post-hoc correction of the interval procedure; point estimates unchanged).
Image groups are resampled within strata defined by the rarest emergency class in the group (police > firetruck > ambulance >
car-only), so every class is present in every resample; the earlier procedure could drop a class and change what emergency-mAP averages.
Random-routing baselines send the same fraction of images (per fold-model) to the heavy model, chosen at random."""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import crop_cascade as cc
import image_cascade_final as ic
from stackcore import *
from stack_eval import aligned, cap, oof_training, wbf_fuse

RES, ROOT = cc.RES, cc.ROOT
MODELS, LIGHT, HEAVY, MINC = cc.MODELS, cc.LIGHT, cc.HEAVY, cc.MINC
assert (RES / "image_cascade_holdout_done.flag").exists(), "run the held-out evaluation first"
P = json.load(open(RES / "image_cascade_protocol_frozen.json")); PF = json.load(open(RES / "protocol_frozen.json"))
HIF, TAU, META = float(PF["policy"].split("<")[1]), P["tau"], PF["meta"]
BOOT = int(os.environ.get("BOOT", 100))
KEYS = ("map50", "emerg", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car")
man = pd.read_csv(ROOT / "datasets/folds_v3_manifest.csv"); gmap = dict(zip(man["id"], man["group"]))

_, _, built = oof_training(MODELS, False, 0.55, MINC)
X = np.concatenate([built[g]["X"] for g in range(5)]); y = np.concatenate([built[g]["y"] for g in range(5)])
clf = make_meta(META).fit(X, y)

store = {}
for f in range(5):
    pm = aligned(MODELS, f, "holdout"); Lr = pm[LIGHT]; n = len(Lr)
    LDf = [cc.cf(r) for r in Lr]; HDf = [cc.cf(r) for r in pm[HEAVY]]
    out = {f"single_{m}": [cc.cf(r) for r in pm[m]] for m in MODELS}
    bt = build_split(pm, MODELS, False, 0.55, MINC)
    out["softavg"] = [fuse_softavg(cl) for cl in bt["cl"]]
    out["wbf"] = cap([wbf_fuse(pm, MODELS, i, MINC) for i in range(n)])
    out["meta_" + META] = per_image(bt, n, proba5(clf, bt["X"]))
    Ef = [bool((r["conf"][r["conf"] >= 0.25] < HIF).any()) for r in Lr]
    out["cascade_frozen_protocol"] = [HDf[i] if Ef[i] else LDf[i] for i in range(n)]
    for al, hi in P["hi"].items():
        if hi is None:
            continue
        D, e = ic.cascade(Lr, LDf, HDf, hi)
        out[f"cascade_a{al}"] = D
        rr = np.random.RandomState(1000 * f + int(float(al) * 1000)).rand(n) < e
        out[f"random_a{al}"] = [HDf[i] if rr[i] else LDf[i] for i in range(n)]
    store[f] = dict(out=out, gts=[(r["gt_boxes"], r["gt_cls"]) for r in Lr], names=[r["image"] for r in Lr])
    print(f"fold-model {f} built", flush=True)

names = store[0]["names"]
assert all(store[f]["names"] == names for f in range(5)), "holdout image order differs between folds"
gid = np.array([str(gmap.get(Path(nm).stem, f"solo{i}")) for i, nm in enumerate(names)])
gts0 = store[0]["gts"]
members = {g: np.where(gid == g)[0] for g in np.unique(gid)}


def stratum(ix):
    cl = {int(c) for i in ix for c in gts0[i][1]}
    return "police" if 1 in cl else "firetruck" if 2 in cl else "ambulance" if 0 in cl else "car"


strata = {}
for g, ix in members.items():
    strata.setdefault(stratum(ix), []).append(g)
print("groups per stratum:", {k: len(v) for k, v in strata.items()})
print("groups containing each class:", {cc.CLASSES[c]: sum(any(c in gts0[i][1] for i in ix) for ix in members.values()) for c in range(4)})
for f in range(5):
    hh = [cc.hit(g[0], g[1], *store[f]["out"]["single_" + HEAVY][i], TAU) for i, g in enumerate(store[f]["gts"])]
    pop = np.array([h.any() for h in hh]); print(f"fold-model {f}: population groups={len(set(gid[pop]))}")


def score(out_k, f, idx):
    r = evaluate_map([out_k[i] for i in idx], [store[f]["gts"][i] for i in idx])
    r["emerg"] = cc.emerg_of(r)
    return r


METHODS = list(store[0]["out"]); ALL = np.arange(len(names))
pt = {k: [score(store[f]["out"][k], f, ALL) for f in range(5)] for k in METHODS}
point = {k: {q: float(np.nanmean([r.get(q, np.nan) for r in pt[k]])) for q in KEYS} for k in METHODS}
print("point estimates:", {k: round(point[k]["emerg"], 4) for k in METHODS}, flush=True)

rng = np.random.RandomState(1)
boot = {k: {q: [] for q in KEYS} for k in METHODS}
for b in range(BOOT):
    idx = []
    for gl in strata.values():
        for j in rng.randint(0, len(gl), len(gl)):
            idx.extend(members[gl[j]].tolist())
    idx = np.array(idx)
    for k in METHODS:
        rs = [score(store[f]["out"][k], f, idx) for f in range(5)]
        for q in KEYS:
            boot[k][q].append(float(np.nanmean([r.get(q, np.nan) for r in rs])))
    if b % 10 == 0:
        print(f"bootstrap {b}/{BOOT}", flush=True)

ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
summary = {k: {q: dict(point=point[k][q], ci95=ci(boot[k][q])) for q in KEYS} for k in METHODS}
M = "meta_" + META
pairs = [("cascade_a0.015", "single_rtdetr_l"), ("cascade_a0.02", "single_rtdetr_l"), ("cascade_a0.03", "single_rtdetr_l"),
         ("cascade_a0.015", "single_yolov8n"), ("cascade_a0.02", "single_yolov8n"), ("cascade_a0.03", "single_yolov8n"),
         ("cascade_a0.015", "random_a0.015"), ("cascade_a0.02", "random_a0.02"), ("cascade_a0.03", "random_a0.03"),
         ("cascade_a0.02", "cascade_frozen_protocol"), (M, "single_rtdetr_l"), ("wbf", "single_rtdetr_l"),
         ("softavg", "single_rtdetr_l"), (M, "wbf")]
paired = {}
for a, b_ in pairs:
    if a in boot and b_ in boot:
        dd = {q: np.array(boot[a][q]) - np.array(boot[b_][q]) for q in ("emerg", "map50", "ap50_firetruck")}
        paired[f"{a}-{b_}"] = {q: dict(mean=float(np.nanmean(v)), ci95=ci(v)) for q, v in dd.items()}
json.dump(dict(note="post-hoc: stratified group bootstrap replaces the earlier unstratified procedure; point estimates unchanged",
               bootstrap=BOOT, strata={k: len(v) for k, v in strata.items()}, summary=summary, paired=paired),
          open(RES / "image_cascade_holdout_ci_stratified.json", "w"), indent=1)
f3 = lambda c: "[%.3f, %.3f]" % tuple(c)
for k in METHODS:
    s = summary[k]
    print(f"{k:<26} emerg {s['emerg']['point']:.4f} {f3(s['emerg']['ci95'])} | mAP50 {s['map50']['point']:.4f} {f3(s['map50']['ci95'])}"
          f" | firetruck {s['ap50_firetruck']['point']:.3f} {f3(s['ap50_firetruck']['ci95'])}")
for k, d in paired.items():
    print(f"{k:<44} emerg {d['emerg']['mean']:+.4f} {f3(d['emerg']['ci95'])} | mAP50 {d['map50']['mean']:+.4f} {f3(d['map50']['ci95'])}"
          f" | firetruck {d['ap50_firetruck']['mean']:+.3f} {f3(d['ap50_firetruck']['ci95'])}")
