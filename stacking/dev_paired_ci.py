import os, json, time
import numpy as np
import crop_cascade as cc
import image_cascade_final as ic
from stackcore import evaluate_map

BOOT = int(os.environ.get("BOOT", 60))
P = json.load(open(cc.RES / "image_cascade_protocol_frozen.json")); HI = P["hi"]["0.02"]
d = ic.dev_data()
flat = lambda k: [x for f in range(5) for x in d[k][f]]
LD, HD, GT, LR = flat("LD"), flat("HD"), flat("GT"), flat("L")
gid = d["gid"]
E = np.array([bool(((r["conf"] >= ic.LO) & (r["conf"] < HI)).any()) for r in LR]); rate = float(E.mean())
cl = {}
for i, (gb, gc) in enumerate(GT):
    cl.setdefault(gid[i], set()).update(int(c) for c in gc)
members = {g: np.where(gid == g)[0] for g in cl}
stratum = lambda s: "police" if 1 in s else "firetruck" if 2 in s else "ambulance" if 0 in s else "car"
strata = {}
for g, s in cl.items():
    strata.setdefault(stratum(s), []).append(g)
print(f"images={len(GT)} groups={len(cl)} strata={ {k: len(v) for k, v in strata.items()} } hi={HI} esc_rate={rate:.3f}", flush=True)
QS = ("map50", "emerg", "ap50_firetruck", "ap50_ambulance", "ap50_police")


def ev(D, g):
    r = evaluate_map(D, g); r["emerg"] = cc.emerg_of(r)
    return {q: r.get(q, np.nan) for q in QS}


def systems(idx, mask):
    return {"light": [LD[i] for i in idx], "heavy": [HD[i] for i in idx],
            "cascade": [HD[i] if E[i] else LD[i] for i in idx],
            "random": [HD[i] if mask[j] else LD[i] for j, i in enumerate(idx)]}


ALL = np.arange(len(GT))
point = {k: ev(D, [GT[i] for i in ALL]) for k, D in systems(ALL, np.random.RandomState(7).rand(len(ALL)) < rate).items()}
for k, v in point.items():
    print(f"point {k:<8}", {q: round(float(x), 4) for q, x in v.items()})
boot = {k: {q: [] for q in QS} for k in point}
rng = np.random.RandomState(11); t0 = time.time()
for b in range(BOOT):
    idx = np.concatenate([members[gl[j]] for gl in strata.values() for j in rng.randint(0, len(gl), len(gl))])
    g = [GT[i] for i in idx]
    for k, D in systems(idx, rng.rand(len(idx)) < rate).items():
        r = ev(D, g)
        for q in QS:
            boot[k][q].append(r[q])
    if b % 5 == 0:
        print(f"bootstrap {b}/{BOOT} ({time.time() - t0:.0f}s)", flush=True)
ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
B = lambda k, q: np.array(boot[k][q])
res = {}
for q in ("map50", "emerg", "ap50_firetruck"):
    for a, b_ in (("cascade", "random"), ("cascade", "heavy"), ("cascade", "light"), ("heavy", "light")):
        dd = B(a, q) - B(b_, q); res[f"{q}: {a} - {b_}"] = dict(mean=float(np.nanmean(dd)), ci95=ci(dd))
    for nm in ("cascade", "random"):
        gap = (B(nm, q) - B("light", q)) / (B("heavy", q) - B("light", q))
        res[f"{q}: share of light->heavy gap closed by {nm}"] = dict(median=float(np.nanmedian(gap)), ci95=ci(gap))
json.dump(dict(note="pooled out-of-fold validation detections; stratified group bootstrap; cascade threshold selected on the same data (optimistic vs heavy/light); random routing unaffected",
               hi=HI, esc_rate=rate, bootstrap=BOOT, point={k: {q: float(x) for q, x in v.items()} for k, v in point.items()}, paired=res),
          open(cc.RES / "dev_paired_ci.json", "w"), indent=1)
for k, v in res.items():
    print(f"{k:<62}", {a: (round(b, 4) if isinstance(b, float) else [round(x, 3) for x in b]) for a, b in v.items()})
