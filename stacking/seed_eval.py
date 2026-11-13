"""Seed robustness of the light detector: thresholds, LO and TAU stay frozen (derived with seed 0); heavy detector is not re-trained.
dev = pooled out-of-fold validation (fold-mean metrics); holdout = mean over the 5 fold-models (post hoc, the split was already used)."""
import json
import numpy as np
import crop_cascade as cc
import image_cascade_final as ic
from stack_eval import aligned

SEEDS = {0: "yolov8n", 1: "yolov8n_s1", 2: "yolov8n_s2"}
SEEDS = {k: v for k, v in SEEDS.items() if (cc.RES / "dets" / f"{v}_fold4_holdout.pkl").exists()}
assert 0 in SEEDS and len(SEEDS) > 1, "no completed extra seed yet"
print("seeds available:", list(SEEDS))
SEEDS = {k: v for k, v in SEEDS.items() if (cc.RES / "dets" / f"{v}_fold4_holdout.pkl").exists()}
assert 0 in SEEDS and len(SEEDS) > 1, "no completed extra seed yet"
print("seeds available:", list(SEEDS))
P = json.load(open(cc.RES / "image_cascade_protocol_frozen.json"))
HI = {a: h for a, h in P["hi"].items() if h is not None}
TAU = ic.TAU
M = ("emerg_map50", "map50", "ap50_firetruck", "emerg_recall")
out = {"dev": {}, "holdout": {}}

for s, name in SEEDS.items():
    ic.LIGHT = name
    d = ic.dev_data()
    row = {"light_only": ic.fm(d, d["LD"]), "heavy_only": ic.fm(d, d["HD"])}
    n0, k0 = ic.pooled_loss(d, d["LD"]); row["light_only"]["loss_rate"] = k0 / n0
    for a, hi in HI.items():
        res = [ic.cascade(d["L"][f], d["LD"][f], d["HD"][f], hi) for f in range(5)]
        n, k = ic.pooled_loss(d, [r[0] for r in res])
        c = ic.fm(d, [r[0] for r in res]); c.update(esc=float(np.mean([r[1] for r in res])), loss_rate=k / n, ucb90=ic.cp_ucb(k, n))
        row[f"cascade_a{a}"] = c
        Dr = []
        for f in range(5):
            rr = np.random.RandomState(1000 * s + f).rand(len(d["LD"][f])) < res[f][1]
            Dr.append([d["HD"][f][i] if rr[i] else d["LD"][f][i] for i in range(len(rr))])
        r_ = ic.fm(d, Dr); r_["esc"] = c["esc"]; row[f"random_a{a}"] = r_
    out["dev"][s] = row; print("dev seed", s, flush=True)

for s, name in SEEDS.items():
    acc, esc = {}, {}
    for f in range(5):
        pm = aligned([name, cc.HEAVY], f, "holdout"); Lr, Hr = pm[name], pm[cc.HEAVY]; n = len(Lr)
        LDf = [cc.cf(r) for r in Lr]; HDf = [cc.cf(r) for r in Hr]; gts = [(r["gt_boxes"], r["gt_cls"]) for r in Lr]
        sy = {"light_only": LDf, "heavy_only": HDf}
        for a, hi in HI.items():
            D, e = ic.cascade(Lr, LDf, HDf, hi); sy[f"cascade_a{a}"] = D; esc.setdefault(f"cascade_a{a}", []).append(e)
            rr = np.random.RandomState(1000 * s + f + 77).rand(n) < e
            sy[f"random_a{a}"] = [HDf[i] if rr[i] else LDf[i] for i in range(n)]; esc.setdefault(f"random_a{a}", []).append(e)
        for k, D in sy.items():
            r = cc.metrics(D, gts, TAU); acc.setdefault(k, []).append([r.get(q, np.nan) for q in M])
    out["holdout"][s] = {k: dict(zip(M, np.nanmean(np.array(v), 0).tolist()), **({"esc": float(np.mean(esc[k]))} if k in esc else {}))
                         for k, v in acc.items()}
    print("holdout seed", s, flush=True)


def show(part):
    print(f"\n=== {part}: values for seeds {list(SEEDS)} | mean sd(ddof=1) | min max")
    for sy in out[part][0]:
        for m in M + ("esc", "loss_rate", "ucb90"):
            v = [out[part][s][sy].get(m) for s in SEEDS]
            if any(x is None for x in v):
                continue
            print(f"{sy:<16}{m:<14}", " ".join(f"{x:.4f}" for x in v), f"| {np.mean(v):.4f} {np.std(v, ddof=1):.4f} | {min(v):.4f} {max(v):.4f}")


show("dev"); show("holdout")
json.dump(dict(note="only the light detector's training seed varies; thresholds frozen on seed 0; heavy detector fixed; holdout is post hoc",
               seeds=SEEDS, dev=out["dev"], holdout=out["holdout"]), open(cc.RES / "seed_eval.json", "w"), indent=1)
