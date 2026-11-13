"""Risk-controlled image-level cascade: parameters frozen on cross-fitted validation data, held-out split evaluated once.
Parameter provenance:
  LO    : 1st percentile of light-model confidence on correctly detected emergency objects (results_v3/crop_plan.json)
  TAU   : confidence maximising emergency-object F1 of the heavy model on validation data (same file); defines 'detected'
  hi    : fixed-sequence testing from the most conservative threshold downwards; one-sided exact binomial (Clopper-Pearson)
          bound at level DELTA on the group-level loss rate. Loss = a group in which an emergency object found by the heavy
          model is lost by the cascade. ALPHAS = tolerated loss rates, reported as a set of operating points.
  DELTA=0.1 (90% confidence); grid step 0.02 is resolution only.
Metrics: emergency-mAP50 (primary; validation car labels are incomplete), 4-class mAP50 (secondary)."""
import os, sys, json, time, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta
import crop_cascade as cc
from stackcore import *
from stack_eval import aligned, cap, oof_training, wbf_fuse

RES, ROOT = cc.RES, cc.ROOT
LIGHT, HEAVY, MODELS = cc.LIGHT, cc.HEAVY, cc.MODELS
PLAN = json.load(open(RES / "crop_plan.json")); LO, TAU = PLAN["lo"], PLAN["tau"]
DELTA, ALPHAS = 0.1, (0.015, 0.02, 0.03)
OUT = RES / "image_cascade_protocol_frozen.json"; FLAG = RES / "image_cascade_holdout_done.flag"
FRZ = RES / "protocol_frozen.json"
KEYS = ("map50", "emerg_map50", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car", "emerg_recall", "car_as_emerg")
SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def cp_ucb(k, n):
    return 1.0 if k >= n else float(beta.ppf(1 - DELTA, k + 1, n - k))


def cascade(L, LDf, HDf, hi):
    E = [bool(((r["conf"] >= LO) & (r["conf"] < hi)).any()) for r in L]
    return [HDf[i] if E[i] else LDf[i] for i in range(len(E))], float(np.mean(E))


def lost_flags(D, GT, hh):
    return [bool((hh[i] & ~cc.hit(GT[i][0], GT[i][1], *D[i], TAU)).any()) for i in range(len(GT))]


def dev_data():
    man = pd.read_csv(ROOT / "datasets/folds_v3_manifest.csv"); gmap = dict(zip(man["id"], man["group"]))
    d = dict(L=[], LD=[], HD=[], GT=[], hh=[], gid=[])
    for f in range(5):
        L = cc.load(LIGHT, f, "val"); H = cc.load(HEAVY, f, "val")
        assert [r["image"] for r in L] == [r["image"] for r in H], f"order mismatch fold{f}"
        HDf = [cc.cf(r) for r in H]; GT = [(r["gt_boxes"], r["gt_cls"]) for r in L]
        d["L"].append(L); d["LD"].append([cc.cf(r) for r in L]); d["HD"].append(HDf); d["GT"].append(GT)
        d["hh"].append([cc.hit(g[0], g[1], *HDf[i], TAU) for i, g in enumerate(GT)])
        d["gid"] += [str(gmap.get(Path(r["image"]).stem, f"solo{f}_{i}")) for i, r in enumerate(L)]
    d["gid"] = np.array(d["gid"])
    d["pop"] = np.array([h.any() for f in range(5) for h in d["hh"][f]])
    return d


def pooled_loss(d, Dl):
    lost = np.concatenate([lost_flags(Dl[f], d["GT"][f], d["hh"][f]) for f in range(5)])
    g = pd.Series(lost[d["pop"]]).groupby(d["gid"][d["pop"]]).max()
    return len(g), int(g.sum())


def fm(d, Dl):
    per = [cc.metrics(Dl[f], d["GT"][f], TAU) for f in range(5)]
    return {k: float(np.nanmean([p.get(k, np.nan) for p in per])) for k in KEYS}


def cmd_freeze():
    assert not FLAG.exists(), "held-out split already evaluated for this protocol"
    if OUT.exists() and not os.environ.get("REFREEZE"):
        sys.exit("already frozen (REFREEZE=1 only before the held-out split is evaluated)")
    d = dev_data()
    grid = np.round(np.arange(np.ceil(LO * 50) / 50 + 0.02, 0.9501, 0.02), 2)
    front = []
    for hi in grid[::-1]:
        hi = float(hi)
        res = [cascade(d["L"][f], d["LD"][f], d["HD"][f], hi) for f in range(5)]
        n, k = pooled_loss(d, [r[0] for r in res])
        front.append(dict(hi=hi, n_groups=n, lost_groups=k, risk=k / n, ucb=cp_ucb(k, n), esc=float(np.mean([r[1] for r in res]))))
    sel = {}
    for al in ALPHAS:
        h = None
        for r in front:
            if r["ucb"] <= al:
                h = r["hi"]
            else:
                break
        sel[str(al)] = h
    n0, k0 = pooled_loss(d, d["LD"])
    dev = {"light_only": fm(d, d["LD"]), "heavy_only": fm(d, d["HD"])}
    for al, h in sel.items():
        if h is not None:
            dev[f"cascade_a{al}"] = fm(d, [cascade(d["L"][f], d["LD"][f], d["HD"][f], h)[0] for f in range(5)])
    P = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), lo=LO, tau=TAU, delta=DELTA, alphas=list(ALPHAS), hi=sel, frontier=front,
             light_only_risk=dict(n_groups=n0, lost_groups=k0, risk=k0 / n0, ucb=cp_ucb(k0, n0)), dev=dev, sha256=SHA, plan=PLAN)
    json.dump(P, open(OUT, "w"), indent=1)
    print(f"lo={LO:.3f} tau={TAU} delta={DELTA} groups={n0} | light_only risk={k0 / n0:.4f} ucb={cp_ucb(k0, n0):.4f}")
    for al, h in sel.items():
        r = next((x for x in front if x["hi"] == h), None)
        print(f"alpha={al}: hi={h}", "" if r is None else f"esc={r['esc']:.3f} risk={r['risk']:.4f} ucb={r['ucb']:.4f}")
    for k, v in dev.items():
        print(f"DEV {k:<18}", {a: round(b, 4) for a, b in v.items()})


def cmd_final(no_stacking):
    assert not FLAG.exists(), "held-out split already evaluated for this protocol"
    P = json.load(open(OUT))
    assert P["sha256"] == SHA, "script changed since freezing; run: REFREEZE=1 python image_cascade_final.py freeze"
    PF = json.load(open(FRZ)); HIF = float(PF["policy"].split("<")[1])
    man = pd.read_csv(ROOT / "datasets/folds_v3_manifest.csv"); gmap = dict(zip(man["id"], man["group"]))
    clf = None
    if not no_stacking:
        _, _, built = oof_training(MODELS, False, 0.55, cc.MINC)
        X = np.concatenate([built[g]["X"] for g in range(5)]); y = np.concatenate([built[g]["y"] for g in range(5)])
        clf = make_meta(PF["meta"]).fit(X, y)
    store, per, esc, rk = {}, {}, {}, {}
    for f in range(5):
        pm = aligned(MODELS, f, "holdout"); Lr, Hr = pm[LIGHT], pm[HEAVY]; n = len(Lr)
        LDf = [cc.cf(r) for r in Lr]; HDf = [cc.cf(r) for r in Hr]
        gts = [(r["gt_boxes"], r["gt_cls"]) for r in Lr]
        hhf = [cc.hit(g[0], g[1], *HDf[i], TAU) for i, g in enumerate(gts)]
        out = {"single_" + LIGHT: LDf, "single_" + HEAVY: HDf}
        Ef = [bool((r["conf"][r["conf"] >= 0.25] < HIF).any()) for r in Lr]
        out["cascade_frozen_protocol"] = [HDf[i] if Ef[i] else LDf[i] for i in range(n)]
        esc[f] = {"cascade_frozen_protocol": float(np.mean(Ef))}
        for al, hi in P["hi"].items():
            if hi is None:
                continue
            D, e = cascade(Lr, LDf, HDf, hi); out[f"cascade_a{al}"] = D; esc[f][f"cascade_a{al}"] = e
        out["wbf"] = cap([wbf_fuse(pm, MODELS, i, cc.MINC) for i in range(n)])
        bt = build_split(pm, MODELS, False, 0.55, cc.MINC)
        out["softavg"] = [fuse_softavg(cl) for cl in bt["cl"]]
        if clf is not None:
            out["meta_" + PF["meta"]] = per_image(bt, n, proba5(clf, bt["X"]))
        per[f] = {k: cc.metrics(v, gts, TAU) for k, v in out.items()}
        names = [r["image"] for r in Lr]
        gidf = np.array([str(gmap.get(Path(nm).stem, f"solo{i}")) for i, nm in enumerate(names)])
        pop = np.array([h.any() for h in hhf]); rk[f] = {}
        for k, v in out.items():
            g = pd.Series(np.array(lost_flags(v, gts, hhf))[pop]).groupby(gidf[pop]).max()
            rk[f][k] = (len(g), int(g.sum()))
        store[f] = dict(out=out, gts=gts, names=names)
        print(f"holdout fold-model {f} done", flush=True)
    names = store[0]["names"]
    assert all(store[f]["names"] == names for f in range(5)), "holdout image order differs between folds"
    gid = np.array([str(gmap.get(Path(nm).stem, f"solo{i}")) for i, nm in enumerate(names)])
    members = {g: np.where(gid == g)[0] for g in np.unique(gid)}; ug = list(members)
    BM = ["single_" + LIGHT, "single_" + HEAVY, "cascade_frozen_protocol"] + [k for k in store[0]["out"] if k.startswith("cascade_a")]
    BOOT = int(os.environ.get("BOOT", 100)); rng = np.random.RandomState(0)
    boot = {k: dict(e=[], m=[]) for k in BM}
    for b in range(BOOT):
        idx = np.concatenate([members[ug[j]] for j in rng.randint(0, len(ug), len(ug))])
        v = {k: ([], []) for k in BM}
        for f in range(5):
            g = [store[f]["gts"][i] for i in idx]
            for k in BM:
                r = evaluate_map([store[f]["out"][k][i] for i in idx], g); v[k][0].append(cc.emerg_of(r)); v[k][1].append(r["map50"])
        for k in BM:
            boot[k]["e"].append(float(np.nanmean(v[k][0]))); boot[k]["m"].append(float(np.mean(v[k][1])))
        if b % 10 == 0:
            print(f"bootstrap {b}/{BOOT}", flush=True)
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    summary = {}
    for k in store[0]["out"]:
        summary[k] = {q: [float(np.nanmean([per[f][k].get(q, np.nan) for f in range(5)])),
                          float(np.nanstd([per[f][k].get(q, np.nan) for f in range(5)], ddof=1))] for q in KEYS}
        summary[k]["loss_rate_holdout"] = float(np.mean([rk[f][k][1] / max(rk[f][k][0], 1) for f in range(5)]))
        summary[k]["esc_rate"] = float(np.mean([esc[f][k] for f in range(5)])) if k in esc[0] else None
        if k in boot:
            summary[k]["emerg_map50_ci95"] = ci(boot[k]["e"]); summary[k]["map50_ci95"] = ci(boot[k]["m"])
    casc = [k for k in BM if k.startswith("cascade_a")]
    pairs = [(c, "single_" + HEAVY) for c in casc] + [(c, "single_" + LIGHT) for c in casc]
    if "cascade_a0.02" in boot:
        pairs.append(("cascade_a0.02", "cascade_frozen_protocol"))
    diffs = {}
    for a_, b_ in pairs:
        de, dm = np.array(boot[a_]["e"]) - np.array(boot[b_]["e"]), np.array(boot[a_]["m"]) - np.array(boot[b_]["m"])
        diffs[f"{a_}-{b_}"] = dict(emerg_mean=float(np.nanmean(de)), emerg_ci95=ci(de), map50_mean=float(dm.mean()), map50_ci95=ci(dm))
    res = dict(note="post-hoc: the held-out split was already used once (frozen band<0.7 cascade); emergency-mAP50 was defined afterwards "
                    "(motivated by documented incomplete car labels). Parameters of this protocol were frozen on validation data before this run.",
               protocol=P, n_images=len(names), bootstrap=BOOT, summary=summary, paired=diffs)
    json.dump(res, open(RES / "image_cascade_holdout_final.json", "w"), indent=1)
    FLAG.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in summary.items():
        print(f"{k:<26} emerg_mAP50={v['emerg_map50'][0]:.4f} {v.get('emerg_map50_ci95', '')} mAP50={v['map50'][0]:.4f} "
              f"firetruck={v['ap50_firetruck'][0]:.4f} recall={v['emerg_recall'][0]:.4f} loss={v['loss_rate_holdout']:.4f} esc={v['esc_rate']}")
    print(json.dumps(diffs, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--no-stacking", action="store_true")
    a = ap.parse_args()
    cmd_freeze() if a.cmd == "freeze" else cmd_final(a.no_stacking)
