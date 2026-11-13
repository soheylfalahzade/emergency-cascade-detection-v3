"""COCO val2017 benchmark of routing/ensembling methods with three COCO-pretrained detectors (no training by us).
Classes: car, motorcycle, bus, truck (crowd annotations excluded; torchmetrics mAP, so not comparable to the COCO leaderboard).
Split: 2500 dev / 2500 test images (seed 42). Everything below is derived on dev and frozen before test is read:
  LO   : 1st percentile of light-model confidence on correctly detected objects (as in the emergency-vehicle protocol)
  TAU  : confidence maximising F1 of the heavy model (defines 'detected')
  hi   : fixed-sequence testing over the distinct dev values of the trigger statistic (exact, no grid resolution), most conservative first; one-sided exact binomial (Clopper-Pearson) bound at
         level DELTA on the image-level loss rate (an image where an object found by the heavy model is lost by the cascade)
  alpha: operating points = RHOS x (loss rate of the light model alone); the emergency-vehicle experiment used alphas of about
         0.33, 0.45 and 0.67 of its light-only loss rate
  meta : the learner selected in the emergency-vehicle protocol (results_v3/protocol_frozen.json), fitted on dev detections
  band0.7 transfer: the emergency protocol's band<0.7 with floor 0.25, applied without recalibration"""
import os, sys, json, time, pickle, hashlib, argparse
from pathlib import Path
import numpy as np, torch, torchvision
from scipy.stats import beta
from stackcore import *
from stack_eval import cap, wbf_fuse

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"; OUT = ROOT / "results_coco"
MODELS, LIGHT, HEAVY = ["yolov8n", "rtdetr_l", "fasterrcnn"], "yolov8n", "rtdetr_l"
MINC, DELTA, RHOS = 0.02, 0.1, (1 / 3, 1 / 2, 2 / 3)
FRZ, FLAG = OUT / "coco_protocol_frozen.json", OUT / "coco_holdout_done.flag"
REN = {"ap50_ambulance": "car", "ap50_police": "motorcycle", "ap50_firetruck": "bus", "ap50_car": "truck"}
SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
META = json.load(open(ROOT / "results_v3/protocol_frozen.json"))["meta"]


def load_rows():
    sp = json.load(open(OUT / "split.json"))
    pm = {m: pickle.load(open(OUT / "dets" / f"{m}_val2017.pkl", "rb")) for m in MODELS}
    ids = [r["image_id"] for r in pm[MODELS[0]]]
    for m in MODELS[1:]:
        assert [r["image_id"] for r in pm[m]] == ids, "image order mismatch"
    assert not set(sp["dev"]) & set(sp["test"])
    res = {}
    for part in ("dev", "test"):
        s = set(sp[part]); ix = [i for i, x in enumerate(ids) if x in s]
        res[part] = {m: [pm[m][i] for i in ix] for m in MODELS}
    return res, sp


def cf(r):
    k = r["conf"] >= MINC
    b, s, l = r["boxes"][k], r["conf"][k], r["cls"][k]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


def hit_all(gb, gc, b, s, l, thr):
    out = np.zeros(len(gc), bool); k = s >= thr
    if len(gc) and k.any():
        out = ((iou_mat(gb, b[k]) >= 0.5) & (gc[:, None] == l[k][None, :])).any(1)
    return out


def cp_ucb(k, n):
    return 1.0 if k >= n else float(beta.ppf(1 - DELTA, k + 1, n - k))


def f1_tau(Hr, GT):
    best = (-1.0, 0.5)
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        tp = ng = nd = ndt = 0
        for r, (gb, gc) in zip(Hr, GT):
            ng += len(gc); tp += int(hit_all(gb, gc, r["boxes"], r["conf"], r["cls"], t).sum())
            e = r["conf"] >= t; nd += int(e.sum())
            if e.any() and len(gb):
                ndt += int((((iou_mat(r["boxes"][e], gb) >= 0.5) & (r["cls"][e][:, None] == gc[None, :])).any(1)).sum())
        p, rc = ndt / max(nd, 1), tp / max(ng, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def light_tp_conf(Lr, GT):
    v = []
    for r, (gb, gc) in zip(Lr, GT):
        k = r["conf"] >= MINC
        if len(gc) == 0 or not k.any():
            continue
        I = iou_mat(gb, r["boxes"][k]); same = gc[:, None] == r["cls"][k][None, :]
        m = np.where((I >= 0.5) & same, r["conf"][k][None, :], -1.0).max(1)
        v += m[m > 0].tolist()
    return np.array(v)


def full(D, GT):
    r = evaluate_map(D, GT)
    o = {"map50": float(r["map50"]), "map": float(r["map"])}
    for k, c in REN.items():
        o["ap50_" + c] = float(r.get(k, float("nan")))
    return o


def map50(dets, gts, chunk=500):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    m = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5])
    for i in range(0, len(dets), chunk):
        P = [dict(boxes=torch.as_tensor(np.asarray(b, np.float32).reshape(-1, 4)), scores=torch.as_tensor(np.asarray(s, np.float32)),
                  labels=torch.as_tensor(np.asarray(l, np.int64))) for b, s, l in dets[i:i + chunk]]
        T = [dict(boxes=torch.as_tensor(np.asarray(gb, np.float32).reshape(-1, 4)), labels=torch.as_tensor(np.asarray(gc, np.int64)))
             for gb, gc in gts[i:i + chunk]]
        m.update(P, T)
    return float(m.compute()["map_50"])


def cmd_freeze():
    assert not FLAG.exists(), "test split already evaluated"
    if FRZ.exists() and not os.environ.get("REFREEZE"):
        sys.exit("already frozen (REFREEZE=1 only before the test split is evaluated)")
    R, sp = load_rows(); D = R["dev"]; L, H = D[LIGHT], D[HEAVY]; n = len(L)
    GT = [(r["gt_boxes"], r["gt_cls"]) for r in L]
    LD, HD = [cf(r) for r in L], [cf(r) for r in H]
    TAU, f1 = f1_tau(H, GT); tpc = light_tp_conf(L, GT); LO = max(MINC, float(np.percentile(tpc, 1)))
    hh = [hit_all(gb, gc, *HD[i], TAU) for i, (gb, gc) in enumerate(GT)]
    ll = [hit_all(gb, gc, *LD[i], TAU) for i, (gb, gc) in enumerate(GT)]
    pop = np.array([h.any() for h in hh]); lost = np.array([bool((hh[i] & ~ll[i]).any()) for i in range(n)])
    cmin = np.array([r["conf"][r["conf"] >= LO].min() if (r["conf"] >= LO).any() else np.inf for r in L])
    npop = int(pop.sum()); Rl = float((lost & pop).sum() / npop)
    print(f"dev images={n} objects={sum(len(g[1]) for g in GT)} population={npop} LO={LO:.3f} TAU={TAU} (F1 {f1:.3f}) light-only loss rate={Rl:.4f}")
    cand = np.unique(np.concatenate([cmin[np.isfinite(cmin) & (cmin <= 0.95)], [0.95]]))[::-1]; front = []
    for hi in cand:
        E = cmin < hi; k = int((lost & pop & ~E).sum())
        front.append(dict(hi=float(hi), n=npop, lost=k, risk=k / npop, ucb=cp_ucb(k, npop), esc=float(E.mean())))
    sel, alphas = {}, {f"{r:.3f}": r * Rl for r in RHOS}
    for key, al in alphas.items():
        h = None
        for f in front:
            if f["ucb"] <= al:
                h = f["hi"]
            else:
                break
        sel[key] = h
    print("candidates:", len(front), "| selection at least-conservative candidate (boundary):", {k: (v == front[-1]["hi"]) for k, v in sel.items()})
    dev = {"light_only": full(LD, GT), "heavy_only": full(HD, GT)}
    for key, h in sel.items():
        if h is not None:
            E = cmin < h; dev[f"cascade_r{key}"] = full([HD[i] if E[i] else LD[i] for i in range(n)], GT)
    P = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), lo=LO, tau=TAU, delta=DELTA, rhos=list(RHOS), light_loss_rate=Rl, alpha=alphas,
             hi=sel, frontier=front, dev=dev, sha256=SHA, split_seed=sp["seed"], meta=META)
    json.dump(P, open(FRZ, "w"), indent=1)
    for key, h in sel.items():
        r = next((x for x in front if x["hi"] == h), None)
        print(f"rho={key} alpha={alphas[key]:.4f}: hi={h}", "" if r is None else f"esc={r['esc']:.3f} risk={r['risk']:.4f} ucb={r['ucb']:.4f}")
    for k, v in dev.items():
        print(f"DEV {k:<18}", {a: round(b, 4) for a, b in v.items()})


def nms_union(pm, i):
    b = np.concatenate([pm[m][i]["boxes"] for m in MODELS]); s = np.concatenate([pm[m][i]["conf"] for m in MODELS])
    l = np.concatenate([pm[m][i]["cls"] for m in MODELS]); k = s >= MINC; b, s, l = b[k], s[k], l[k]
    if len(b):
        kp = torchvision.ops.batched_nms(torch.from_numpy(b.astype(np.float32)), torch.from_numpy(s.astype(np.float32)),
                                         torch.from_numpy(l.astype(np.int64)), 0.5).numpy()
        b, s, l = b[kp], s[kp], l[kp]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


def cmd_final():
    assert not FLAG.exists(), "test split already evaluated"
    P = json.load(open(FRZ)); assert P["sha256"] == SHA, "script changed since freezing; REFREEZE=1 python coco_eval.py freeze"
    R, _ = load_rows(); dev, test = R["dev"], R["test"]; LO = P["lo"]
    clf = make_meta(META).fit(*[build_split(dev, MODELS, False, 0.55, MINC)[k] for k in ("X", "y")])
    L = test[LIGHT]; n = len(L); GT = [(r["gt_boxes"], r["gt_cls"]) for r in L]
    LD, HD = [cf(r) for r in L], [cf(r) for r in test[HEAVY]]
    out = {"single_" + m: [cf(r) for r in test[m]] for m in MODELS}; esc = {}
    for key, hi in P["hi"].items():
        if hi is None:
            continue
        E = np.array([bool(((r["conf"] >= LO) & (r["conf"] < hi)).any()) for r in L]); esc[key] = float(E.mean())
        out[f"cascade_r{key}"] = [HD[i] if E[i] else LD[i] for i in range(n)]
        rr = np.random.RandomState(int(float(key) * 1000)).rand(n) < esc[key]
        out[f"random_r{key}"] = [HD[i] if rr[i] else LD[i] for i in range(n)]
    E7 = np.array([bool(((r["conf"] >= 0.25) & (r["conf"] < 0.7)).any()) for r in L]); esc["band0.7"] = float(E7.mean())
    out["cascade_band0.7_transfer"] = [HD[i] if E7[i] else LD[i] for i in range(n)]
    out["nms_union"] = [nms_union(test, i) for i in range(n)]
    bt = build_split(test, MODELS, False, 0.55, MINC)
    out["wbf"] = cap([wbf_fuse(test, MODELS, i, MINC) for i in range(n)])
    out["top1_ensemblenet_style"] = cap([fuse_top1(cl) for cl in bt["cl"]])
    out["softavg"] = [fuse_softavg(cl) for cl in bt["cl"]]
    out["meta_" + META] = per_image(bt, n, proba5(clf, bt["X"]))
    pt = {k: full(v, GT) for k, v in out.items()}; print("point estimates done", flush=True)
    BM = ["single_" + LIGHT, "single_" + HEAVY, "softavg", "wbf", "meta_" + META, "cascade_band0.7_transfer"] + \
         [k for k in out if k.startswith(("cascade_r", "random_r"))]
    BOOT = int(os.environ.get("BOOT", 100)); rng = np.random.RandomState(0); boot = {k: [] for k in BM}; t0 = time.time()
    for b in range(BOOT):
        idx = rng.randint(0, n, n); g = [GT[i] for i in idx]
        for k in BM:
            if k.startswith("random_r"):
                mk = rng.rand(n) < esc[k.split("_r")[1]]; D = [HD[i] if mk[j] else LD[i] for j, i in enumerate(idx)]
            else:
                D = [out[k][i] for i in idx]
            boot[k].append(map50(D, g))
        if b % 5 == 0:
            print(f"bootstrap {b}/{BOOT} ({time.time() - t0:.0f}s)", flush=True)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    B = {k: np.array(v) for k, v in boot.items()}; pairs = {}
    for key in esc:
        if key == "band0.7":
            continue
        c, r_ = f"cascade_r{key}", f"random_r{key}"
        for a_, b_ in ((c, r_), (c, "single_" + HEAVY), (c, "single_" + LIGHT)):
            d = B[a_] - B[b_]; pairs[f"{a_}-{b_}"] = dict(mean=float(d.mean()), ci95=ci(d))
        for nm in (c, r_):
            gap = (B[nm] - B["single_" + LIGHT]) / (B["single_" + HEAVY] - B["single_" + LIGHT])
            pairs[f"share of light->heavy gap closed by {nm}"] = dict(median=float(np.median(gap)), ci95=ci(gap))
    for a_, b_ in (("meta_" + META, "single_" + HEAVY), ("wbf", "single_" + HEAVY), ("softavg", "single_" + HEAVY), ("meta_" + META, "wbf"),
                   ("cascade_band0.7_transfer", "single_" + HEAVY)):
        d = B[a_] - B[b_]; pairs[f"{a_}-{b_}"] = dict(mean=float(d.mean()), ci95=ci(d))
    res = dict(protocol=P, n_test_images=n, bootstrap=BOOT, esc_rate=esc, point=pt, map50_ci95={k: ci(v) for k, v in B.items()}, paired=pairs)
    json.dump(res, open(OUT / "coco_final.json", "w"), indent=1); FLAG.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in pt.items():
        print(f"{k:<26} mAP50={v['map50']:.4f} {[round(x, 3) for x in ci(B[k])] if k in B else ''} mAP50-95={v['map']:.4f}",
              {c: round(v['ap50_' + c], 3) for c in REN.values()})
    for k, v in pairs.items():
        print(f"{k:<62}", {a: (round(b, 4) if isinstance(b, float) else [round(x, 3) for x in b]) for a, b in v.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); a = ap.parse_args()
    cmd_freeze() if a.cmd == "freeze" else cmd_final()
