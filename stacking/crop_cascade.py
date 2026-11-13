"""Crop-level cascade for still images: light detector -> heavy detector on uncertain regions only.
Parameter provenance (derived on cross-fitted validation data, frozen before the held-out split is used):
  lo     : 1st percentile of light-model true-positive confidences on emergency objects (keeps 99% of them)
  tau    : confidence maximising emergency-object F1 of the heavy model (defines a 'detected' object)
  hi     : Learn-then-Test fixed-sequence selection (Hoeffding bound over image groups), tolerance alpha, level delta
  config : (context margin, crop input size) with the lowest certified cost in full-image-equivalent heavy passes
  MIN_SIDE=64 : two strides of the heavy backbone. REG_IOU=0.6 : class-agnostic grouping of light hypotheses.
  alpha, delta : service-level requirement (sensitivity table printed for alpha in {0.02, 0.05, 0.10}).
Primary metric: emergency-mAP50 (mean AP50 of ambulance/police/firetruck); 4-class mAP50 reported as secondary because
car labels of the validation source are incomplete."""
import os, sys, json, time, pickle, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision, cv2
from stackcore import *
from stack_eval import aligned, cap, oof_training, wbf_fuse

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
DS, RES = ROOT / "datasets", ROOT / "results_v3"
CC = RES / "cropcache"; CC.mkdir(exist_ok=True)
LIGHT, HEAVY, MODELS = "yolov8n", "rtdetr_l", ["yolov8n", "rtdetr_l", "fasterrcnn"]
MINC, REG_IOU, HI_MAX, MIN_SIDE, EM = 0.02, 0.6, 0.95, 64, (0, 1, 2)
DEV = os.environ.get("DEVICE", "0")
CONFIGS = [(0.25, 320), (0.25, 640), (0.5, 320), (0.5, 640)]
EXTRA = ("ap50_ambulance", "ap50_police", "ap50_firetruck")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def img_dir(f, split):
    return DS / "holdout_v3/images" if split == "holdout" else DS / f"folds_v3/fold_{f}/val/images"


def load(m, f, split):
    return pickle.load(open(RES / "dets" / f"{m}_fold{f}_{split}.pkl", "rb"))


def cf(r):
    k = r["conf"] >= MINC
    b, s, l = r["boxes"][k], r["conf"][k], r["cls"][k]
    o = np.argsort(-s)[:100]
    return b[o], s[o], l[o]


def regions(r):
    k = r["conf"] >= MINC
    b, c, l = r["boxes"][k], r["conf"][k], r["cls"][k]
    o = np.argsort(-c, kind="stable"); b, c, l = b[o], c[o], l[o]
    rid, reps = np.zeros(len(b), int), []
    for j in range(len(b)):
        if reps:
            I = iou_mat(b[j:j + 1], b[reps])[0]
            m = int(I.argmax())
            if I[m] >= REG_IOU:
                rid[j] = m; continue
        rid[j] = len(reps); reps.append(j)
    return b, c, l, rid, np.array(reps, int)


def hit(gb, gc, b, s, l, thr, classes=EM):
    out = np.zeros(len(gc), bool)
    m = np.isin(gc, classes); k = s >= thr
    if m.any() and k.any():
        same = gc[m][:, None] == l[k][None, :]
        out[np.where(m)[0]] = ((iou_mat(gb[m], b[k]) >= 0.5) & same).any(1)
    return out


def make_crop(im, box, margin):
    H, W = im.shape[:2]; x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    px, py = max(margin * w, (MIN_SIDE - w) / 2, 0.0), max(margin * h, (MIN_SIDE - h) / 2, 0.0)
    cx1, cy1 = int(max(0, np.floor(x1 - px))), int(max(0, np.floor(y1 - py)))
    cx2, cy2 = int(min(W, np.ceil(x2 + px))), int(min(H, np.ceil(y2 + py)))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    return im[cy1:cy2, cx1:cx2], cx1, cy1


def emerg_of(r):
    v = [r[k] for k in EXTRA if r.get(k, -1) >= 0]
    return float(np.mean(v)) if v else float("nan")


def rates(dets, gts, tau):
    er = ec = ct = cc = 0
    for (b, s, l), (gb, gc) in zip(dets, gts):
        er += int(hit(gb, gc, b, s, l, tau).sum()); ec += int(np.isin(gc, EM).sum())
        mc = gc == 3; cc += int(mc.sum())
        e = np.isin(l, EM) & (s >= tau)
        if mc.any() and e.any():
            ct += int((iou_mat(gb[mc], b[e]) >= 0.5).any(1).sum())
    return dict(emerg_recall=er / max(ec, 1), car_as_emerg=ct / max(cc, 1))


def metrics(dets, gts, tau):
    r = evaluate_map(dets, gts)
    r["emerg_map50"] = emerg_of(r)
    r.update(rates(dets, gts, tau))
    return r


def f1_tau(Hs):
    best = (-1.0, 0.5)
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        tp = ng = nd = ndt = 0
        for r in Hs:
            gb, gc, b, s, l = r["gt_boxes"], r["gt_cls"], r["boxes"], r["conf"], r["cls"]
            ng += int(np.isin(gc, EM).sum()); tp += int(hit(gb, gc, b, s, l, t).sum())
            e = np.isin(l, EM) & (s >= t); nd += int(e.sum())
            if e.any() and len(gb):
                ndt += int((((iou_mat(b[e], gb) >= 0.5) & (l[e][:, None] == gc[None, :])).any(1)).sum())
        p, rc = ndt / max(nd, 1), tp / max(ng, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def compose(rg, row, lo, hi):
    b, c, l, rid, reps = rg
    if len(b) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)), 0
    rc = c[reps][rid]
    has = np.array([q in row["heavy"] for q in range(len(reps))])
    keep = (rc >= hi) | ((rc >= lo) & (rc < hi) & ~has[rid])
    B, S, Lb, n = [b[keep]], [c[keep]], [l[keep]], 0
    for q, (hb, hs, hl) in row["heavy"].items():
        if not (lo <= c[reps[q]] < hi):
            continue
        n += 1
        x1, y1, x2, y2 = b[reps[q]]
        cx, cy = (hb[:, 0] + hb[:, 2]) / 2, (hb[:, 1] + hb[:, 3]) / 2
        k = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        B.append(hb[k]); S.append(hs[k]); Lb.append(hl[k])
    B, S, Lb = np.concatenate(B), np.concatenate(S), np.concatenate(Lb)
    if len(B):
        kk = torchvision.ops.batched_nms(torch.from_numpy(B.astype(np.float32)), torch.from_numpy(S.astype(np.float32)),
                                         torch.from_numpy(Lb.astype(np.int64)), 0.5).numpy()[:100]
        B, S, Lb = B[kk], S[kk], Lb[kk]
    return (B.astype(np.float32).reshape(-1, 4), S.astype(np.float32), Lb.astype(np.int64)), n


def crop_dets(RGf, rows, hi, lo):
    out, nc = [], 0
    for rg, row in zip(RGf, rows):
        d, n = compose(rg, row, lo, hi); out.append(d); nc += n
    return out, nc


def cmd_plan():
    Ls = [load(LIGHT, f, "val") for f in range(5)]
    Hs = [r for f in range(5) for r in load(HEAVY, f, "val")]
    tp, tot, cov = [], 0, 0
    for L in Ls:
        for r in L:
            gb, gc = r["gt_boxes"], r["gt_cls"]; m = np.where(np.isin(gc, EM))[0]
            if not len(m) or not len(r["boxes"]):
                tot += len(m); continue
            I = iou_mat(gb[m], r["boxes"]); ok = (I >= 0.5) & (r["cls"][None, :] == gc[m][:, None])
            for q in range(len(m)):
                tot += 1; cov += int(((I[q] >= 0.5) & (r["conf"] >= MINC)).any())
                if ok[q].any():
                    tp.append(float(r["conf"][ok[q]].max()))
    lo = max(MINC, float(np.percentile(tp, 1)))
    tau, f1 = f1_tau(Hs)
    P = dict(lo=lo, tau=tau, tau_f1=f1, tp_recall_light=len(tp) / tot, proposal_recall_light=cov / tot, n_emergency_objects=tot)
    json.dump(P, open(RES / "crop_plan.json", "w"), indent=1); print(json.dumps(P, indent=1))


def cmd_cache(split, margin, size):
    lo = json.load(open(RES / "crop_plan.json"))["lo"]
    from ultralytics import RTDETR
    for f in range(5):
        out = CC / f"{split}_m{margin}_s{size}_fold{f}.pkl"
        if out.exists():
            continue
        net = RTDETR(str(RES / f"{HEAVY}_fold{f}/weights/best.pt"))
        L = load(LIGHT, f, split); d = img_dir(f, split)
        rows = [dict(image=r["image"], heavy={}) for r in L]; pend = []

        def flush():
            for s0 in range(0, len(pend), 16):
                ch = pend[s0:s0 + 16]
                rs = net.predict([p[2] for p in ch], conf=MINC, imgsz=size, device=DEV, verbose=False)
                for (i, q, _, ox, oy), r_ in zip(ch, rs):
                    bb = r_.boxes.xyxy.cpu().numpy().astype(np.float32) + np.array([ox, oy, ox, oy], np.float32)
                    rows[i]["heavy"][q] = (bb, r_.boxes.conf.cpu().numpy().astype(np.float32), r_.boxes.cls.cpu().numpy().astype(np.int64))
            pend.clear()

        for i, r in enumerate(L):
            b, c, l, rid, reps = regions(r)
            im = cv2.imread(str(d / r["image"])); assert im is not None, r["image"]
            for q, rep in enumerate(reps):
                if lo <= c[rep] < HI_MAX:
                    mc = make_crop(im, b[rep], margin)
                    if mc:
                        pend.append((i, q, mc[0], mc[1], mc[2]))
            if len(pend) >= 32:
                flush()
            if i % 200 == 0:
                print(f"{split} m{margin} s{size} fold{f}: {i}/{len(L)}", flush=True)
        flush()
        pickle.dump(rows, open(out, "wb")); print("saved", out)


def cmd_sweep(alpha, delta):
    FL = RES / "crop_holdout_done.flag"
    assert not FL.exists(), "held-out split already used; protocol cannot change"
    plan = json.load(open(RES / "crop_plan.json")); lo, tau = plan["lo"], plan["tau"]
    PF = json.load(open(RES / "protocol_frozen.json")); HIF = float(PF["policy"].split("<")[1])
    man = pd.read_csv(ROOT / "datasets/folds_v3_manifest.csv"); gmap = dict(zip(man["id"], man["group"]))
    Ls = [load(LIGHT, f, "val") for f in range(5)]; Hs = [load(HEAVY, f, "val") for f in range(5)]
    RG = [[regions(r) for r in L] for L in Ls]
    GT = [[(r["gt_boxes"], r["gt_cls"]) for r in L] for L in Ls]
    LD = [[cf(r) for r in L] for L in Ls]; HD = [[cf(r) for r in H] for H in Hs]
    gid = np.array([str(gmap.get(Path(r["image"]).stem, f"solo{f}_{i}")) for f, L in enumerate(Ls) for i, r in enumerate(L)])
    hh = [[hit(g[0], g[1], *HD[f][i], tau) for i, g in enumerate(GT[f])] for f in range(5)]
    pop = np.array([h.any() for f in range(5) for h in hh[f]])
    N = int(len(pop))
    print(f"images={N} population={int(pop.sum())} groups={len(set(gid[pop]))} lo={lo:.3f} tau={tau}")

    def risk(D):
        lost = np.array([bool((hh[f][i] & ~hit(*GT[f][i], *D[f][i], tau)).any()) for f in range(5) for i in range(len(GT[f]))])
        g = pd.Series(lost[pop].astype(float)).groupby(gid[pop]).mean().values
        R = float(g.mean())
        return R, R + float(np.sqrt(np.log(1 / delta) / (2 * len(g))))

    def esc_of(hi):
        return [[bool(((r["conf"] >= lo) & (r["conf"] < hi)).any()) for r in L] for L in Ls]

    def img_dets(hi):
        E = esc_of(hi)
        return [[HD[f][i] if E[f][i] else LD[f][i] for i in range(len(E[f]))] for f in range(5)], float(np.mean([e for x in E for e in x]))

    grid = np.round(np.arange(np.ceil(lo * 100) / 100 + 0.01, HI_MAX + 1e-9, 0.01), 2)
    systems = {"image": None}
    for m_, s_ in CONFIGS:
        systems[(m_, s_)] = [pickle.load(open(CC / f"val_m{m_}_s{s_}_fold{f}.pkl", "rb")) for f in range(5)]
    curves, sel, rows = {}, {}, []
    for key, cache in systems.items():
        cur = {}
        for hi in grid:
            if cache is None:
                D, cost = img_dets(hi)
            else:
                res = [crop_dets(RG[f], cache[f], hi, lo) for f in range(5)]
                D = [x[0] for x in res]; cost = sum(x[1] for x in res) / N * (key[1] / 640) ** 2
            R, U = risk(D); cur[hi] = (R, U, cost)
            rows.append(dict(system=str(key), hi=float(hi), risk=R, ucb=U, cost=cost))
        curves[key] = cur
        for al in sorted({alpha, 0.02, 0.05, 0.10}):
            h = None
            for hi in sorted(grid, reverse=True):
                if cur[hi][1] <= al:
                    h = float(hi)
                else:
                    break
            sel[(key, al)] = h
            print(f"system={key} alpha={al}: " + ("none certified" if h is None else f"hi={h} risk={cur[h][0]:.4f} ucb={cur[h][1]:.4f} cost={cur[h][2]:.3f}"), flush=True)
    pd.DataFrame(rows).to_csv(RES / "crop_sweep_dev.csv", index=False)
    cands = [(curves[k][h][2], k, h) for k in CONFIGS if (h := sel[(k, alpha)]) is not None]
    if not cands:
        sys.exit("no crop configuration certified at this alpha")
    cost_c, (m_, s_), hi_c = min(cands)
    hi_i = sel[("image", alpha)]
    cost_i = curves["image"][hi_i][2] if hi_i is not None else None

    def fold_metrics(Df):
        per = [metrics(Df[f], GT[f], tau) for f in range(5)]
        return {k: [float(np.nanmean([p[k] for p in per])), float(np.nanstd([p[k] for p in per], ddof=1))]
                for k in ("map50", "emerg_map50", "ap50_firetruck", "emerg_recall", "car_as_emerg")}
    dev = dict(light_only=fold_metrics(LD), heavy_only=fold_metrics(HD))
    dev["cascade_frozen_protocol"] = fold_metrics([[HD[f][i] if (r["conf"][r["conf"] >= 0.25] < HIF).any() else LD[f][i]
                                                   for i, r in enumerate(Ls[f])] for f in range(5)])
    if hi_i is not None:
        dev["image_cascade_certified"] = fold_metrics(img_dets(hi_i)[0])
    dev["crop_cascade"] = fold_metrics([crop_dets(RG[f], systems[(m_, s_)][f], hi_c, lo)[0] for f in range(5)])
    for k, v in dev.items():
        print(f"DEV {k:<26}", {a: round(b[0], 4) for a, b in v.items()})
    P = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), lo=lo, tau=tau, alpha=alpha, delta=delta, margin=m_, size=s_,
             hi_crop=hi_c, cost_crop_full_image_equiv=cost_c, hi_image=hi_i, cost_image=cost_i, dev=dev,
             sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    json.dump(P, open(RES / "crop_protocol_frozen.json", "w"), indent=1)
    print("FROZEN", {k: v for k, v in P.items() if k not in ("dev", "sha256")})


def cmd_final(no_stacking):
    FL = RES / "crop_holdout_done.flag"
    assert not FL.exists(), "held-out split already used for this protocol"
    P = json.load(open(RES / "crop_protocol_frozen.json")); PF = json.load(open(RES / "protocol_frozen.json"))
    lo, tau, m, s = P["lo"], P["tau"], P["margin"], P["size"]; HIF = float(PF["policy"].split("<")[1])
    man = pd.read_csv(ROOT / "datasets/folds_v3_manifest.csv"); gmap = dict(zip(man["id"], man["group"]))
    clf = None
    if not no_stacking:
        _, _, built = oof_training(MODELS, False, 0.55, MINC)
        X = np.concatenate([built[g]["X"] for g in range(5)]); y = np.concatenate([built[g]["y"] for g in range(5)])
        clf = make_meta(PF["meta"]).fit(X, y)
    store, per, ncr, esc_i = {}, {}, [], []
    for f in range(5):
        pm = aligned(MODELS, f, "holdout"); Lr, Hr = pm[LIGHT], pm[HEAVY]; n = len(Lr)
        rows = pickle.load(open(CC / f"holdout_m{m}_s{s}_fold{f}.pkl", "rb")); RGf = [regions(r) for r in Lr]
        out = {f"single_{k}": [cf(r) for r in pm[k]] for k in MODELS}
        cd, nc = crop_dets(RGf, rows, P["hi_crop"], lo); out["crop_cascade"] = cd; ncr.append(nc / n)
        if P["hi_image"] is not None:
            E = [bool(((r["conf"] >= lo) & (r["conf"] < P["hi_image"])).any()) for r in Lr]; esc_i.append(float(np.mean(E)))
            out["image_cascade_certified"] = [out[f"single_{HEAVY}"][i] if E[i] else out[f"single_{LIGHT}"][i] for i in range(n)]
        out["cascade_frozen_protocol"] = [out[f"single_{HEAVY}"][i] if (Lr[i]["conf"][Lr[i]["conf"] >= 0.25] < HIF).any()
                                          else out[f"single_{LIGHT}"][i] for i in range(n)]
        out["wbf"] = cap([wbf_fuse(pm, MODELS, i, MINC) for i in range(n)])
        bt = build_split(pm, MODELS, False, 0.55, MINC)
        out["softavg"] = [fuse_softavg(cl) for cl in bt["cl"]]
        if clf is not None:
            out["meta_" + PF["meta"]] = per_image(bt, n, proba5(clf, bt["X"]))
        gts = [(r["gt_boxes"], r["gt_cls"]) for r in Lr]
        store[f] = dict(out=out, gts=gts, names=[r["image"] for r in Lr])
        per[f] = {k: metrics(v, gts, tau) for k, v in out.items()}
        print(f"holdout point estimates fold{f} done", flush=True)
    names = store[0]["names"]
    gid = np.array([str(gmap.get(Path(nm).stem, f"solo{i}")) for i, nm in enumerate(names)])
    members = {g: np.where(gid == g)[0] for g in np.unique(gid)}; ug = list(members)
    BM = [k for k in ("single_" + LIGHT, "single_" + HEAVY, "cascade_frozen_protocol", "image_cascade_certified", "crop_cascade", "wbf", "softavg",
                      "meta_" + PF["meta"]) if k in store[0]["out"]]
    BOOT = int(os.environ.get("BOOT", 100)); rng = np.random.RandomState(0)
    boot = {k: dict(e=[], m=[]) for k in BM}
    for b in range(BOOT):
        idx = np.concatenate([members[ug[j]] for j in rng.randint(0, len(ug), len(ug))])
        v = {k: ([], []) for k in BM}
        for f in range(5):
            g = [store[f]["gts"][i] for i in idx]
            for k in BM:
                r = evaluate_map([store[f]["out"][k][i] for i in idx], g); v[k][0].append(emerg_of(r)); v[k][1].append(r["map50"])
        for k in BM:
            boot[k]["e"].append(float(np.nanmean(v[k][0]))); boot[k]["m"].append(float(np.mean(v[k][1])))
        if b % 10 == 0:
            print(f"bootstrap {b}/{BOOT}", flush=True)
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    keys = ("map50", "emerg_map50", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car", "emerg_recall", "car_as_emerg")
    summary = {}
    for k in store[0]["out"]:
        summary[k] = {q: [float(np.nanmean([per[f][k][q] for f in range(5)])), float(np.nanstd([per[f][k][q] for f in range(5)], ddof=1))] for q in keys}
        if k in boot:
            summary[k]["emerg_map50_ci95"] = ci(boot[k]["e"]); summary[k]["map50_ci95"] = ci(boot[k]["m"])
    pairs = [("crop_cascade", "single_" + HEAVY), ("crop_cascade", "single_" + LIGHT), ("crop_cascade", "image_cascade_certified"),
             ("crop_cascade", "cascade_frozen_protocol")]
    diffs = {}
    for a_, b_ in pairs:
        if a_ in boot and b_ in boot:
            de, dm = np.array(boot[a_]["e"]) - np.array(boot[b_]["e"]), np.array(boot[a_]["m"]) - np.array(boot[b_]["m"])
            diffs[f"{a_}-{b_}"] = dict(emerg_mean=float(np.nanmean(de)), emerg_ci95=ci(de), map50_mean=float(dm.mean()), map50_ci95=ci(dm))
    res = dict(note="post-hoc: the held-out split was already used once for the frozen image-level protocol; "
                    "emergency-mAP50 was defined after that evaluation (motivated by documented incomplete car labels). "
                    "The crop-cascade parameters were frozen on validation data before this run.",
               protocol=P, crop_regions_per_image=float(np.mean(ncr)), image_cascade_esc_rate=float(np.mean(esc_i)) if esc_i else None,
               bootstrap=BOOT, summary=summary, paired=diffs)
    json.dump(res, open(RES / "crop_holdout_final.json", "w"), indent=1); FL.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in summary.items():
        print(f"{k:<26} emerg_mAP50={v['emerg_map50'][0]:.4f} {v.get('emerg_map50_ci95', '')} mAP50={v['map50'][0]:.4f} "
              f"firetruck={v['ap50_firetruck'][0]:.4f} car_as_emerg={v['car_as_emerg'][0]:.4f}")
    print(json.dumps(diffs, indent=1))


def cmd_bench(n=300):
    P = json.load(open(RES / "crop_protocol_frozen.json")); lo, hi, m, s = P["lo"], P["hi_crop"], P["margin"], P["size"]
    from ultralytics import YOLO, RTDETR
    allp = sorted(p for p in img_dir(0, "val").iterdir() if p.suffix.lower() in EXT)
    paths = [allp[j] for j in np.random.RandomState(0).permutation(len(allp))[:n]]
    ims = [cv2.imread(str(p)) for p in paths]
    L = YOLO(str(RES / f"{LIGHT}_fold0/weights/best.pt")); Hn = RTDETR(str(RES / f"{HEAVY}_fold0/weights/best.pt"))
    kw = dict(conf=MINC, device=DEV, verbose=False); ncr = []

    def light(im):
        r = L.predict(im, imgsz=640, **kw)[0]
        return dict(boxes=r.boxes.xyxy.cpu().numpy(), conf=r.boxes.conf.cpu().numpy(), cls=r.boxes.cls.cpu().numpy().astype(int))

    def f_light(im):
        light(im)

    def f_heavy(im):
        Hn.predict(im, imgsz=640, **kw)

    def f_crop(im):
        r = light(im); b, c, l, rid, reps = regions(r); cr = []
        for rep in reps:
            if lo <= c[rep] < hi:
                mc = make_crop(im, b[rep], m)
                if mc:
                    cr.append(mc[0])
        ncr.append(len(cr))
        if cr:
            Hn.predict(cr, imgsz=s, **kw)

    def f_img(im):
        r = light(im)
        if P["hi_image"] is not None and ((r["conf"] >= lo) & (r["conf"] < P["hi_image"])).any():
            Hn.predict(im, imgsz=640, **kw)

    out = {}
    for name, fn in (("light", f_light), ("heavy", f_heavy), ("image_cascade", f_img), ("crop_cascade", f_crop)):
        ts = []
        for im in ims:
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(im); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        out[name] = dict(ms_median=float(np.median(ts[5:]) * 1000), ms_mean=float(np.mean(ts[5:]) * 1000), ms_p95=float(np.percentile(ts[5:], 95) * 1000))
        print(name, out[name], flush=True)
    out["crop_regions_per_image"] = float(np.mean(ncr[5:])); out["n_images"] = len(ims)
    out["note"] = "end-to-end per image, batch 1, GPU synchronised, includes pre/post-processing, excludes disk read"
    json.dump(out, open(RES / "crop_bench.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--split", default="val")
    ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--no-stacking", action="store_true"); a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan()
    elif a.cmd == "cache":
        if a.split == "val":
            for m_, s_ in CONFIGS:
                cmd_cache("val", m_, s_)
        else:
            Pz = json.load(open(RES / "crop_protocol_frozen.json")); cmd_cache(a.split, Pz["margin"], Pz["size"])
    elif a.cmd == "sweep":
        cmd_sweep(a.alpha, a.delta)
    elif a.cmd == "final":
        cmd_final(a.no_stacking)
    elif a.cmd == "bench":
        cmd_bench()
