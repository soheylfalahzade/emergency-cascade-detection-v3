import json, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision
from scipy.stats import beta
import crop_cascade as cc

P = json.load(open(cc.RES / "crop_protocol_frozen.json"))
lo, tau, m, s = P["lo"], P["tau"], P["margin"], P["size"]
DELTA, ALPHAS, EM = 0.1, (0.015, 0.02, 0.03), cc.EM
GRID = np.round(np.arange(np.ceil(lo * 50) / 50 + 0.02, cc.HI_MAX + 1e-9, 0.02), 2)
KEYS = ("map50", "emerg_map50", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car", "emerg_recall", "car_as_emerg")
EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


def fin(B, S, L):
    B, S, L = np.concatenate(B), np.concatenate(S), np.concatenate(L)
    if len(B):
        kk = torchvision.ops.batched_nms(torch.from_numpy(B.astype(np.float32)), torch.from_numpy(S.astype(np.float32)),
                                         torch.from_numpy(L.astype(np.int64)), 0.5).numpy()[:100]
        B, S, L = B[kk], S[kk], L[kk]
    return B.astype(np.float32).reshape(-1, 4), S.astype(np.float32), L.astype(np.int64)


def inside(hb, hs, hl, box):
    x1, y1, x2, y2 = box
    cx, cy = (hb[:, 0] + hb[:, 2]) / 2, (hb[:, 1] + hb[:, 3]) / 2
    k = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    return hb[k], hs[k], hl[k]


def v_gate(rg, row, hi, add):
    b, c, l, rid, reps = rg
    if len(b) == 0:
        return EMPTY, 0
    inb = np.array([(q in row["heavy"]) and (lo <= c[reps[q]] < hi) for q in range(len(reps))])
    emr = np.array([int(l[reps[q]]) in EM for q in range(len(reps))])
    act = inb & emr
    keep = ~act[rid]
    B, S, L, n = [b[keep]], [c[keep]], [l[keep]], 0
    for q in np.where(inb & (emr | add))[0]:
        n += 1
        xb, xs, xl = inside(*row["heavy"][q], b[reps[q]])
        if not emr[q]:
            k = np.isin(xl, EM); xb, xs, xl = xb[k], xs[k], xl[k]
        B.append(xb); S.append(xs); L.append(xl)
    return fin(B, S, L), n


man = pd.read_csv(cc.ROOT / "datasets/folds_v3_manifest.csv"); gmap = dict(zip(man["id"], man["group"]))
Ls, LD, HD, GT, RG, ROWS = [], [], [], [], [], []
for f in range(5):
    L = cc.load(cc.LIGHT, f, "val"); H = cc.load(cc.HEAVY, f, "val")
    Ls.append(L); LD.append([cc.cf(r) for r in L]); HD.append([cc.cf(r) for r in H])
    GT.append([(r["gt_boxes"], r["gt_cls"]) for r in L]); RG.append([cc.regions(r) for r in L])
    ROWS.append(pickle.load(open(cc.CC / f"val_m{m}_s{s}_fold{f}.pkl", "rb")))
gid = np.array([str(gmap.get(Path(r["image"]).stem, f"solo{f}_{i}")) for f in range(5) for i, r in enumerate(Ls[f])])
hh = [[cc.hit(g[0], g[1], *HD[f][i], tau) for i, g in enumerate(GT[f])] for f in range(5)]
pop = np.array([h.any() for f in range(5) for h in hh[f]]); N = len(pop)
print(f"images={N} population={int(pop.sum())} groups={len(set(gid[pop]))} lo={lo:.3f} tau={tau} delta={DELTA}")


def risk(D):
    lost = np.array([bool((hh[f][i] & ~cc.hit(GT[f][i][0], GT[f][i][1], *D[f][i], tau)).any())
                     for f in range(5) for i in range(len(GT[f]))])
    g = pd.Series(lost[pop]).groupby(gid[pop]).max()
    n, k = len(g), int(g.sum())
    return k / n, (1.0 if k >= n else float(beta.ppf(1 - DELTA, k + 1, n - k)))


def build(name, hi):
    D, nc = [], 0
    for f in range(5):
        if name == "image":
            E = [bool(((r["conf"] >= lo) & (r["conf"] < hi)).any()) for r in Ls[f]]
            D.append([HD[f][i] if E[i] else LD[f][i] for i in range(len(E))]); nc += sum(E)
        else:
            res = [v_gate(RG[f][i], ROWS[f][i], hi, name == "gate_add") for i in range(len(Ls[f]))]
            D.append([x[0] for x in res]); nc += sum(x[1] for x in res)
    return D, (nc / N if name == "image" else nc / N * (s / 640) ** 2)


def fold_metrics(D):
    per = [cc.metrics(D[f], GT[f], tau) for f in range(5)]
    return {k: float(np.nanmean([p.get(k, np.nan) for p in per])) for k in KEYS}


cur, rows = {}, []
for name in ("image", "gate", "gate_add"):
    cur[name] = {}
    for hi in GRID[::-1]:
        D, cost = build(name, hi); R, U = risk(D); cur[name][hi] = (R, U, cost)
        rows.append(dict(system=name, hi=float(hi), risk=R, ucb=U, cost=cost))
    print(name, "swept", flush=True)
pd.DataFrame(rows).to_csv(cc.RES / "gate_sweep_dev.csv", index=False)
R0, U0 = risk(LD)
print(f"light_only risk={R0:.4f} ucb={U0:.4f} | heavy_only risk=0 by definition")
for al in ALPHAS:
    for name in cur:
        sel = None
        for hi in GRID[::-1]:
            if cur[name][hi][1] <= al:
                sel = hi
            else:
                break
        if sel is None:
            print(f"alpha={al} {name:<8}: none certified"); continue
        D, cost = build(name, sel)
        print(f"alpha={al} {name:<8} hi={sel} cost={cost:.3f} risk={cur[name][sel][0]:.4f} ucb={cur[name][sel][1]:.4f}",
              {k: round(v, 4) for k, v in fold_metrics(D).items()}, flush=True)
for tag, D in (("light_only", LD), ("heavy_only", HD)):
    print(tag, {k: round(v, 4) for k, v in fold_metrics(D).items()})
