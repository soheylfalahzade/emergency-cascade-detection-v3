"""Similarity-stratified evaluation: metrics on validation / held-out images that have no near neighbour among the fold's training files.
Distance = 64-bit pHash Hamming distance to the nearest training file (query also hashed mirrored)."""
import json, pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2
import crop_cascade as cc
import image_cascade_final as ic
from stack_eval import aligned

DS, RES = cc.ROOT / "datasets", cc.RES
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
CACHE = RES / "phash_cache.pkl"
H = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
TS = (-1, 6, 10, 14, 18)
P = json.load(open(RES / "image_cascade_protocol_frozen.json")); HI = P["hi"]["0.02"]; LO, TAU = ic.LO, ic.TAU
KEYS = ("map50", "emerg_map50", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car")


def hashes(p):
    g = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if g is None:
        return None
    out = []
    for im in (g, g[:, ::-1]):
        x = cv2.resize(np.ascontiguousarray(im), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        d = cv2.dct(x)[:8, :8].flatten()
        out.append(np.packbits(d > np.median(d[1:])))
    return out


def hash_dir(d, prefix):
    ps = sorted(p for p in d.iterdir() if p.suffix.lower() in cc.EXT)
    need = [p for p in ps if prefix + p.name not in H]
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(hashes, need))
    for p, r in zip(need, res):
        if r is not None:
            H[prefix + p.name] = r
    return [prefix + p.name for p in ps if prefix + p.name in H]


def near(qkeys, tkeys, chunk=64):
    Q = np.stack([H[k][0] for k in qkeys]); F = np.stack([H[k][1] for k in qkeys]); T = np.stack([H[k][0] for k in tkeys])
    out = np.empty(len(Q), np.int32)
    for s in range(0, len(Q), chunk):
        d1 = POP[np.bitwise_xor(Q[s:s + chunk, None, :], T[None])].sum(2, dtype=np.int32).min(1)
        d2 = POP[np.bitwise_xor(F[s:s + chunk, None, :], T[None])].sum(2, dtype=np.int32).min(1)
        out[s:s + chunk] = np.minimum(d1, d2)
    return out


def ev(D, GT):
    r = cc.metrics(D, GT, TAU)
    return [r.get(q, np.nan) for q in KEYS]


def casc(rows, Ld, Hd):
    E = [bool(((r["conf"] >= LO) & (r["conf"] < HI)).any()) for r in rows]
    return [Hd[i] if E[i] else Ld[i] for i in range(len(E))]


trk = {k: hash_dir(DS / f"folds_v3/fold_{k}/train/images", "T:") for k in range(5)}
vkeys = {k: hash_dir(DS / f"folds_v3/fold_{k}/val/images", f"V{k}:") for k in range(5)}
hall = hash_dir(DS / "holdout_v3/images", "H:")
pickle.dump(H, open(CACHE, "wb")); print("hashed:", len(H), flush=True)

d = ic.dev_data(); dist_val = []
for k in range(5):
    keys = [f"V{k}:{r['image']}" for r in d["L"][k]]
    assert all(q in H for q in keys), "missing hash"
    dist_val.append(near(keys, trk[k]))
print("VALIDATION (out-of-fold); cascade = alpha 0.02 operating point (hi=%s)" % HI)
for T in TS:
    acc = {s: [] for s in ("light", "heavy", "cascade")}; n = 0; boxes = np.zeros(4, int)
    for k in range(5):
        keep = np.where(dist_val[k] > T)[0]; n += len(keep)
        if len(keep) < 30:
            continue
        GT = [d["GT"][k][i] for i in keep]
        for _, gc in GT:
            for c in gc:
                boxes[int(c)] += 1
        Ld = [d["LD"][k][i] for i in keep]; Hd = [d["HD"][k][i] for i in keep]
        Cd = casc([d["L"][k][i] for i in keep], Ld, Hd)
        for s, D in (("light", Ld), ("heavy", Hd), ("cascade", Cd)):
            acc[s].append(ev(D, GT))
    print(f"d>{T}: images={n} ({n / sum(len(x) for x in dist_val):.0%} of validation) boxes(ambulance,police,firetruck,car)={boxes.tolist()}")
    for s, v in acc.items():
        if v:
            print(f"   {s:<8}", dict(zip(KEYS, np.round(np.nanmean(np.array(v), 0), 4))))

pm = {f: aligned(cc.MODELS, f, "holdout") for f in range(5)}
hk = ["H:" + r["image"] for r in pm[0][cc.LIGHT]]
tunion = sorted(set(k for v in trk.values() for k in v))
dist_h = near(hk, tunion)
GTh = [(r["gt_boxes"], r["gt_cls"]) for r in pm[0][cc.LIGHT]]
LDh = {f: [cc.cf(r) for r in pm[f][cc.LIGHT]] for f in range(5)}; HDh = {f: [cc.cf(r) for r in pm[f][cc.HEAVY]] for f in range(5)}
print("\nHOLDOUT (mean of 5 fold-models), post-hoc; distance to nearest file of any fold's training set")
for T in TS:
    keep = np.where(dist_h > T)[0]
    if len(keep) < 30:
        print(f"d>{T}: images={len(keep)} (too few)"); continue
    boxes = np.zeros(4, int)
    for i in keep:
        for c in GTh[i][1]:
            boxes[int(c)] += 1
    GT = [GTh[i] for i in keep]; acc = {s: [] for s in ("light", "heavy", "cascade")}
    for f in range(5):
        Ld = [LDh[f][i] for i in keep]; Hd = [HDh[f][i] for i in keep]
        Cd = casc([pm[f][cc.LIGHT][i] for i in keep], Ld, Hd)
        for s, D in (("light", Ld), ("heavy", Hd), ("cascade", Cd)):
            acc[s].append(ev(D, GT))
    print(f"d>{T}: images={len(keep)} ({len(keep) / len(GTh):.0%} of holdout) boxes(ambulance,police,firetruck,car)={boxes.tolist()}")
    for s, v in acc.items():
        print(f"   {s:<8}", dict(zip(KEYS, np.round(np.nanmean(np.array(v), 0), 4))))
