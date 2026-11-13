import re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, cv2

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"; DS = ROOT / "datasets"; RES = ROOT / "results_v3"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
man = pd.read_csv(DS / "folds_v3_manifest.csv"); man["id"] = man["id"].astype(str); man["group"] = man["group"].astype(str)
gmap = dict(zip(man["id"], man["group"])); smap = dict(zip(man["id"], man["src"]))
SUF = re.compile(r"(_aug\d*|_dup\d*|_os\d*|_copy\d*|_flip\d*|_noise\d*|_sharp\d*)$")
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def base(s):
    for _ in range(4):
        if s in gmap:
            return s
        t = SUF.sub("", s)
        if t == s:
            break
        s = t
    return s if s in gmap else None


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


def load(paths):
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(hashes, paths))
    ok = [i for i, r in enumerate(res) if r is not None]
    return ok, np.stack([res[i][0] for i in ok]), np.stack([res[i][1] for i in ok])


def nearest(Q, T, chunk=32):
    dm, ag = np.empty(len(Q), np.int32), np.empty(len(Q), np.int64)
    for s in range(0, len(Q), chunk):
        d = POP[np.bitwise_xor(Q[s:s + chunk, None, :], T[None, :, :])].sum(2, dtype=np.int32)
        ag[s:s + chunk], dm[s:s + chunk] = d.argmin(1), d.min(1)
    return dm, ag


def both(Q, F, T):
    d1, a1 = nearest(Q, T); d2, a2 = nearest(F, T)
    return np.minimum(d1, d2), np.where(d1 <= d2, a1, a2)


ls = lambda d: sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)
hold, train, val0 = ls(DS / "holdout_v3/images"), ls(DS / "folds_v3/fold_0/train/images"), ls(DS / "folds_v3/fold_0/val/images")
okh, Hq, Hf = load(hold); okt, Tt, _ = load(train); okv, Vq, Vf = load(val0)
print(f"hashed: holdout {len(okh)}/{len(hold)}  train {len(okt)}/{len(train)}  val0 {len(okv)}/{len(val0)}", flush=True)
d, arg = both(Hq, Hf, Tt); dv, _ = both(Vq, Vf, Tt)
sh = lambda x: {t: round(float((x <= t).mean()), 3) for t in (2, 4, 6, 10)}
hg = np.array([gmap.get(base(hold[i].stem) or "", "") for i in okh]); is731 = hg == "731"
print("pHash distance to nearest fold-0 train image (share with distance <= t)")
print("  fold-0 val (reference)      ", len(dv), sh(dv))
print("  holdout, all                ", len(d), sh(d))
print("  holdout, group 731          ", int(is731.sum()), sh(d[is731]))
print("  holdout, other groups       ", int((~is731).sum()), sh(d[~is731]))
hs = np.array([smap.get(base(hold[i].stem) or "", "?") for i in okh])
for s_ in sorted(set(hs[~is731])):
    m = (hs == s_) & ~is731
    print(f"    other groups, source {s_:<16}", int(m.sum()), sh(d[m]))
tg = np.array([gmap.get(base(train[i].stem) or "", "") for i in okt]); i731 = np.where(tg == "731")[0]
print(f"train images of group 731 in fold 0: {len(i731)}")
if len(i731) and is731.any():
    dg, _ = both(Hq[is731], Hf[is731], Tt[i731])
    print("  group-731 holdout vs group-731 train images:", sh(dg), "min", int(dg.min()), "median", float(np.median(dg)))
print("  source of nearest train image, group-731 holdout:", Counter(smap.get(base(train[okt[a]].stem) or "", "?") for a in arg[is731]).most_common(4))


def tile(p, label):
    im = cv2.resize(cv2.imread(str(p)), (320, 240))
    cv2.putText(im, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return im


sel = np.where(is731)[0]; order = sel[np.argsort(d[sel])][:8]; tiles = []
for j in order:
    hp, tp = hold[okh[j]], train[okt[arg[j]]]
    tiles += [tile(hp, f"holdout d={d[j]}"), tile(tp, f"train {smap.get(base(tp.stem), '?')}")]
while len(tiles) < 16:
    tiles.append(np.zeros((240, 320, 3), np.uint8))
out = RES / "diag"; out.mkdir(exist_ok=True)
cv2.imwrite(str(out / "g731_nearest_pairs.jpg"), np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)]))
print("saved", out / "g731_nearest_pairs.jpg")
