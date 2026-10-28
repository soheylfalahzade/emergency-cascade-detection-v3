import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from PIL import Image, ImageOps
from scipy.fftpack import dct

D = Path.home() / "Portfolio/Smart_Traffic_System/datasets"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def ph(im):
    a = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float64)
    c = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    bits = (c > np.median(c.flatten()[1:])).astype(np.uint8).flatten()
    return int(np.packbits(bits).view(np.uint64)[0])


def hash_one(p):
    try:
        im = Image.open(p); im.load()
        return ph(im), ph(ImageOps.mirror(im))
    except Exception:
        return None


def files(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)


def base_name(p):
    n = p.stem
    for s in ("_aug",):
        n = n.replace(s, "")
    import re
    return re.sub(r"_dup\d+", "", n)


def nearest(q, ref):
    dm = np.empty(len(q), np.int32); da = np.empty(len(q), np.int64)
    for i in range(0, len(q), 64):
        x = np.ascontiguousarray(np.bitwise_xor(q[i:i + 64, None], ref[None, :]))
        d = TABLE[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)
        dm[i:i + 64] = d.min(1); da[i:i + 64] = d.argmin(1)
    return dm, da


def run(fold):
    tr = files(D / f"folds_v3/fold_{fold}/train/images")
    seen, trs = set(), []
    for p in tr:
        b = base_name(p)
        if b in seen or p.stem.endswith("_aug"):
            continue
        seen.add(b); trs.append(p)
    trs = [p for p in trs]
    va = files(D / f"folds_v3/fold_{fold}/val/images") + files(D / "holdout_v3/images")
    print(f"fold{fold}: train unique originals={len(trs)} (aug/dup collapsed) | val+holdout={len(va)}", flush=True)
    with Pool(12) as pool:
        ht = pool.map(hash_one, [str(p) for p in trs], chunksize=64)
        hv = pool.map(hash_one, [str(p) for p in va], chunksize=32)
    T = [(p, h) for p, h in zip(trs, ht) if h]; V = [(p, h) for p, h in zip(va, hv) if h]
    t = np.array([h[0] for _, h in T], np.uint64)
    v0 = np.array([h[0] for _, h in V], np.uint64); v1 = np.array([h[1] for _, h in V], np.uint64)
    d0, a0 = nearest(v0, t); d1, a1 = nearest(v1, t)
    dm = np.minimum(d0, d1)
    src = np.array([p.name.split("__")[0] for p, _ in V])
    print(f"{'source':<26}{'n':>5}{'median':>8}{'<=4':>8}{'<=6':>8}{'<=8':>8}")
    for k in sorted(set(src)) + ["ALL"]:
        m = np.ones(len(src), bool) if k == "ALL" else src == k
        print(f"{k:<26}{m.sum():>5}{np.median(dm[m]):>8.1f}{(dm[m] <= 4).mean():>8.1%}{(dm[m] <= 6).mean():>8.1%}{(dm[m] <= 8).mean():>8.1%}")
    print("closest pairs:")
    for i in np.argsort(dm)[:6]:
        j = a0[i] if d0[i] <= d1[i] else a1[i]
        print(f"  d={dm[i]}  val={V[i][0].name[:60]}  <->  train={T[j][0].name[:60]}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
