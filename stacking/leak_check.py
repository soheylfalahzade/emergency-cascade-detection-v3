import sys, re
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from PIL import Image, ImageOps

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 600
D = ROOT / f"datasets/folds/fold_{FOLD}"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dh(im):
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).astype(np.uint8).flatten()
    return int(np.packbits(bits).view(np.uint64)[0])


def hash_one(p):
    try:
        im = Image.open(p); im.load()
        return dh(im), dh(ImageOps.mirror(im))
    except Exception:
        return None


def nearest(hv, ht):
    dmin = np.empty(len(hv), dtype=np.int32); darg = np.empty(len(hv), dtype=np.int64)
    for i in range(0, len(hv), 50):
        x = np.ascontiguousarray(np.bitwise_xor(hv[i:i + 50, None], ht[None, :]))
        d = TABLE[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)
        dmin[i:i + 50] = d.min(1); darg[i:i + 50] = d.argmin(1)
    return dmin, darg


def src(n):
    return n.rsplit("_", 1)[0]


def main():
    tr = sorted(p for p in (D / "train/images").iterdir() if p.suffix.lower() in EXT)
    va_all = sorted(p for p in (D / "val/images").iterdir() if p.suffix.lower() in EXT)
    rng = np.random.RandomState(0)
    idx = sorted(rng.choice(len(va_all), min(NVAL, len(va_all)), replace=False))
    va = [va_all[i] for i in idx]
    aug = [p.name for p in tr if re.search(r"flip|noise|sharp|aug", p.name, re.I)]
    print(f"fold{FOLD}: train={len(tr)} val_sample={len(va)}/{len(va_all)} | train names with aug markers: {len(aug)} {aug[:4]}", flush=True)
    with Pool(4) as pool:
        htr = pool.map(hash_one, [str(p) for p in tr], chunksize=64)
        hva = pool.map(hash_one, [str(p) for p in va], chunksize=16)
    tr_ok = [(p, h) for p, h in zip(tr, htr) if h]; va_ok = [(p, h) for p, h in zip(va, hva) if h]
    print(f"hashed ok: train {len(tr_ok)}/{len(tr)}, val {len(va_ok)}/{len(va)}", flush=True)
    ht = np.array([h[0] for _, h in tr_ok], dtype=np.uint64)
    hv = np.array([h[0] for _, h in va_ok], dtype=np.uint64)
    hf = np.array([h[1] for _, h in va_ok], dtype=np.uint64)
    d0, a0 = nearest(hv, ht); d1, a1 = nearest(hf, ht)
    dm = np.minimum(d0, d1)
    s = np.array([src(p.name) for p, _ in va_ok])
    print(f"\n{'source':<26}{'n':>5}{'median':>8}{'<=4':>8}{'<=8':>8}{'flip<=4':>9}   (Hamming از 64 بیت، کمتر = شبیه‌تر)")
    for k in sorted(set(s)) + ["ALL"]:
        m = np.ones(len(s), bool) if k == "ALL" else s == k
        print(f"{k:<26}{m.sum():>5}{np.median(dm[m]):>8.1f}{(dm[m] <= 4).mean():>8.1%}{(dm[m] <= 8).mean():>8.1%}{(d1[m] <= 4).mean():>9.1%}")
    print("\nنزدیک‌ترین جفت‌ها (برای بازبینی چشمی، فاصله ≤ 4):")
    shown = 0
    for i in np.argsort(dm):
        if dm[i] > 4 or shown >= 8:
            break
        j = a0[i] if d0[i] <= d1[i] else a1[i]
        print(f"  d={dm[i]} {'flip' if d1[i] < d0[i] else 'direct'}  val={va_ok[i][0].name}  <->  train={tr_ok[j][0].name}")
        shown += 1


if __name__ == "__main__":
    main()
