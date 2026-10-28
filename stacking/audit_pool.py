import os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageOps
from scipy.fftpack import dct

DS = Path.home() / "Portfolio/Smart_Traffic_System/datasets"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
EMERG = {"emergency_vehicles_raw", "ev_bahrain_v6", "ev_yolo_v1", "ev_police_v1"}
H = {}


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


def originals(d):
    out = [os.path.realpath(f) for f in sorted(d.iterdir())
           if f.suffix.lower() in EXT and not f.stem.endswith("_aug") and "_dup" not in f.stem]
    return sorted(set(out))


def src_of(p):
    return Path(p).parents[2].name


def nearest(q, ref):
    dm = np.empty(len(q), np.int32); da = np.empty(len(q), np.int64)
    for i in range(0, len(q), 64):
        x = np.ascontiguousarray(np.bitwise_xor(q[i:i + 64, None], ref[None, :]))
        d = TABLE[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)
        dm[i:i + 64] = d.min(1); da[i:i + 64] = d.argmin(1)
    return dm, da


def report(tag, qs, ts, montage=None):
    qs = [p for p in qs if p in H]; ts = [p for p in ts if p in H]
    t = np.array([H[p][0] for p in ts], np.uint64)
    v0 = np.array([H[p][0] for p in qs], np.uint64); v1 = np.array([H[p][1] for p in qs], np.uint64)
    d0, a0 = nearest(v0, t); d1, a1 = nearest(v1, t); dm = np.minimum(d0, d1)
    src = np.array([src_of(p) for p in qs])
    print(f"\n## {tag}: queries={len(qs)} reference={len(ts)}")
    print(f"{'source':<26}{'n':>6}{'median':>8}{'<=4':>8}{'<=6':>8}{'<=8':>8}")
    groups = [("ALL", np.ones(len(qs), bool)), ("EMERGENCY_SRC", np.isin(src, list(EMERG)))] + [(s, src == s) for s in sorted(set(src))]
    for name, m in groups:
        if m.sum():
            print(f"{name:<26}{m.sum():>6}{np.median(dm[m]):>8.1f}{(dm[m] <= 4).mean():>8.1%}{(dm[m] <= 6).mean():>8.1%}{(dm[m] <= 8).mean():>8.1%}")
    if montage:
        order = np.argsort(dm)[:8]
        sheet = Image.new("RGB", (960, 8 * 300), "white")
        for r, i in enumerate(order):
            j = a0[i] if d0[i] <= d1[i] else a1[i]
            for c, p in enumerate((qs[i], ts[j])):
                sheet.paste(Image.open(p).convert("RGB").resize((480, 300)), (c * 480, r * 300))
            print(f"  closest d={dm[i]}  query={Path(qs[i]).name[:50]}  ref={Path(ts[j]).name[:50]}")
        sheet.save(montage); print("saved", montage)


if __name__ == "__main__":
    folds = {k: {s: originals(DS / f"folds_v3/fold_{k}/{s}/images") for s in ("train", "val")} for k in range(5)}
    hold = originals(DS / "holdout_v3/images")
    allp = sorted(set(hold) | {p for k in folds for s in folds[k] for p in folds[k][s]})
    print("unique original images to hash:", len(allp), "| holdout:", len(hold), flush=True)
    with Pool(4) as pool:
        hs = pool.map(hash_one, allp, chunksize=64)
    H.update({p: h for p, h in zip(allp, hs) if h})
    print("hashed:", len(H), flush=True)
    for k in range(5):
        report(f"fold{k} VAL vs fold{k} TRAIN", folds[k]["val"], folds[k]["train"])
    hs_ = set(hold)
    report("HOLDOUT vs ALL other pool images", hold, [p for p in allp if p not in hs_], Path.home() / "audit_holdout_pairs.png")
