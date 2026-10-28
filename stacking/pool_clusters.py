import numpy as np
from pathlib import Path
from multiprocessing import Pool
from PIL import Image, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

M = Path.home() / "Portfolio/Smart_Traffic_System/datasets/emergency_merged"
OUT = Path.home() / "Portfolio/Smart_Traffic_System/stacking/pool_hashes.npz"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def dh(im):
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int(np.packbits((a[:, 1:] > a[:, :-1]).astype(np.uint8).flatten()).view(np.uint64)[0])

def hash_one(p):
    try:
        im = Image.open(p); im.load()
        return dh(im), dh(ImageOps.mirror(im))
    except Exception:
        return None

if __name__ == "__main__":
    items = [(sp, p) for sp in ("train", "valid", "test") for p in sorted((M / sp / "images").iterdir()) if p.suffix.lower() in EXT]
    with Pool(12) as pool:
        hs = pool.map(hash_one, [str(p) for _, p in items], chunksize=64)
    ok = [(it, h) for it, h in zip(items, hs) if h]
    print(f"hashed {len(ok)}/{len(items)}")
    split = np.array([it[0] for it, _ in ok]); name = np.array([it[1].name for it, _ in ok])
    h = np.array([x[0] for _, x in ok], dtype=np.uint64); hf = np.array([x[1] for _, x in ok], dtype=np.uint64)
    N = len(h); I, J, Dd = [], [], []
    for i in range(0, N, 64):
        d = None
        for q in (h, hf):
            x = np.ascontiguousarray(np.bitwise_xor(q[i:i + 64, None], h[None, :]))
            dq = TABLE[x.view(np.uint8)].reshape(x.shape[0], N, 8).sum(-1, dtype=np.int16)
            d = dq if d is None else np.minimum(d, dq)
        a, b = np.nonzero(d <= 8)
        a = a + i; keep = a < b
        I.append(a[keep]); J.append(b[keep]); Dd.append(d[a[keep] - i, b[keep]])
        if (i // 64) % 100 == 0: print(f"  {i}/{N}", flush=True)
    I, J, Dd = np.concatenate(I), np.concatenate(J), np.concatenate(Dd)
    np.savez(OUT, split=split, name=name, I=I, J=J, D=Dd)
    print(f"pairs(d<=8): {len(I)} saved {OUT}")
    for T in (0, 2, 4, 6, 8):
        m = Dd <= T
        g = coo_matrix((np.ones(m.sum()), (I[m], J[m])), shape=(N, N))
        n, lab = connected_components(g, directed=False)
        sz = np.bincount(lab)
        multi = (sz[lab] > 1).mean()
        te = split == "test"; trv = ~te
        te_leak = 0
        for c in np.unique(lab[te]):
            if (trv & (lab == c)).any(): te_leak += (te & (lab == c)).sum()
        print(f"T<={T}: components={n} largest={sz.max()} imgs_in_multi={multi:.1%} | test imgs sharing cluster with train/valid: {te_leak}/{te.sum()} ({te_leak/max(te.sum(),1):.1%})")
