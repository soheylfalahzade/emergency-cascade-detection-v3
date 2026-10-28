import sys, cv2, numpy as np
from pathlib import Path
from collections import Counter
EXT = {".jpg", ".jpeg", ".png"}
def dh(p):
    im = cv2.imread(str(p), 0)
    if im is None: return None
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    return (s[:, 1:] > s[:, :-1]).flatten().astype(np.uint8)
def load(d):
    fs = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXT)
    ok, H = [], []
    for p in fs:
        h = dh(p)
        if h is not None: ok.append(p); H.append(h)
    return ok, np.array(H)
def labels(files):
    c, sz = Counter(), []
    for p in files:
        lp = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lp.exists(): continue
        for ln in lp.read_text().splitlines():
            t = ln.split()
            if len(t) >= 5:
                c[int(t[0])] += 1; sz.append((float(t[3]) * float(t[4])) ** 0.5 * 960)
    return c, np.array(sz)
tf, Ht = load(sys.argv[1]); vf, Hv = load(sys.argv[2])
print("train imgs:", len(tf), "val imgs:", len(vf))
for name, fs in (("train", tf), ("val", vf)):
    c, sz = labels(fs)
    print(name, "class counts:", dict(sorted(c.items())))
    if len(sz): print(name, "box size px @960 (p5/p50/p95):", np.percentile(sz, [5, 50, 95]).round(1))
mins = []
for i in range(0, len(Hv), 100):
    d = (Hv[i:i+100, None, :] != Ht[None]).sum(2)
    mins += list(d.min(1))
mins = np.array(mins)
print("val images with near-duplicate in train (hamming<=4):", int((mins <= 4).sum()), "/", len(mins))
print("val images with hamming<=8:", int((mins <= 8).sum()))
