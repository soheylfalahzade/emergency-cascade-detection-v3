import os, re, sys, csv, shutil, random
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import yaml
from PIL import Image, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

SEED, K, HOLDOUT_SPLITS = 42, 5, 10
T_GROUP, T_REPORT = 2, 6
MAX_GROUP_FRAC = 0.08
BASE = Path.home() / "Portfolio/Smart_Traffic_System/datasets"
POOL, FOLDS, HOLD = BASE / "clean_pool", BASE / "folds_clean", BASE / "holdout_clean"
CLASSES = ["ambulance", "police", "firetruck", "car"]
NAME_TO_ID = {n: i for i, n in enumerate(CLASSES)}
PRIORITY = ["police", "firetruck", "ambulance", "car"]
DUP_FACTOR = {"police": 8, "firetruck": 3, "ambulance": 2, "car": 1}
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
SOURCES = [
    (BASE / "emergency_vehicles_raw", {0: "ambulance", 1: "police"}),
    (BASE / "ev_bahrain_v6", {0: "ambulance", 1: "ambulance", 2: "firetruck", 3: "firetruck"}),
    (BASE / "ev_yolo_v1", {0: "ambulance", 1: "ambulance", 2: "firetruck", 3: "firetruck"}),
    (BASE / "ev_police_v1", {0: "police"}),
    (BASE / "kitti_yolo", {0: "car"}),
    (BASE / "uadetrac_yolo", {0: "car"}),
    (BASE / "visdrone_yolo", {0: "car"}),
    (BASE / "miotcd_yolo", {0: "car"}),
]
odd_cols = Counter()


def mapped_lines(lp, cmap):
    out = []
    with open(lp) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old = int(parts[0])
            if old not in cmap:
                continue
            if len(parts) != 5:
                odd_cols[lp.parent.parent.parent.name] += 1
            out.append(" ".join([str(NAME_TO_ID[cmap[old]])] + parts[1:]))
    return out


def collect():
    recs = []
    for sdir, cmap in SOURCES:
        for split in ("train", "valid", "test"):
            idir, ldir = sdir / split / "images", sdir / split / "labels"
            if not idir.exists():
                continue
            for p in sorted(idir.iterdir()):
                if p.suffix.lower() not in EXT or p.stem.endswith("_aug"):
                    continue
                lp = ldir / (p.stem + ".txt")
                if not lp.exists():
                    continue
                lines = mapped_lines(lp, cmap)
                if not lines:
                    continue
                ap, alp = idir / (p.stem + "_aug" + p.suffix), ldir / (p.stem + "_aug.txt")
                alines = mapped_lines(alp, cmap) if ap.exists() and alp.exists() else None
                cls = {CLASSES[int(l.split()[0])] for l in lines}
                rarity = next(c for c in PRIORITY if c in cls)
                recs.append(dict(id=f"{sdir.name}__{split}__{p.stem}", src=sdir.name, split=split, img=p,
                                 lines=lines, aug=ap if alines else None, aug_lines=alines or None, rarity=rarity))
    return recs


def dh(im):
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int(np.packbits((a[:, 1:] > a[:, :-1]).astype(np.uint8).flatten()).view(np.uint64)[0])


def hash_one(p):
    try:
        im = Image.open(p); im.load()
        return dh(im), dh(ImageOps.mirror(im))
    except Exception:
        return None


def near_pairs(h, hf, T):
    N = len(h); I, J, D = [], [], []
    for i in range(0, N, 64):
        d = None
        for q in (h, hf):
            x = np.ascontiguousarray(np.bitwise_xor(q[i:i + 64, None], h[None, :]))
            dq = TABLE[x.view(np.uint8)].reshape(x.shape[0], N, 8).sum(-1, dtype=np.int16)
            d = dq if d is None else np.minimum(d, dq)
        a, b = np.nonzero(d <= T)
        keep = (a + i) < b
        I.append((a + i)[keep]); J.append(b[keep]); D.append(d[a[keep], b[keep]])
        if (i // 64) % 50 == 0:
            print(f"  hash pairs {i}/{N}", flush=True)
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def name_key(r):
    n = r["img"].name
    m = re.match(r"(MVI_\d+)_", n)
    if m:
        return f'{r["src"]}:{m.group(1)}'
    if ".rf." in n:
        return f'{r["src"]}:{n.split(".rf.")[0]}'
    return None


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(Path(src).resolve(), dst)


def versions(r, with_aug):
    v = [(r["id"], r["img"])]
    if with_aug and r["aug"]:
        v.append((r["id"] + "_aug", r["aug"]))
    return v


def emit(d, members, recs, with_aug, dup):
    (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for k in members:
        r = recs[k]
        reps = DUP_FACTOR[r["rarity"]] if dup else 1
        for name, img in versions(r, with_aug):
            for j in range(reps):
                nm = name if j == 0 else f"{name}_dup{j}"
                link(img, d / "images" / (nm + img.suffix))
                link(POOL / "labels" / (name + ".txt"), d / "labels" / (nm + ".txt"))
                n += 1
    return n


def write_yaml(path, train, val):
    with open(path, "w") as f:
        yaml.dump({"train": str(train.resolve()), "val": str(val.resolve()), "nc": len(CLASSES), "names": CLASSES},
                  f, default_flow_style=False, sort_keys=False)


def main():
    random.seed(SEED); np.random.seed(SEED)
    recs = collect()
    print(f"pool (originals only): {len(recs)}  | expected 13122 if nothing else differs")
    print("rows with !=5 columns per source (should be empty):", dict(odd_cols))
    with Pool(12) as pool:
        hs = pool.map(hash_one, [str(r["img"]) for r in recs], chunksize=32)
    bad = sum(h is None for h in hs)
    print(f"hash failures dropped: {bad}")
    recs = [r for r, h in zip(recs, hs) if h]; hs = [h for h in hs if h]
    N = len(recs)
    h = np.array([x[0] for x in hs], dtype=np.uint64); hf = np.array([x[1] for x in hs], dtype=np.uint64)
    I, J, D = near_pairs(h, hf, T_REPORT)
    m = D <= T_GROUP
    ei, ej, first = [I[m]], [J[m]], {}
    ni, nj = [], []
    for k, r in enumerate(recs):
        key = name_key(r)
        if key is None:
            continue
        if key in first:
            ni.append(first[key]); nj.append(k)
        else:
            first[key] = k
    ei.append(np.array(ni, dtype=np.int64)); ej.append(np.array(nj, dtype=np.int64))
    ei, ej = np.concatenate(ei), np.concatenate(ej)
    ng, lab = connected_components(coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(N, N)), directed=False)
    sz = np.bincount(lab)
    print(f"groups={ng} largest={sz.max()} ({sz.max()/N:.1%}) singletons={(sz==1).sum()}")
    for g in np.argsort(-sz)[:5]:
        print("  big group", sz[g], dict(Counter(recs[k]["src"] for k in np.where(lab == g)[0])))
    if sz.max() > MAX_GROUP_FRAC * N:
        print("ABORT: giant group, lower T_GROUP or inspect"); sys.exit(2)

    y = np.array([PRIORITY.index(r["rarity"]) for r in recs]); idx = np.arange(N)
    rest, hold = next(iter(StratifiedGroupKFold(HOLDOUT_SPLITS, shuffle=True, random_state=SEED).split(idx, y, lab)))
    where = np.full(N, -2); where[hold] = -1
    for f, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=SEED).split(rest, y[rest], lab[rest])):
        where[rest[te]] = f
    assert (where != -2).all()
    gl = defaultdict(set)
    for k in range(N):
        gl[lab[k]].add(where[k])
    assert all(len(s) == 1 for s in gl.values()), "group straddles sets"
    cross = where[I] != where[J]
    assert not (cross & (D <= T_GROUP)).any()
    for T in (4, 6):
        mm = cross & (D <= T); ex = np.zeros(N, bool); ex[I[mm]] = True; ex[J[mm]] = True
        print(f"residual near-neighbors across sets, d<={T} (includes look-alike scenes): holdout {ex[where==-1].mean():.1%} | folds {ex[where>=0].mean():.1%}")

    def summ(name, members):
        c = Counter()
        for k in members:
            c.update(CLASSES[int(l.split()[0])] for l in recs[k]["lines"])
        print(f"{name:<9} images={len(members):>6} boxes={dict(c)}")
    summ("holdout", np.where(where == -1)[0])
    for f in range(K):
        summ(f"fold{f}val", np.where(where == f)[0])

    for d in (POOL, FOLDS, HOLD):
        if d.exists():
            shutil.rmtree(d)
    (POOL / "labels").mkdir(parents=True)
    for r in recs:
        for name, _ in versions(r, True):
            lines = r["lines"] if name == r["id"] else r["aug_lines"]
            with open(POOL / "labels" / (name + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
    emit(HOLD, np.where(where == -1)[0], recs, False, False)
    write_yaml(HOLD / "data.yaml", HOLD / "images", HOLD / "images")
    for f in range(K):
        fd = FOLDS / f"fold_{f}"
        ntr = emit(fd / "train", np.where((where >= 0) & (where != f))[0], recs, True, True)
        nva = emit(fd / "val", np.where(where == f)[0], recs, False, False)
        write_yaml(fd / "data.yaml", fd / "train/images", fd / "val/images")
        print(f"fold{f}: train files={ntr} val files={nva}")
    with open(BASE / "folds_clean_manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "src", "orig_split", "group", "rarity", "set", "has_aug", f"seed={SEED}"])
        for k, r in enumerate(recs):
            w.writerow([r["id"], r["src"], r["split"], int(lab[k]), r["rarity"], "holdout" if where[k] == -1 else f"fold{where[k]}", int(bool(r["aug"]))])
    print("BUILD OK")


if __name__ == "__main__":
    main()
