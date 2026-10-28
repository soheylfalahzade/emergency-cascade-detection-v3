import os, sys, csv, shutil, random
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
import make_folds_clean as m

SEED, K, HOLDOUT_SPLITS = 42, 5, 10
TRAIN_ONLY = ["kitti_yolo", "miotcd_yolo", "visdrone_yolo"]
EMERG = ["emergency_vehicles_raw", "ev_bahrain_v6", "ev_yolo_v1", "ev_police_v1"]
T_GROUP = 2
T_EMERG = int(os.environ.get("T_EMERG", 6))
T_PURGE = int(os.environ.get("T_PURGE", 6))
MAX_GROUP_FRAC = 0.08
BASE = m.BASE
POOL, FOLDS, HOLD = BASE / "pool_v3", BASE / "folds_v3", BASE / "holdout_v3"
m.POOL = POOL


def main():
    random.seed(SEED); np.random.seed(SEED)
    recs = m.collect()
    print(f"pool (originals only): {len(recs)}")
    with Pool(12) as pool:
        hs = pool.map(m.hash_one, [str(r["img"]) for r in recs], chunksize=32)
    ok = [i for i, h in enumerate(hs) if h]
    print(f"hash failures dropped: {len(recs) - len(ok)}")
    recs = [recs[i] for i in ok]; hs = [hs[i] for i in ok]
    N = len(recs)
    h = np.array([x[0] for x in hs], dtype=np.uint64); hf = np.array([x[1] for x in hs], dtype=np.uint64)
    I, J, D = m.near_pairs(h, hf, max(T_PURGE, T_EMERG, T_GROUP))
    src = np.array([r["src"] for r in recs])
    emerg = np.isin(src, EMERG)
    edge = (D <= T_GROUP) | (emerg[I] & emerg[J] & (D <= T_EMERG))
    ei, ej, first, ni, nj = [I[edge]], [J[edge]], {}, [], []
    for k, r in enumerate(recs):
        key = m.name_key(r)
        if key is None:
            continue
        if key in first:
            ni.append(first[key]); nj.append(k)
        else:
            first[key] = k
    ei.append(np.array(ni, dtype=np.int64)); ej.append(np.array(nj, dtype=np.int64))
    ei, ej = np.concatenate(ei), np.concatenate(ej)
    ng, lab = connected_components(coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(N, N)), directed=False)

    elig = np.where(~np.isin(src, TRAIN_ONLY))[0]
    sz = np.bincount(lab[elig])
    print(f"eligible={len(elig)} train_only={N - len(elig)} groups(eligible)={(sz > 0).sum()} largest={sz.max()} ({sz.max() / len(elig):.1%})")
    if sz.max() > MAX_GROUP_FRAC * len(elig):
        print("ABORT: giant group among eligible"); sys.exit(2)

    y = np.array([m.PRIORITY.index(r["rarity"]) for r in recs])
    rest_i, hold_i = next(iter(StratifiedGroupKFold(HOLDOUT_SPLITS, shuffle=True, random_state=SEED).split(elig, y[elig], lab[elig])))
    where = np.full(N, -3)
    where[elig[hold_i]] = -1
    rest = elig[rest_i]
    for f, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=SEED).split(rest, y[rest], lab[rest])):
        where[rest[te]] = f
    for g in np.unique(lab[elig]):
        assert len(set(where[elig][lab[elig] == g])) == 1, "group straddles sets"

    def summ(name, idx):
        c = Counter()
        for k in idx:
            c.update(m.CLASSES[int(l.split()[0])] for l in recs[k]["lines"])
        print(f"{name:<10} images={len(idx):>6} boxes={dict(c)}")
    summ("holdout", np.where(where == -1)[0])
    for f in range(K):
        summ(f"fold{f}val", np.where(where == f)[0])

    plans = {}
    for f in range(K):
        inV = (where == f) | (where == -1)
        p = np.zeros(N, bool)
        mk = D <= T_PURGE
        s = mk & inV[I] & ~inV[J]; p[J[s]] = True
        s = mk & inV[J] & ~inV[I]; p[I[s]] = True
        tr = np.where(~inV & ~p)[0]
        trm = np.zeros(N, bool); trm[tr] = True
        assert not ((D <= T_PURGE) & ((trm[I] & inV[J]) | (trm[J] & inV[I]))).any(), "residual near-duplicate"
        pur = np.where(p)[0]
        print(f"fold{f}: train originals={len(tr)} purged={len(pur)} by_rarity={dict(Counter(recs[k]['rarity'] for k in pur))} by_src={dict(Counter(recs[k]['src'] for k in pur))}")
        c = Counter()
        for k in tr:
            c.update(m.CLASSES[int(l.split()[0])] for l in recs[k]["lines"])
        print(f"        train boxes (originals)={dict(c)}")
        plans[f] = tr

    for d in (POOL, FOLDS, HOLD):
        if d.exists():
            shutil.rmtree(d)
    (POOL / "labels").mkdir(parents=True)
    for r in recs:
        for name, _ in m.versions(r, True):
            lines = r["lines"] if name == r["id"] else r["aug_lines"]
            with open(POOL / "labels" / (name + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + "\n")
    m.emit(HOLD, np.where(where == -1)[0], recs, False, False)
    m.write_yaml(HOLD / "data.yaml", HOLD / "images", HOLD / "images")
    for f in range(K):
        fd = FOLDS / f"fold_{f}"
        ntr = m.emit(fd / "train", plans[f], recs, True, True)
        nva = m.emit(fd / "val", np.where(where == f)[0], recs, False, False)
        m.write_yaml(fd / "data.yaml", fd / "train/images", fd / "val/images")
        print(f"fold{f}: train files={ntr} val files={nva}")
    with open(BASE / "folds_v3_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "src", "group", "rarity", "set", f"seed={SEED}", f"T_PURGE={T_PURGE}"])
        for k, r in enumerate(recs):
            st = "holdout" if where[k] == -1 else ("train_only" if where[k] == -3 else f"fold{where[k]}")
            w.writerow([r["id"], r["src"], int(lab[k]), r["rarity"], st])
    print("BUILD OK")


if __name__ == "__main__":
    main()
