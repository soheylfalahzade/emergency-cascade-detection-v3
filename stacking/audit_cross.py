#!/usr/bin/env python3
"""Cross-dataset leakage audit.

Usage:  python audit_cross.py [--examples N] A.yaml B.yaml [B2.yaml ...]

Asks: how many images of dataset A (every split it declares) have a
near-duplicate, or a horizontally mirrored near-duplicate, in dataset B
(every split it declares)? Typical use: A = a holdout set, B = the training
folds the model was fitted on. A non-zero count means holdout results are
optimistic.

Needs audit_yaml.py in the same folder. Hash distance <= 4 of 64 bits counts
as a near-duplicate. Different frames of the same video are NOT detected, so
zero here is not proof of independence.
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_yaml import SPLITS, HASH_HAMMING, dhash, list_images, resolve  # noqa: E402

LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(x):
    return LUT[x.view(np.uint8)].reshape(-1, 8).sum(axis=1)


def collect(yaml_path):
    d = yaml.safe_load(open(yaml_path))
    root = d.get("path", os.path.dirname(os.path.abspath(yaml_path)))
    files, seen = [], set()
    for s in SPLITS:
        if not d.get(s):
            continue
        vals = d[s] if isinstance(d[s], list) else [d[s]]
        key = tuple(sorted(str(resolve(root, v)) for v in vals))
        if key in seen:
            continue
        seen.add(key)
        for v in vals:
            files += list_images(resolve(root, v))
    return files


def hashes(files):
    out_f, out_p, paths = [], [], []
    for p in files:
        h = dhash(p)
        if h is not None:
            out_p.append(h[0])
            out_f.append(h[1])
            paths.append(p)
    return np.array(out_p, dtype=np.uint64), np.array(out_f, dtype=np.uint64), paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b", nargs="+")
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()

    a_files = collect(args.a)
    ha, haf, pa = hashes(a_files)
    print(f"\n## A = {args.a}: {len(pa)} unique-location images hashed")
    for b_yaml in args.b:
        hb, _, pb = hashes(collect(b_yaml))
        n_plain = n_mirror = n_exact = 0
        examples = []
        for i in range(len(pa)):
            d_plain = popcount(hb ^ ha[i])
            d_flip = popcount(hb ^ haf[i])
            j1, j2 = int(d_plain.argmin()), int(d_flip.argmin())
            best_plain, best_flip = int(d_plain[j1]), int(d_flip[j2])
            if min(best_plain, best_flip) <= HASH_HAMMING:
                if best_plain <= HASH_HAMMING and best_plain <= best_flip:
                    n_plain += 1
                    j, dist, mir = j1, best_plain, False
                else:
                    n_mirror += 1
                    j, dist, mir = j2, best_flip, True
                n_exact += dist == 0
                if len(examples) < args.examples:
                    examples.append((pa[i], pb[j], dist, mir))
        total = n_plain + n_mirror
        print(f"   vs {b_yaml}: {len(pb)} images")
        print(f"      A images with a near-duplicate in B: {total} of {len(pa)} "
              f"({100.0 * total / max(len(pa), 1):.1f}%)  "
              f"[direct {n_plain}, only after horizontal flip {n_mirror}, identical hash {n_exact}]")
        for ap_, bp_, dist, mir in examples:
            print(f"      pair (distance {dist}){' [mirrored]' if mir else ''}:\n        {ap_}\n        {bp_}")


if __name__ == "__main__":
    main()
