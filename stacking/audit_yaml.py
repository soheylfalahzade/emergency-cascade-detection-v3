#!/usr/bin/env python3
"""Audit YOLO-format datasets described by one or more data.yaml files.

Usage:  python audit_yaml.py [--examples N] path/to/data.yaml [more.yaml ...]
        --examples N  also print up to N sample cross-split pairs per yaml, so the
                      matches can be inspected by eye (hash matches can be false
                      positives on low-texture images)

For every split present in the yaml (train / val / test, whichever exist):
  - image count, labelled-object count per class, images with empty labels,
    missing label files, malformed label lines, tiny boxes
  - near-duplicate images (perceptual difference hash) inside a split
  - near-duplicate images ACROSS splits, which is a direct leakage signal

A yaml with no train/val/test keys is reported and skipped, not a crash.
Image/label pairing follows the YOLO convention: .../images/x.jpg -> .../labels/x.txt
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
HASH_HAMMING = 4        # <= this many differing bits (of 64) counts as a near-duplicate
TINY = 0.005            # box width or height below this fraction of the image


def resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() else (Path(root) / p).resolve()


def list_images(path):
    p = Path(path)
    if p.is_dir():
        return sorted(x for x in p.rglob("*") if x.suffix.lower() in EXT)
    if p.suffix == ".txt" and p.exists():
        base = p.parent
        out = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                q = Path(line)
                out.append(q if q.is_absolute() else (base / q).resolve())
        return out
    return []


def label_path(img):
    s = str(img)
    i = s.rfind("/images/")
    if i >= 0:
        return Path(s[:i] + "/labels/" + s[i + len("/images/"):]).with_suffix(".txt")
    return Path(s).with_suffix(".txt")


def _dhash_one(gray):
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash(path):
    """(hash of the image, hash of its horizontal mirror). The mirror hash lets a
    flipped copy of a training image be recognised in another split."""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return _dhash_one(im), _dhash_one(im[:, ::-1].copy())


def hamming(a, b):
    return bin(a ^ b).count("1")


def class_names(d):
    names = d.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {}


def audit_split(files):
    counts, empty, missing, bad, tiny = Counter(), 0, 0, 0, 0
    for p in files:
        lp = label_path(p)
        if not lp.exists():
            missing += 1
            continue
        lines = [l for l in lp.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
        for l in lines:
            parts = l.split()
            try:
                c = int(float(parts[0]))
                w, h = float(parts[3]), float(parts[4])
                if len(parts) < 5:
                    raise ValueError
            except (ValueError, IndexError):
                bad += 1
                continue
            counts[c] += 1
            if w < TINY or h < TINY:
                tiny += 1
    return counts, empty, missing, bad, tiny


def near_duplicates(hashes_by_split):
    """Return (within-split pairs, cross-split pairs) using exact and near hash matches.
    Exact-hash buckets plus a linear scan over bucket representatives keep it fast
    enough for tens of thousands of images."""
    items = [(s, p, h) for s, lst in hashes_by_split.items() for p, h in lst if h is not None]
    within, cross, exact, flipped = Counter(), Counter(), Counter(), Counter()
    examples = []
    reps = []   # (hash, split, path)
    for s, p, hh in items:
        h, hf = hh
        hit = None
        for rh, rs, rp in reps:
            d_plain, d_flip = hamming(h, rh), hamming(hf, rh)
            dist = min(d_plain, d_flip)
            if dist <= HASH_HAMMING:
                hit = (rs, rp, dist, d_flip < d_plain)
                break
        if hit is None:
            reps.append((h, s, p))
        else:
            key = tuple(sorted((hit[0], s)))
            (within if hit[0] == s else cross)[key] += 1
            if hit[0] != s:
                if hit[2] == 0:
                    exact[key] += 1
                if hit[3]:
                    flipped[key] += 1
                examples.append((hit[1], p, hit[2], hit[3]))
    return within, cross, exact, flipped, examples


def main(paths, n_examples=0):
    for y in paths:
        print(f"\n## {y}")
        try:
            d = yaml.safe_load(open(y))
        except Exception as exc:
            print(f"   cannot read yaml: {exc}")
            continue
        root = d.get("path", Path(y).parent)
        names = class_names(d)
        present = [s for s in SPLITS if d.get(s)]
        if not present:
            print("   no train/val/test key in this yaml; nothing to audit.")
            continue
        print(f"   classes: {names}")
        hashes = defaultdict(list)
        seen_locations = {}
        for s in present:
            val = d[s]
            vals = val if isinstance(val, list) else [val]
            loc = tuple(sorted(str(resolve(root, v)) for v in vals))
            if loc in seen_locations:
                print(f"   [{s}] points to the same location as [{seen_locations[loc]}] "
                      f"(typical for a holdout yaml); cross-split check skipped for this pair")
                continue
            seen_locations[loc] = s
            files = []
            for v in vals:
                files += list_images(resolve(root, v))
            counts, empty, missing, bad, tiny = audit_split(files)
            per_class = {names.get(c, c): n for c, n in sorted(counts.items())}
            print(f"   [{s}] images={len(files)} objects={sum(counts.values())} "
                  f"empty_labels={empty} missing_label_files={missing} "
                  f"malformed_lines={bad} tiny_boxes={tiny}")
            print(f"        per class: {per_class}")
            hashes[s] = [(p, dhash(p)) for p in files]
        if len(present) >= 1:
            within, cross, exact, flipped, examples = near_duplicates(hashes)
            for k, n in sorted(within.items()):
                print(f"   near-duplicates inside {k[0]}: {n}")
            if cross:
                for k, n in sorted(cross.items()):
                    print(f"   LEAKAGE near-duplicates between {k[0]} and {k[1]}: {n} "
                          f"(identical hash: {exact.get(k, 0)}, "
                          f"match only after horizontal flip: {flipped.get(k, 0)})")
                for rp, p, dist, fl in examples[:n_examples]:
                    tag = " [mirrored]" if fl else ""
                    print(f"      pair (hash distance {dist}){tag}:\n        {rp}\n        {p}")
            else:
                print("   no near-duplicates found across splits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yamls", nargs="+")
    ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args()
    main(a.yamls, a.examples)
