import re
from pathlib import Path
from collections import Counter
import pandas as pd

DS = Path.home() / "Portfolio/Smart_Traffic_System/datasets"
man = pd.read_csv(DS / "folds_v3_manifest.csv")
man["id"], man["group"] = man["id"].astype(str), man["group"].astype(str)
gmap = dict(zip(man["id"], man["group"]))
print("sets:", man["set"].value_counts().to_dict())
print("\ngroup 731 by (set, src):"); print(man[man["group"] == "731"].groupby(["set", "src"]).size().to_string())

tr_only, vh = man[man["set"] == "train_only"], man[man["set"] != "train_only"]
both = set(tr_only["group"]) & set(vh["group"])
print(f"\ngroups present in train_only AND in a val/holdout set: {len(both)} | their train_only images: {int(tr_only['group'].isin(both).sum())}")
rows = [(g, int((tr_only['group'] == g).sum()), vh[vh['group'] == g]['set'].value_counts().to_dict()) for g in both]
for r in sorted(rows, key=lambda x: -x[1])[:10]:
    print("  group", r[0], "train_only images", r[1], "| val/holdout:", r[2], "| src:", tr_only[tr_only['group'] == r[0]]['src'].value_counts().to_dict())
multi = man[man["set"] != "train_only"].groupby("group")["set"].nunique()
print("groups spanning more than one val/holdout set:", int((multi > 1).sum()))

SUF = re.compile(r"(_aug\d*|_dup\d*|_os\d*|_copy\d*|_flip\d*|_noise\d*|_sharp\d*)$")


def base(s):
    for _ in range(4):
        if s in gmap:
            return s
        t = SUF.sub("", s)
        if t == s:
            break
        s = t
    return s if s in gmap else None


hg = set(man.loc[man["set"] == "holdout", "group"])
print("\nactual training files per fold:")
for k in range(5):
    tr = [p.stem for p in (DS / f"folds_v3/fold_{k}/train/images").iterdir()]
    vg = set(man.loc[man["set"] == f"fold{k}", "group"])
    cv, ch, un = Counter(), Counter(), []
    for s in tr:
        b = base(s)
        if b is None:
            un.append(s); continue
        g = gmap[b]
        if g in vg: cv[g] += 1
        if g in hg: ch[g] += 1
    print(f"fold{k}: train files={len(tr)} unmatched={len(un)} | in this fold's val groups: {sum(cv.values())} ({len(cv)} groups) | in holdout groups: {sum(ch.values())} ({len(ch)} groups)")
    print("   top holdout-overlap:", ch.most_common(5), "| top val-overlap:", cv.most_common(5))
    if un:
        print("   unmatched examples:", un[:5])
