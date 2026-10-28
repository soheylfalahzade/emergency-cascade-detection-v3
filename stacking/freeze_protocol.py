import json, hashlib, time, os, sys
from pathlib import Path
import numpy as np

R = Path.home() / "Portfolio/Smart_Traffic_System"
RES, S = R / "results_v3", R / "stacking"
if (RES / "holdout_done.flag").exists():
    sys.exit("holdout already evaluated; protocol cannot change")
OUT = RES / "protocol_frozen.json"
if OUT.exists() and not os.environ.get("REFREEZE"):
    sys.exit("protocol already frozen (REFREEZE=1 only before the holdout is touched)")
cv = json.load(open(RES / "stack_cv.json")); cc = json.load(open(RES / "cascade_cv.json"))
folds = sorted(cv)
if len(folds) != 5:
    sys.exit(f"CV has {len(folds)} folds, expected 5")
mean = lambda m: float(np.mean([cv[f][m]["map50"] for f in folds]))
metas = {m: mean(m) for m in ("meta_LR", "meta_HGB", "meta_RF", "meta_SVM", "meta_MLP", "meta_KNN") if m in cv[folds[0]]}
meta = max(metas, key=metas.get)
heavy = max(("rtdetr_l", "fasterrcnn"), key=lambda h: mean(f"single_{h}"))
tbl = cc["res"][heavy]
mmap = lambda n: float(np.mean([e["map50"] for e in tbl[n]]))
mesc = lambda n: float(np.mean([e["esc"] for e in tbl[n]]))
target = mmap("heavy_only") - 0.01
cands = [(mesc(n), n) for n in tbl if n.startswith("band<") and mmap(n) >= target]
policy, fallback = (min(cands)[1], False) if cands else ("band<0.9", True)
files = [S / "stackcore.py", S / "stack_eval.py", S / "cascade_cv.py", S / "dump_dets.py", S / "holdout_final.py",
         R / "datasets/folds_v3_manifest.csv"]
import torch, ultralytics
P = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), light=cc["light"], heavy=heavy, meta=meta.replace("meta_", ""),
         policy=policy, fallback=fallback, cv_meta_map50=metas, cv_heavy_only=mmap("heavy_only"),
         cv_policy_map50=mmap(policy), cv_policy_esc=mesc(policy),
         rules=["meta = highest CV mean mAP50 among all meta-learners run in CV",
                "heavy = higher CV mean single mAP50 of {rtdetr_l,fasterrcnn}",
                "policy = band<x with lowest mean esc_rate whose CV mAP50 >= heavy_only - 0.01; else fallback band<0.9",
                "holdout evaluated once; image-level escalation to full-image heavy model"],
         sha256={f.name: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files if f.exists()},
         versions=dict(torch=torch.__version__, ultralytics=ultralytics.__version__))
json.dump(P, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in P.items() if k not in ("sha256", "rules")}, indent=1))
