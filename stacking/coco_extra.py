"""Post-hoc: paired comparisons against the strongest single model on COCO test (added after the one-time evaluation; no parameter changed)."""
import json
import numpy as np
import coco_eval as ce
from stackcore import *
from stack_eval import cap, wbf_fuse

R, _ = ce.load_rows(); dev, test = R["dev"], R["test"]
bd = build_split(dev, ce.MODELS, False, 0.55, ce.MINC)
clf = make_meta(ce.META).fit(bd["X"], bd["y"])
n = len(test[ce.LIGHT]); GT = [(r["gt_boxes"], r["gt_cls"]) for r in test[ce.LIGHT]]
out = {"single_" + m: [ce.cf(r) for r in test[m]] for m in ce.MODELS}
bt = build_split(test, ce.MODELS, False, 0.55, ce.MINC)
out["softavg"] = [fuse_softavg(cl) for cl in bt["cl"]]
out["meta_" + ce.META] = per_image(bt, n, proba5(clf, bt["X"]))
out["wbf"] = cap([wbf_fuse(test, ce.MODELS, i, ce.MINC) for i in range(n)])
out["nms_union"] = [ce.nms_union(test, i) for i in range(n)]
pt = {k: ce.map50(v, GT) for k, v in out.items()}; print({k: round(v, 4) for k, v in pt.items()}, flush=True)
rng = np.random.RandomState(5); B = {k: [] for k in out}
for b in range(100):
    idx = rng.randint(0, n, n); g = [GT[i] for i in idx]
    for k in out:
        B[k].append(ce.map50([out[k][i] for i in idx], g))
    if b % 10 == 0:
        print(f"bootstrap {b}/100", flush=True)
B = {k: np.array(v) for k, v in B.items()}; ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
M = "meta_" + ce.META
pairs = [("single_fasterrcnn", "single_rtdetr_l"), ("softavg", "single_fasterrcnn"), (M, "single_fasterrcnn"), ("wbf", "single_fasterrcnn"),
         ("nms_union", "single_fasterrcnn"), ("softavg", M), ("softavg", "wbf")]
res = {f"{a}-{b}": dict(mean=float((B[a] - B[b]).mean()), ci95=ci(B[a] - B[b])) for a, b in pairs}
res["map50_ci95"] = {k: ci(v) for k, v in B.items()}; res["point"] = pt
json.dump(res, open(ce.OUT / "coco_extra.json", "w"), indent=1)
for k, v in res.items():
    if "-" in k:
        print(f"{k:<40} {v['mean']:+.4f} {v['ci95']}")
