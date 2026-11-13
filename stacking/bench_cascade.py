import json, time
import numpy as np, torch, cv2
from ultralytics import YOLO, RTDETR
import crop_cascade as cc

R = cc.RES
P = json.load(open(R / "image_cascade_protocol_frozen.json")); LO = P["lo"]
HIS = {a: h for a, h in P["hi"].items() if h is not None}
allp = sorted(p for p in cc.img_dir(0, "val").iterdir() if p.suffix.lower() in cc.EXT)
paths = [allp[j] for j in np.random.RandomState(0).permutation(len(allp))[:300]]
ims = [cv2.imread(str(p)) for p in paths]
L = YOLO(str(R / f"{cc.LIGHT}_fold0/weights/best.pt")); H = RTDETR(str(R / f"{cc.HEAVY}_fold0/weights/best.pt"))
kw = dict(conf=cc.MINC, device=cc.DEV, verbose=False, imgsz=640)
for _ in range(5):
    L.predict(ims[0], **kw); H.predict(ims[0], **kw)


def timed(fn):
    ts = []
    for im in ims:
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(im); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def stats(ts, esc=None):
    d = dict(ms_mean=float(ts.mean()), ms_median=float(np.median(ts)), ms_p95=float(np.percentile(ts, 95)))
    if esc is not None:
        d["escalation_rate"] = float(np.mean(esc))
    return d


out = {"n_images": len(ims), "note": "end-to-end per image, batch 1, GPU synchronised, random sample of validation images (fold 0), includes pre/post-processing, excludes disk read"}
out["light"] = stats(timed(lambda im: L.predict(im, **kw)))
out["heavy"] = stats(timed(lambda im: H.predict(im, **kw)))
print("light", out["light"]); print("heavy", out["heavy"], flush=True)
for a, hi in HIS.items():
    esc = []

    def casc(im, hi=hi, esc=esc):
        c = L.predict(im, **kw)[0].boxes.conf.cpu().numpy()
        e = bool(((c >= LO) & (c < hi)).any()); esc.append(e)
        if e:
            H.predict(im, **kw)
    out[f"cascade_a{a}"] = stats(timed(casc), esc)
    print(a, hi, out[f"cascade_a{a}"], flush=True)
json.dump(out, open(R / "image_cascade_bench.json", "w"), indent=1)
