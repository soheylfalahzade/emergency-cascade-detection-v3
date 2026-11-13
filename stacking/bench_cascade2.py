import json, time, subprocess
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
for _ in range(10):
    L.predict(ims[0], **kw); H.predict(ims[0], **kw)


def gpu():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader"]).decode().strip()


def mk(hi):
    esc = []

    def f(im):
        c = L.predict(im, **kw)[0].boxes.conf.cpu().numpy()
        e = bool(((c >= LO) & (c < hi)).any()); esc.append(e)
        if e:
            H.predict(im, **kw)
    return f, esc


systems = {"light": (lambda im: L.predict(im, **kw), None), "heavy": (lambda im: H.predict(im, **kw), None)}
for a, hi in HIS.items():
    systems[f"cascade_a{a}"] = mk(hi)
rng = np.random.RandomState(0); res = {k: [] for k in systems}; clocks = []
for rd in range(5):
    clocks.append(gpu()); print(f"round {rd} gpu: {clocks[-1]}", flush=True)
    for k in rng.permutation(list(systems)):
        fn, esc = systems[k]
        if esc is not None:
            esc.clear()
        ts = []
        for im in ims:
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(im); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        res[k].append(dict(mean=float(ts.mean()), p95=float(np.percentile(ts, 95)), esc=None if esc is None else float(np.mean(esc))))
out = {"n_images": len(ims), "rounds": 5, "gpu_clocks_per_round": clocks,
       "note": "end-to-end per image, batch 1, GPU synchronised, random validation sample (fold 0), order randomised per round"}
for k, v in res.items():
    m = [r["mean"] for r in v]; p = [r["p95"] for r in v]
    out[k] = dict(ms_mean_median_over_rounds=float(np.median(m)), ms_mean_range=[min(m), max(m)], ms_p95_median=float(np.median(p)),
                  escalation_rate=None if v[0]["esc"] is None else float(np.mean([r["esc"] for r in v])))
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in out[k].items()})
json.dump(out, open(R / "image_cascade_bench2.json", "w"), indent=1)
