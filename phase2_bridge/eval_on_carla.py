#!/usr/bin/env python
"""Phase-1 detectors (YOLO / RT-DETR, ultralytics) on CARLA frames vs CARLA ground truth. Run in WSL (env traffic_env).
Label every number 'CARLA' (simulator), never real-world. Matching: IoU>=0.5 class-agnostic for 'found', class of the best match for 'correct'.
GT targets: inst_px>MIN_INST, bbox area>=MIN_AREA, dist<=MAX_DIST. Predictions with IoU<0.3 to every recorded GT vehicle are false positives.
Repeated frames show the same vehicles, so CIs are cluster-bootstrap over (capture, vehicle id).
Usage: python eval_on_carla.py --captures DIR --weights PATH[,PATH...] [--imgsz 640] [--out DIR]"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

CLS = ["ambulance", "police", "firetruck", "car"]      # model class indices 0..3 (Phase 1)
EM = ("ambulance", "police", "firetruck")


def iou_one(a, bs: np.ndarray) -> np.ndarray:
    if len(bs) == 0:
        return np.zeros(0)
    x1, y1 = np.maximum(a[0], bs[:, 0]), np.maximum(a[1], bs[:, 1])
    x2, y2 = np.minimum(a[2], bs[:, 2]), np.minimum(a[3], bs[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def load_frames(cap_root):
    frames = []
    for gt in sorted(glob.glob(os.path.join(cap_root, "*", "gt.jsonl"))):
        d = os.path.dirname(gt)
        for line in open(gt, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["capture"] = os.path.basename(d)
                r["key"] = os.path.join(d, r["file"])
                frames.append(r)
    return frames


def run_model(weights, keys, imgsz, device):
    from ultralytics import RTDETR, YOLO
    M = RTDETR if "rtdetr" in weights.lower() else YOLO
    model = M(weights)
    out = {}
    for i in range(0, len(keys), 16):
        for k, r in zip(keys[i:i + 16], model.predict(keys[i:i + 16], imgsz=imgsz, conf=0.02, verbose=False, device=device)):
            out[k] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
    return out


def evaluate(frames, preds, thr, min_area=400.0, max_dist=100.0, min_inst=0):
    obs, fp = [], {c: 0 for c in CLS}
    for fr in frames:
        b, c, k = preds[fr["key"]]
        m = c >= thr
        b, c, k = b[m], c[m], k[m]
        allgt = np.array([v["bbox"] for v in fr["vehicles"]], dtype=float).reshape(-1, 4)
        for v in fr["vehicles"]:
            area = (v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1])
            if v["inst_px"] <= min_inst or area < min_area or v["dist_m"] > max_dist:
                continue
            ious = iou_one(np.array(v["bbox"], dtype=float), b)
            ok = np.where(ious >= 0.5)[0]
            if len(ok):
                j = ok[np.argmax(ious[ok])]
                found, pc, pconf = True, CLS[k[j]], float(c[j])
            else:
                found, pc, pconf = False, None, 0.0
            obs.append(dict(cls=v["cls"], found=found, pred=pc, conf=pconf, dist=v["dist_m"], area=area,
                            weather=fr["weather"], track=(fr["capture"], v["id"])))
        for i in range(len(b)):
            if len(allgt) == 0 or iou_one(b[i], allgt).max() < 0.3:
                fp[CLS[k[i]]] += 1
    return obs, fp, len(frames)


def boot_ci(obs, fn, n=2000, seed=0):
    tracks = {}
    for o in obs:
        tracks.setdefault(o["track"], []).append(o)
    keys = list(tracks)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        samp = [o for i in rng.integers(0, len(keys), len(keys)) for o in tracks[keys[i]]]
        v = fn(samp)
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else None


def rate(obs, num):
    return (sum(1 for o in obs if num(o)) / len(obs)) if obs else None


def summarise(obs, fp, nfr):
    out = {}
    for c in CLS:
        sub = [o for o in obs if o["cls"] == c]
        ntr = len({o["track"] for o in sub})
        d = dict(n_obs=len(sub), n_tracks=ntr)
        if sub:
            d["found"] = rate(sub, lambda o: o["found"])
            d["found_ci"] = boot_ci(sub, lambda s: rate(s, lambda o: o["found"]))
            d["correct_class"] = rate(sub, lambda o: o["pred"] == c)
            d["correct_ci"] = boot_ci(sub, lambda s: rate(s, lambda o: o["pred"] == c))
            if c in EM:
                d["emerg_any"] = rate(sub, lambda o: o["pred"] in EM)
            d["confusion"] = {p: sum(1 for o in sub if o["pred"] == p) for p in CLS}
            d["confusion"]["missed"] = sum(1 for o in sub if not o["found"])
            if c == "car":
                d["car_as_emergency"] = rate(sub, lambda o: o["pred"] in EM)
                d["car_as_emergency_ci"] = boot_ci(sub, lambda s: rate(s, lambda o: o["pred"] in EM))
        out[c] = d
    out["false_positives_per_frame"] = {c: fp[c] / max(nfr, 1) for c in CLS}
    em = [o for o in obs if o["cls"] in EM]
    near, far = [o for o in em if o["dist"] < 50.0], [o for o in em if o["dist"] >= 50.0]
    fnd = [o for o in em if o["found"]]
    out["emerg_pooled"] = dict(
        n=len(em), n_near=len(near), n_far=len(far), found=rate(em, lambda o: o["found"]),
        found_near=rate(near, lambda o: o["found"]), found_far=rate(far, lambda o: o["found"]),
        as_car_given_found=(sum(1 for o in fnd if o["pred"] == "car") / len(fnd)) if fnd else None)
    out["fp_emerg_per_frame"] = sum(fp[c] for c in EM) / max(nfr, 1)
    out["car_as_emerg_per_frame"] = sum(1 for o in obs if o["cls"] == "car" and o["pred"] in EM) / max(nfr, 1)
    out["n_frames"] = nfr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="carla_eval")
    ap.add_argument("--min-area", type=float, default=400.0)
    ap.add_argument("--max-dist", type=float, default=100.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = load_frames(a.captures)
    print("frames:", len(frames), " captures:", sorted({f["capture"] for f in frames}))
    keys = [f["key"] for f in frames]
    result = dict(source="CARLA", imgsz=a.imgsz, min_area=a.min_area, max_dist=a.max_dist, models={})
    for w in a.weights.split(","):
        tag = os.path.basename(os.path.dirname(os.path.dirname(w))) or os.path.basename(w)
        pk = os.path.join(a.out, "preds_%s_%d.pkl" % (tag, a.imgsz))
        if os.path.exists(pk):
            preds = pickle.load(open(pk, "rb"))
        else:
            preds = run_model(w, keys, a.imgsz, a.device)
            pickle.dump(preds, open(pk, "wb"))
        res = {}
        for thr in (0.25, 0.5, 0.75):
            obs, fp, nfr = evaluate(frames, preds, thr, a.min_area, a.max_dist)
            res["conf>=%.2f" % thr] = summarise(obs, fp, nfr)
            if thr == 0.5:
                bins = {}
                for lo, hi in ((0, 50), (50, 80), (80, 101)):
                    sub = [o for o in obs if o["cls"] in EM and lo <= o["dist"] < hi]
                    bins["%d-%dm" % (lo, hi)] = dict(n=len(sub), found=rate(sub, lambda o: o["found"]),
                                                    emerg_any=rate(sub, lambda o: o["pred"] in EM))
                res["emergency_by_distance@0.5"] = bins
                wb = {}
                for wn in sorted({o["weather"] for o in obs}):
                    sub = [o for o in obs if o["cls"] in EM and o["weather"] == wn]
                    wb[wn] = dict(n=len(sub), found=rate(sub, lambda o: o["found"]),
                                  emerg_any=rate(sub, lambda o: o["pred"] in EM))
                res["emergency_by_weather@0.5"] = wb
                bw = {}
                for wn in sorted({f["weather"] for f in frames}):
                    fw = [f for f in frames if f["weather"] == wn]
                    o2, f2, n2 = evaluate(fw, preds, thr, a.min_area, a.max_dist)
                    bw[wn] = summarise(o2, f2, n2)
                res["by_weather@0.5"] = bw
        result["models"][tag] = res
        print("\n== %s (imgsz %d, CARLA) ==" % (tag, a.imgsz))
        for thr in ("conf>=0.25", "conf>=0.50", "conf>=0.75"):
            r = res[thr]
            print(thr, " frames:", r["n_frames"], " FP/frame:", {k: round(v, 3) for k, v in r["false_positives_per_frame"].items()})
            for c in CLS:
                d = r[c]
                if d["n_obs"]:
                    print("   %-9s n=%4d tracks=%3d found=%.2f ci=%s correct=%.2f ci=%s%s" % (
                        c, d["n_obs"], d["n_tracks"], d["found"],
                        None if d["found_ci"] is None else tuple(round(x, 2) for x in d["found_ci"]), d["correct_class"],
                        None if d["correct_ci"] is None else tuple(round(x, 2) for x in d["correct_ci"]),
                        ("  car->emerg=%.3f" % d["car_as_emergency"]) if c == "car" else ""))
        print("by distance (emergency, conf>=0.5):", res["emergency_by_distance@0.5"])
        print("by weather  (emergency, conf>=0.5):", res["emergency_by_weather@0.5"])
    json.dump(result, open(os.path.join(a.out, "carla_detection_eval.json"), "w"), indent=1)
    print("\nwrote", os.path.join(a.out, "carla_detection_eval.json"))


if __name__ == "__main__":
    main()
