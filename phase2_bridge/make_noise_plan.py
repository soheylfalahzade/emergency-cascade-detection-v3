#!/usr/bin/env python
"""Turn the MEASURED detector behaviour on CARLA frames (carla_detection_eval.json) into per-weather noise parameters for carla_runner.
All values are CARLA-simulator measurements of the Phase-1 detectors, NOT real-world error rates.
Usage: python make_noise_plan.py carla_eval/carla_detection_eval.json [--model TAG] [--thr conf>=0.50] [--fps 1.0] [--out noise_plan.json]"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("eval_json")
ap.add_argument("--model", default="")
ap.add_argument("--thr", default="conf>=0.50")
ap.add_argument("--fps", type=float, default=1.0, help="captured frames per simulated second (capture uses every=20 ticks of 0.05 s = 1.0)")
ap.add_argument("--cam-to-stop", type=float, default=38.5, help="camera -> approach stop line distance (m); camera-distance 50 m maps to stop-line distance 50-cam_to_stop")
ap.add_argument("--near-m", type=float, default=50.0)
ap.add_argument("--min-n", type=int, default=15)
ap.add_argument("--out", default="noise_plan.json")
a = ap.parse_args()

J = json.load(open(a.eval_json))
tag = a.model or sorted(J["models"])[0]
M = J["models"][tag]
if "by_weather@0.5" not in M:
    raise SystemExit("this eval json has no by_weather@0.5 (needs the updated eval_on_carla.py)")
far_m = max(a.near_m - a.cam_to_stop, 5.0)
plan = {"_note": "CARLA-measured detector rates (%s, %s); far_m is an approximate camera->stop-line conversion" % (tag, a.thr)}
for wn, S in M["by_weather@0.5"].items():
    ep = S["emerg_pooled"]
    low = []
    pm = (1.0 - ep["found_near"]) if (ep["found_near"] is not None and ep["n_near"] >= a.min_n) else None
    if pm is None:
        pm = (1.0 - ep["found"]) if ep["found"] is not None else 0.0
        low.append("near-range emergency n=%d < %d: used pooled miss" % (ep["n_near"], a.min_n))
    pmf = 0.0
    if ep["found_far"] is not None and ep["n_far"] >= a.min_n:
        pmf = max((1.0 - ep["found_far"]) - pm, 0.0)
    else:
        low.append("far-range emergency n=%d < %d: extra far miss set to 0" % (ep["n_far"], a.min_n))
    pac = ep["as_car_given_found"] or 0.0
    cs = S.get("car", {})
    if cs.get("n_obs", 0) >= a.min_n and cs.get("found") is not None:
        pmc = 1.0 - cs["found"]
    else:
        pmc = pm
        low.append("car n too small: car miss set equal to emergency miss")
    fpm = (S["fp_emerg_per_frame"] + S["car_as_emerg_per_frame"]) * 60.0 * a.fps
    plan[wn] = dict(p_miss_car=round(pmc, 4), p_miss_emerg=round(pm, 4), p_miss_far=round(pmf, 4), far_m=round(far_m, 1), p_emerg_as_car=round(pac, 4),
                    fp_per_min=round(fpm, 3), n_emergency_obs=ep["n"], n_near=ep["n_near"], n_far=ep["n_far"], caveats=low)
    print("%-13s p_miss_car=%.3f p_miss_emerg=%.3f p_miss_far=%.3f far_m=%.1f p_emerg_as_car=%.3f fp_per_min=%.2f   (emerg obs n=%d near=%d far=%d) %s"
          % (wn, pmc, pm, pmf, far_m, pac, fpm, ep["n"], ep["n_near"], ep["n_far"], low))
json.dump(plan, open(a.out, "w"), indent=1)
print("wrote", a.out)
