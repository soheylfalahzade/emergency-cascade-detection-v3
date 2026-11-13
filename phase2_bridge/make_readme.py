"""Builds README.md for the Phase-2 (CARLA) study. Every number in the tables is computed from the result files at build time.
Run from the phase2_bridge folder with the Windows venv python."""
import glob, json, os, re, shutil, sys
import numpy as np

WEATHERS = ["ClearNight", "ClearNoon", "Fog", "HardRainNoon"]
FIG = "docs/figures"
os.makedirs(FIG, exist_ok=True)


def rd(path):
    if not os.path.exists(path):
        return None
    b = open(path, "rb").read()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16")
    return b.decode("utf-8-sig", errors="replace")


def jl(path):
    t = rd(path)
    return [json.loads(l) for l in t.splitlines() if l.strip()] if t else []


def jsn(path):
    t = rd(path)
    return json.loads(t) if t else None


def mdt(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f2(x):
    return "n/a" if x is None else "%.2f" % x


def ci2(c):
    return "" if not c else " [%.2f, %.2f]" % (c[0], c[1])


def boot(d, n=4000):
    d = np.asarray(d, float)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    m = np.random.default_rng(0).choice(d, (n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def byps(rows):
    o = {}
    for r in rows:
        o.setdefault(r["policy"], {})[r["seed"]] = r
    return o


def pdiff(a, b, k):
    s = sorted(set(a) & set(b))
    d = [a[i][k] - b[i][k] for i in s if a[i].get(k) is not None and b[i].get(k) is not None]
    if not d:
        return "n/a"
    lo, hi = boot(d)
    return "%+.2f [%+.2f, %+.2f]" % (np.mean(d), lo, hi)


def fig(src, name, caption):
    if src and os.path.exists(src):
        shutil.copy(src, os.path.join(FIG, name))
        return "![%s](%s/%s)\n\n*%s*\n" % (name, FIG, name, caption)
    return "*(figure %s not available at build time)*\n" % name


def det_rows(J, tag):
    S = J["models"][tag]["conf>=0.50"]
    rows = []
    for c in ("ambulance", "police", "firetruck", "car"):
        d = S[c]
        if not d["n_obs"]:
            continue
        rows.append([c, d["n_obs"], d["n_tracks"], f2(d["found"]) + ci2(d["found_ci"]), f2(d["correct_class"]) + ci2(d["correct_ci"]),
                     "%.3f" % S["false_positives_per_frame"][c]])
    return rows


def det_sentence(J, tag):
    S = J["models"][tag]["conf>=0.50"]
    return ", ".join("%s %.2f" % (c, S[c]["found"]) for c in ("ambulance", "police", "firetruck", "car") if S[c]["n_obs"])


val = jsn("carla_eval_val/carla_detection_eval.json")
held = jsn("carla_eval_heldout/carla_detection_eval.json")

# ---------------------------------------------------------------- tables
clean = byps(jl("results_lowflow.jsonl"))
CM = [("amb_stopped_s", "ambulance stopped (s)"), ("norm_stopped_s_per_veh", "stopped / ordinary vehicle (s)"), ("max_wait_s", "max wait (s)"),
      ("switches_per_h", "phase switches / h"), ("norm_crossed_per_min", "ordinary vehicles crossed / min")]
rows = []
for p in ("fixed", "preempt", "maxpressure", "fuzzy"):
    if p in clean:
        R = list(clean[p].values())
        rows.append([p, len(R)] + ["%.2f (sd %.2f)" % (np.mean([r[m] for r in R]), np.std([r[m] for r in R], ddof=1)) for m, _ in CM])
clean_tbl = mdt(["policy", "n seeds"] + [h for _, h in CM], rows)
cd = []
if "fuzzy" in clean and "preempt" in clean:
    for m, h in CM[:4]:
        cd.append([h, pdiff(clean["fuzzy"], clean["preempt"], m)])
clean_diff = mdt(["metric (fuzzy - preempt, paired by seed, 95% bootstrap CI)", "difference"], cd)


def plan_tbl(f):
    P = jsn(f)
    if not P:
        return "*(not available)*"
    keys = ["p_miss_car", "p_miss_emerg", "p_miss_far", "far_m", "p_emerg_as_car", "fp_per_min"]
    return mdt(["weather"] + keys, [[w] + [("%.3f" % P[w][k]) if k in P[w] else "" for k in keys] for w in WEATHERS if w in P])


def pair_tbl(prefix_dir, name_fmt):
    rows = []
    for det in ("ft", "p1"):
        for w in WEATHERS:
            R = byps(jl(os.path.join(prefix_dir, name_fmt % (det, w))))
            if "fuzzy" in R and "preempt" in R:
                rows.append([det, w, len(R["fuzzy"]), pdiff(R["fuzzy"], R["preempt"], "amb_stopped_s"), pdiff(R["fuzzy"], R["preempt"], "norm_stopped_s_per_veh"),
                             pdiff(R["fuzzy"], R["preempt"], "max_wait_s"), pdiff(R["fuzzy"], R["preempt"], "false_preempt_s")])
    return mdt(["detector", "weather", "n", "d ambulance stopped (s)", "d stopped / ordinary veh (s)", "d max wait (s)", "d false-preempt (s)"], rows) if rows else "*(not available)*"


def loop_tbl():
    rows = []
    for det in ("ft", "p1"):
        for w in WEATHERS:
            R = jl("v12/results_hyb_%s_%s.jsonl" % (det, w))
            if not R:
                continue
            vis = sum(r["det_amb_vis_frames"] for r in R)
            never = sum(1 for r in R if r["det_amb_vis_frames"] > 0 and r["det_amb_emerg_frames"] == 0)
            lat = [r["det_first_emerg_t"] - r["det_first_vis_t"] for r in R if r["det_first_emerg_t"] is not None and r["det_first_vis_t"] is not None]
            rows.append([det, w, len(R), vis, f2(sum(r["det_amb_found_frames"] for r in R) / max(vis, 1)), f2(sum(r["det_amb_emerg_frames"] for r in R) / max(vis, 1)),
                         f2(sum(r["det_amb_cls_frames"] for r in R) / max(vis, 1)), "%d/%d" % (never, len(R)), "n/a" if not lat else "%.1f" % np.median(lat),
                         "%.2f" % np.mean([r["det_fp_emerg_frames"] for r in R]), "%.1f" % np.mean([r["det_car_as_emerg_n"] for r in R])])
    return mdt(["detector", "weather", "runs", "ambulance-visible frames", "found", "recognised as emergency", "exact class", "runs never recognised",
                "median latency (s)", "false-alarm frames / run", "ordinary vehicles reported as emergency / run"], rows) if rows else "*(not available)*"


def fp_compare():
    rows = []
    for w in WEATHERS:
        A = byps(jl("v11/results_v10_ft_%s.jsonl" % w))
        B = byps(jl("v12/results_hyb_ft_%s.jsonl" % w))
        row = [w]
        for p in ("preempt", "fuzzy"):
            for X in (A, B):
                v = [r["false_preempt_s"] for r in X.get(p, {}).values()]
                row.append("%.1f" % np.mean(v) if v else "n/a")
        rows.append(row)
    return mdt(["weather", "preempt: injected noise", "preempt: detector in loop", "fuzzy: injected noise", "fuzzy: detector in loop"], rows)


def val_vs_held():
    if not (val and held):
        return "*(not available)*"
    rows = []
    for tag in ("carla_ft", "yolov8n_fold0"):
        if tag in val["models"] and tag in held["models"]:
            for c in ("ambulance", "police", "firetruck", "car"):
                a, b = val["models"][tag]["conf>=0.50"][c], held["models"][tag]["conf>=0.50"][c]
                if a["n_obs"] and b["n_obs"]:
                    rows.append([tag, c, f2(a["found"]) + ci2(a["found_ci"]), f2(b["found"]) + ci2(b["found_ci"]), "%+.2f" % (b["found"] - a["found"])])
    return mdt(["model", "class", "validation capture (checkpoint selection set)", "held-out capture", "difference"], rows)


def details(title, path, fence=True):
    t = rd(path)
    if t is None:
        return ""
    return "<details><summary>%s</summary>\n\n```\n%s\n```\n</details>\n" % (title, t.rstrip())


# ---------------------------------------------------------------- figures
fclean = fig("figs_carla_v11/fig_carla_clean.png", "fig_clean.png", "Four policies on one CARLA junction without perception error (10 seeds). Bars: mean; whiskers: 95% bootstrap CI over seeds; dots: seeds.")
fswept = fig("figs_carla_v11/fig_carla_swept_noise.png", "fig_swept_noise.png", "Injected (swept) perception error. The parameters are sweep settings, NOT measured rates.")
fmeas = fig("figs_carla_v11/fig_carla_measured_noise.png", "fig_injected_measured_noise.png", "Perception error injected at the rates measured offline (ft = CARLA-fine-tuned detector, optimistic; p1 = Phase-1 detector). ft: 10 seeds, p1: 5 seeds.")
floop = fig("figs_carla_hybrid/fig_carla_measured_noise.png", "fig_detector_in_loop.png", "Detector in the loop (hybrid perception). Same layout as the previous figure; false-alarm values are about 0 to 5 s here against about 23 to 75 s with injected noise.")
fdetv = fig("figs_carla_v11/fig_carla_detection.png", "fig_detection_validation.png", "Detectors on CARLA frames of the validation capture (conf >= 0.5). carla_ft was selected on this set: optimistic.")
fdeth = fig("figs_heldout/fig_carla_detection.png", "fig_detection_heldout.png", "Detectors on CARLA frames of the held-out capture (new camera pose, new seeds, conf >= 0.5).")
fcond = fig("figs_sim/fig_heldout_by_condition.png", "fig_heldout_by_condition.png", "Held-out capture: emergency vehicles reported as an emergency class, by camera distance and by weather. Frames of one vehicle are not independent.")
fsimf = fig("figs_sim/fig_sim_heldout_frames.png", "fig_sim_heldout_frames.png", "Held-out CARLA frames: ground truth (left) and detections of the two detectors (conf >= 0.5). Frame selection rule is fixed in advance (see make_sim_images.py).")
fsig = fig("figs_sim/fig_sim_signal_heads.png", "fig_sim_signal_heads.png", "View along each approach (camera 22 m before the stop line, 4.5 m high) in phase 0 (top row) and phase 1 (bottom row). The ring marks the light actor assigned to the approach; the colour of the light heads comes from the render, the text gives the API state.")
fchk = fig("figs_sim/fig_sim_signal_check.png", "fig_sim_signal_check.png", "Signal-head check: for each approach, a crop around the pixels that change between the phase-0 and phase-1 renders (the signal heads visible from that approach), phase 0 on top, phase 1 below. If the mapping is right, the heads are green when the approach is green and red otherwise.")
fjun = fig("figs_sim/fig_sim_junction_phases.png", "fig_sim_junction_phases.png", "Top-down view of the junction in signal phase 0 and 1 (layout only; labels are overlays at the stop lines of A1 and A3, the stop lines of A0 and A2 are outside the frame, and the light heads are too small to read).")

# ---------------------------------------------------------------- text
T = []
A = T.append
A("""# Signal control with emergency-vehicle priority: CARLA simulation study (Phase 2)

This folder holds the closed-loop simulation study of an adaptive traffic-signal controller (a fuzzy controller with emergency-vehicle arbitration) on one junction of the CARLA simulator, with perception errors that are injected, measured, or produced by a real detector in the loop.

**All numbers in this document come from the CARLA simulator (one junction, Town05_Opt, junction index 14). They are not real-world measurements.** Numbers from the earlier abstract model (`toy_sim.py`) are not reported here.

## 1. Scope, labels and status

| Label | Meaning |
|---|---|
| CARLA | measured in the CARLA simulator, one junction |
| DEV | a choice (checkpoint, threshold, design) was made while looking at the same data: optimistic |
| injected noise | perception error added to ground truth with sweep parameters or with rates measured offline |
| detector in loop (hybrid) | an actual detector runs on a camera frame inside the control loop; parts of the pipeline stay oracle (Section 3.5) |
| survivor-biased | `amb_time_to_cross_s` is defined only for runs in which the ambulance crossed |

Status: simulation results are complete for the single junction. The mapping between traffic lights and approach labels is checked by stop-line distance (mismatch <= 0.01 m) and by behaviour (an ambulance on A0 stops behind the red of A0); the by-eye check uses the renders in Section 4.7 (its status is stated there).

## 2. Summary of findings
""")
A("1. **Phase-1 detector on CARLA frames (domain gap).** A detector trained on real images only (`yolov8n_fold0`) finds few vehicles in CARLA frames (validation capture, conf >= 0.5, found rate: %s)." % (det_sentence(val, "yolov8n_fold0") if val and "yolov8n_fold0" in val["models"] else "n/a"))
A("2. **Detector fine-tuned on CARLA frames (`carla_ft`).** On the validation capture (checkpoint selected on it, DEV): %s. On the held-out capture (new camera pose and seeds; same junction and same vehicle models): %s. Section 4.3 gives the comparison; it does not show real-world performance." % (
    det_sentence(val, "carla_ft") if val and "carla_ft" in val["models"] else "n/a", det_sentence(held, "carla_ft") if held and "carla_ft" in held["models"] else "n/a (not run)"))
A("3. **Controller comparison without perception error.** `fuzzy` and the simple `preempt` baseline give the ambulance the same stopped time (floor of about 2 s); `maxpressure` is best for ordinary traffic. Table in Section 4.1.")
A("4. **Detector in the loop.** With the CARLA-fine-tuned detector the ambulance is recognised in nearly all visible frames, and fuzzy shows no advantage over preempt for the ambulance (difference +0.5 to +0.6 s, lower CI bound 0.0 s). Ordinary vehicles stop about 2.0 to 2.5 s less per vehicle under fuzzy (CI excludes 0 in all four weathers, but the four weathers share the same traffic scenes, so these are not four independent confirmations). With the Phase-1 detector the ambulance is almost never recognised (Section 4.4).")
A("5. **Injected noise is a pessimistic stress test, not a prediction.** The false emergency action (`false_preempt_s`) is about 23 to 75 s per run with injected measured-rate noise and about 0 to 5 s with the detector in the loop (Section 4.5). The cause is not tested (hypothesis: the injected false alarms persist 1 to 5 s by assumption).")
A("6. **Reproducibility defect found and fixed.** Noisy runs were not reproducible because the noise generator drew random numbers in the order of the simulator's actor list, which changes between runs (Section 3.7). All noisy results reported here were re-run after the fix and audited.")
A("""
## 3. Method

### 3.1 Simulator, junction and software

* CARLA 0.9.16 (Windows), synchronous mode, fixed step 0.05 s; Python 3.12.8, `carla` 0.9.16 wheel, numpy 2.5.4, OpenCV 5.0.0.93. Detectors: ultralytics 8.4.150 in WSL (conda env `traffic_env`). GPU: laptop RTX 5060 (Blackwell).
* Junction: Town05_Opt, junction index 14, centre near (30, 0), four approaches A0 (yaw 0), A1 (yaw 90), A2 (yaw -180), A3 (yaw -90), two lanes each, four traffic lights, two phases (phase 0 = A0 and A2, phase 1 = A1 and A3). Geometry and phases are in `junction_spec_town05_idx14.json`.
* Light actor ids change after `reload_world`; each approach's light is re-found by the distance of its stop waypoints to the specified stop line (`resolve_lights`).

### 3.2 Controllers

* `fixed`: green 30 s, yellow 3 s, two phases.
* `preempt`: `fixed` plus a simple emergency pre-emption (emergency track with confidence >= 0.6 and track age >= 1 s).
* `maxpressure`: queue-pressure controller, minimum green 10 s, maximum 60 s.
* `fuzzy` (`junction_controller.py`): fuzzy phase pressure from queue, waiting time and downstream state; emergency arbitration by the plan with the minimum expected weighted delay; three modes (normal, degraded, static fixed-time) chosen from a smoothed perception-health signal with hysteresis. Defaults (`junction_model.py`): minimum green 10 s, maximum 60 s, yellow 3 s, emergency minimum green 4 s, confirm confidence 0.6, minimum track age 1.0 s. In every run reported here the health signal is constant 1.0 unless stated, so the degraded and static modes are not exercised (Section 4.6).
* All policies use an all-red interval of 2 s after yellow.

### 3.3 Scenario and metrics

* Duration 160 s, warm-up 20 s, control step 1 s. Initial traffic 20 vehicles (Traffic Manager), continuous inflow 0.25 vehicles/s, all-red 2 s. Higher inflow (0.5) produced junction deadlock in some runs; saturated-regime files are kept but not used for conclusions.
* One ambulance per run on approach A0, straight route, no lane change, spawned 25 to 38 m upstream of the stop line, speed 0 to 40% below the limit; spawn time = 40 s + U(0, 66) s from `seed + 12345`, identical across policies for a given seed.
* Metrics come from CARLA ground truth, not from the controller's noisy observation: ambulance stopped time (`amb_stopped_s`), time to cross (`amb_time_to_cross_s`, survivor-biased), stopped time per ordinary vehicle, maximum continuous wait, phase switches per hour, ordinary vehicles crossed per minute, and `false_preempt_s` (seconds in emergency mode while the ground-truth ambulance is not in the approach, with the controller's clear-grace).
* Every run is a separate process (several runs in one process were unstable). Paired differences use the seed as the pairing unit with a 95% percentile bootstrap (4000 resamples). Seeds: 10 for clean and for the fine-tuned detector, 5 for the Phase-1 detector.

### 3.4 Perception regimes

1. **No error:** observation = ground truth.
2. **Injected swept noise** (`perception_noise.py`): per vehicle and step a miss probability (optionally larger beyond a distance, and separate for emergency classes), an emergency vehicle reported as `car`, and false emergency reports (Poisson per lane per minute). Assumptions that were not measured: false-alarm confidence U(0.5, 0.9), false-alarm lifetime U(1, 5) s; true detections have confidence N(0.9, 0.05) (N(0.6, 0.05) when reported as car). Settings: mild (miss 0.1, emergency-as-car 0.1, 0.5 false alarms/min), severe (0.3, 0.3, 2.0/min).
3. **Injected measured noise:** the same model with parameters measured offline for a detector on CARLA frames, per weather (Section 3.6, tables below).
4. **Detector in the loop (hybrid):** Section 3.5.

### 3.5 Detector in the loop (hybrid perception)

* Queue, waiting time, distance and speed come from the CARLA API (as in regime 1). The emergency class evidence comes from a detector run on an RGB frame (1280 x 720) of a camera mounted at the stop line of the opposite approach (height 16 m, pitch -28 degrees, FOV 70), looking at approach A0 (`vision_bridge.py`).
* The detector runs in WSL on CPU (image size 960); the simulator sends raw frames over a local socket (`det_server.py`). Detections with confidence >= 0.5 (fixed in advance) are used.
* **Oracle parts:** association of detections to vehicles (projected 3D box IoU >= 0.5), distance and speed of detected vehicles, lane of true vehicles. A vehicle reported as `car` still counts for the queue. Emergency-class detections with IoU < 0.3 to every vehicle (for example parked props, which are not actors) are false alarms; they are attributed to the target approach and tracked by quantised box position.
* Not covered: emergency vehicles on other approaches (one camera), occlusion handling in the statistics (no instance-mask filter online), vision-only estimation of queue, distance and speed.
* Protocol written before results: `v12/PROTOCOL.txt`. Gates: three identical repeats per policy; with-camera and without-camera runs of the `fixed` policy gave identical traffic metrics (the camera does not perturb the traffic).

### 3.6 Offline detector evaluation on CARLA frames

* Capture (`carla_capture.py`): fixed-time signal (green 20 s, yellow 3 s, all-red 2 s), one frame per second, ground-truth boxes from projected 3D boxes, visibility from an instance-segmentation camera, four weathers (ClearNoon, HardRainNoon, ClearNight, Fog with fog density 70).
* Sets: training capture (4 weathers, seeds 0 to 4, 1700 frames, camera 15 m / -25 / FOV 60); validation capture (seeds 10 to 12, 1020 frames, 16 m / -28 / FOV 70) used for checkpoint selection; **held-out capture** (seeds 20 to 22, camera 13 m / -22 / FOV 80, different emergency spawn spacing and inflow), never used for training or selection.
* Fine-tuning (`carla_ft`): from `yolov8n_fold0` (Phase 1), 40 epochs, patience 15, image size 960, batch 8, first 10 layers frozen, validation mAP50 0.941 on the validation capture.
* Matching (`eval_on_carla.py`): targets have instance pixels > 0, box area >= 400 px, distance <= 100 m; "found" = IoU >= 0.5 regardless of class; "correct class" = class of the best match; a prediction with IoU < 0.3 to every ground-truth vehicle is a false positive. Confidence intervals are cluster bootstraps over (capture, vehicle id).
* Parked cars in the map are props, not actors: they have no ground truth, so detections on them count as false positives.
* Noise parameters measured on the validation capture (`make_noise_plan.py`): miss rate of ordinary cars, miss rate of emergency vehicles within 50 m of the camera, extra miss rate beyond, emergency-as-car rate, and false emergency alarms per minute (including ordinary cars reported as emergency).

Measured parameters, fine-tuned detector (`noise_plan_ft.json`):

""" + plan_tbl("noise_plan_ft.json") + """

Measured parameters, Phase-1 detector (`noise_plan_phase1.json`):

""" + plan_tbl("noise_plan_phase1.json") + """

### 3.7 Reproducibility and a defect that was found

* Each run is a separate process; the CARLA server is restarted every 10 runs and after any failure by a watchdog driver; a failed run is repeated once.
* **Defect:** with injected noise, the random stream of `PerceptionNoise` was consumed in the order of `world.get_actors()`, which is not stable across runs. A different order shifted the stream, changed the confidence values of the same ambulance, and flipped a controller decision at one time step. A stored result (Phase-1 detector, ClearNoon, seed 0, fuzzy) was 49 s of ambulance stopped time in the stored row and 2 s in eleven re-runs.
* **Fix:** the vehicles are sorted by (lane, distance rounded to 0.01 m, class) before the noise step (`carla_adapter.py`, marker `PATCH_SORTED_NOISE_INPUT`). Results from before the fix are not reproducible and are not used.
* **Checks after the fix:** three identical repeats per policy (all fields except wall time); a re-run audit of 8 stored rows (2 policies x 4 conditions), all identical. Reports: `v11/gate_check.txt`, `v11/audit_report.txt`, `v12/gate_check.txt`.

## 4. Results
""")
A("### 4.1 Policies without perception error (CARLA, clean)\n\n" + fclean + "\n" + clean_tbl + "\n\n" + clean_diff + "\n\nThe clean-regime runs do not use the noise path, so the ordering fix does not apply to them.\n")
A("### 4.2 Injected swept noise (CARLA; parameters are not measured)\n\n" + fswept + "\nTable of means and CIs: `v11/swept_summaries.txt`.\n")
A("### 4.3 Detectors on CARLA frames\n\nValidation capture (DEV for `carla_ft`):\n\n" + (mdt(["class", "n observations", "n tracks", "found [95% CI]", "correct class [95% CI]", "false positives / frame"], det_rows(val, "carla_ft")) if val and "carla_ft" in val["models"] else "n/a")
  + "\n\n(Phase-1 `yolov8n_fold0`, same set:)\n\n" + (mdt(["class", "n observations", "n tracks", "found [95% CI]", "correct class [95% CI]", "false positives / frame"], det_rows(val, "yolov8n_fold0")) if val and "yolov8n_fold0" in val["models"] else "n/a") + "\n\n" + fdetv)
if held:
    A("Held-out capture (new camera pose and seeds; conf >= 0.5):\n\n" + "\n\n".join("`%s`:\n\n%s" % (t, mdt(["class", "n observations", "n tracks", "found [95% CI]", "correct class [95% CI]", "false positives / frame"], det_rows(held, t))) for t in held["models"])
      + "\n\nValidation versus held-out found rates:\n\n" + val_vs_held() + "\n\n" + fdeth + "\n" + fcond + "\n" + fsimf
      + "Numbers only; frames of one vehicle are correlated and CARLA has one 3D model per emergency class, so held-out here means a new camera pose, new seeds and new spawn spacing on the same junction, not a new environment.\n")
else:
    A("*(held-out evaluation not available at build time)*\n")
A("### 4.4 Injected measured noise (v11) and detector in the loop (v12)\n\n" + fmeas + "\nDifference fuzzy - preempt, injected measured noise (paired by seed, 95% CI):\n\n" + pair_tbl("v11", "results_v10_%s_%s.jsonl") + "\n\n" + floop
  + "\nDetector performance inside the loop (frames in which the ambulance is visible to the camera, after warm-up; no occlusion filter):\n\n" + loop_tbl()
  + "\n\nDifference fuzzy - preempt, detector in the loop (paired by seed, 95% CI):\n\n" + pair_tbl("v12", "results_hyb_%s_%s.jsonl") + "\n\nWith the Phase-1 detector in the loop the ambulance is mostly not recognised, so both policies run mostly on queue information; the weathers then give nearly identical closed-loop results.\n")
A("### 4.5 Injected noise versus detector in the loop\n\nMean `false_preempt_s` per run, fine-tuned detector (10 seeds):\n\n" + fp_compare() + "\n\nInjected noise gives about 23 to 75 s of emergency action without an ambulance; the detector in the loop gives about 0 to 5 s. In the loop the controller also received ordinary vehicles reported as emergency (table above), so the difference does not come from a lack of misclassifications. The cause is not tested.\n")
A("### 4.6 Weather-visibility proxy for the fuzzy controller\n\nThe controller can take a perception-health signal; here it was a weather-based proxy (`visibility_from_weather`, uncalibrated). Comparison fuzzy with proxy versus fuzzy without proxy, Phase-1 detector noise, 5 seeds (`v11/table_v11.txt`, last four blocks): the results were identical with and without the proxy in ClearNoon and Fog, and they changed in ClearNight and HardRainNoon with wide intervals. The controller modes were not analysed per run. These are effects of an uncalibrated heuristic and are not reported as a perception mechanism.\n")
A("### 4.7 Simulator images\n\n" + fjun + "\n" + fchk + "\n" + details("Signal-head colour check (pixels that change between the phase-0 and phase-1 renders)", "figs_sim/signal_check.txt") + "\nStatus of the by-eye check of the light-to-approach mapping (in phase 0 the light heads next to A0 and A2 are green; in phase 1 those next to A1 and A3): **by-eye confirmation pending (automated colour check above)**.\n")
A("""## 5. Discussion

**What the data support**

* A detector trained on real images does not transfer to CARLA frames (domain gap), in the offline evaluation and in the loop. This says nothing about real-world performance.
* With a detector fine-tuned on CARLA frames the ambulance is recognised in nearly all visible frames in the loop, and the fuzzy controller and the simple pre-emption baseline give the ambulance the same stopped time within the resolution of 10 seeds. The ambulance stopped time has a floor of about 2 s in this scenario (interpretation: the ambulance starts from standstill).
* The fuzzy controller reduces the stopped time of ordinary vehicles by about 2 to 2.5 s per vehicle compared with the simple baseline when the detector is in the loop. It does not reduce false emergency action.
* Results depend on the traffic regime (inflow) and the all-red interval; deadlock appeared at inflow 0.5.

**What the data do not support**

* "The fuzzy controller is safer against false alarms", "better than simple pre-emption for the ambulance", real-world error rates, generalisation to other junctions, cameras or vehicle models, "no leakage".
* Any claim from `amb_time_to_cross_s` without the survivor-bias caveat, or from the weather proxy (uncalibrated).

**Threats to validity**

1. One junction, one ambulance model and one model per other emergency class; the detector sees the same appearance in training and test.
2. `carla_ft` was selected on the validation capture; the held-out capture changes camera pose, seeds and spawn spacing only.
3. The four weathers share the same traffic scenes (same seeds), so effects across weathers are not independent replications; Phase-1 runs have 5 seeds.
4. Hybrid perception keeps association, distance and speed as ground truth, covers one approach, and has no online occlusion filter in its statistics.
5. Injected-noise parameters include assumptions (false-alarm confidence and lifetime) that were not measured.
6. CPU inference in the loop differs numerically from the GPU used offline.
7. Metrics are computed from simulator ground truth; vehicle dynamics and driver behaviour are those of the CARLA Traffic Manager.

## 6. Summary and future work

**Summary.** On one CARLA junction, a fuzzy controller with emergency arbitration was compared with fixed-time, simple pre-emption and max-pressure controllers under four perception regimes. Without perception error and with a good detector, the controllers are indistinguishable for the ambulance; the fuzzy controller improves ordinary-traffic stopped time. Injected noise overstates false emergency action compared with a real detector in the loop. A defect in the noise stream was found, fixed and audited.

**Future work**

1. Held-out junction and held-out camera layouts; more than one model per emergency class if CARLA assets allow.
2. Vision-only perception: tracking, queue, distance and speed estimated from the image; cameras on all approaches.
3. A measured perception-health signal (detector confidence statistics) to replace the weather proxy and exercise the degraded and static modes.
4. Persistence and confidence distributions of detector false alarms measured from cached predictions, replacing the assumed distributions in the noise model.
5. More seeds, independent traffic scenes per weather, and an occlusion-aware online evaluation.
6. Multi-junction corridor with message passing (implemented in the abstract model; not run in CARLA).
7. Evaluation on real-image data for the detector side is a separate study.

## 7. Repository layout and reproduction

| Path | Content |
|---|---|
| `junction_spec_town05_idx14.json` | junction geometry, lights, phases |
| `carla_runner.py` | one policy and one seed per process; JSON line output |
| `carla_adapter.py`, `junction_controller.py`, `junction_model.py`, `fuzzy_core.py`, `perception_noise.py` | adapter, controller, data types, noise model |
| `carla_capture.py`, `eval_on_carla.py`, `make_noise_plan.py`, `export_yolo_dataset.py` | frame capture, offline detector evaluation, measured noise parameters, dataset export |
| `vision_bridge.py`, `det_server.py` | detector-in-loop client and the WSL detector server |
| `make_figs_carla.py`, `make_sim_images.py`, `snap_junction.py`, `make_readme.py` | figures and this document |
| `v11/`, `v12/`, `results_clean/`, `eval_json/`, `noise_plan_*.json` | result files, gate and audit reports, evaluation JSON, measured noise parameters |
| `docs/figures/` | figures used here |

Reproduction (PowerShell on Windows; WSL for detectors and figures): start CARLA (`CarlaUE4.exe -windowed`), then `python carla_runner.py --policies fuzzy --seeds 0 --weather ClearNoon --inflow 0.25 --n-traffic 20 --all-red 2 --out out.jsonl`; for the detector in the loop start `det_server.py` in WSL and add `--hybrid-port <port>`. Result tables: `table_v10.py`, `analyze_v12.py`. Every run must be its own process; restart CARLA regularly (the drivers `run_v11.ps1`, `run_v12.ps1` do both).

## Appendix: raw tables
""")
A(details("v11 gate check (determinism)", "v11/gate_check.txt") + details("v11 audit (re-run of stored rows)", "v11/audit_report.txt") + details("v11 tables (injected noise)", "v11/table_v11.txt")
  + details("v12 gate check", "v12/gate_check.txt") + details("v12 analysis (detector in loop)", "v12/analysis_v12.txt") + details("v12 absolute means", "v12/abs_means.txt")
  + details("v12 protocol (written before results)", "v12/PROTOCOL.txt"))
txt = "\n".join(T)
open("README.md", "w", encoding="utf-8", newline="\n").write(txt)
print("README.md written: %d lines, %d characters" % (txt.count("\n"), len(txt)))
