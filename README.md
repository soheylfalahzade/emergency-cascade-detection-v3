<div align="center">

# Emergency-Vehicle-Aware Traffic Signal Control

**Detection, cascade inference and closed-loop signal control with emergency-vehicle priority**

![Phase 1](https://img.shields.io/badge/Phase%201-perception-1f6feb?style=flat-square)
![Phase 2](https://img.shields.io/badge/Phase%202-control%20in%20CARLA-2da44e?style=flat-square)
![Detectors](https://img.shields.io/badge/detectors-YOLOv8n%20%C2%B7%20RT--DETR--l%20%C2%B7%20Faster%20R--CNN-8250df?style=flat-square)
![Simulator](https://img.shields.io/badge/simulator-CARLA%200.9.16-e36209?style=flat-square)
![Evaluation](https://img.shields.io/badge/evaluation-frozen%20protocol%20%2B%20leakage%20audit-d1242f?style=flat-square)

</div>

A traffic signal that gives way to an ambulance first has to *see* the ambulance. This repository follows that chain end to end:
a four-class vehicle detector with a light/heavy cascade (**Phase 1**), and a fuzzy junction controller with emergency arbitration evaluated in closed loop in the CARLA simulator, driven by ground truth, by injected perception errors and by a real detector in the loop (**Phase 2**).

| | Phase 1: perception | Phase 2: control |
|---|---|---|
| **Question** | How accurate and how costly is detection of ambulance, police, firetruck and car, and does stacking or a cascade help? | Does an emergency-aware fuzzy controller beat simple preemption, and what happens when perception is imperfect? |
| **Setting** | 5-fold cross-validation, one holdout, frozen protocol | One CARLA junction, 4 policies, 4 perception regimes |
| **Headline** | RT-DETR-l is the most accurate model (0.9208 holdout mAP50); the cascade keeps most of its accuracy at 11.9 ms instead of 26.7 ms (cross-validation) | No ambulance advantage of fuzzy over simple preemption with the detector in the loop; large gap between detectors trained on real images and CARLA frames |

> [!IMPORTANT]
> **How to read the numbers.** Every number carries a label: **DEV** (used for selection, optimistic), **FROZEN-ONCE** (protocol frozen before the holdout was evaluated, evaluated once), **POST-HOC** (evaluated after the holdout was consumed), **CARLA** (simulator, one junction), **abstract queue model** (never evidence about CARLA or real traffic). Section 3.5 defines them.

## Contents

- [Chapter 3 - Method](#chapter-3---method)
- [Chapter 4 - Results](#chapter-4---results)
- [Chapter 5 - Discussion, summary and future work](#chapter-5---discussion-summary-and-future-work)
- [Repository map and reproduction](#repository-map-and-reproduction)

---

# Chapter 3 - Method

## 3.1 System overview

```mermaid
flowchart LR
  A["Image datasets<br/>merge and 5-fold split"] --> B["Detectors<br/>YOLOv8n, RT-DETR-l, Faster R-CNN"]
  B --> C["Detections<br/>(dumped per image)"]
  C --> D["Ensembles and stacking"]
  C --> E["Light / heavy cascade"]
  E --> F["Perception output<br/>emergency class and confidence"]
  F --> G["Fuzzy junction controller<br/>emergency arbitration"]
  G --> H["CARLA closed loop<br/>one junction"]
  H -->|"queues, waits, ambulance state"| F
```

## 3.2 Data and splits (Phase 1)

- **Classes:** ambulance, police, firetruck, car.
- **Cross-validation:** 5 folds (`datasets/folds_v3`); every fold trains on 28,099 images and validates on 689 images. Selection of checkpoints, meta-learner, heavy model and cascade policy uses these folds only.
- **Holdout:** 381 images (boxes: ambulance 191, police 15, firetruck 221, car 259), evaluated once under a protocol frozen beforehand. Intervals are 95% group-bootstrap intervals (image groups from `datasets/folds_v3_manifest.csv`).
- Datasets are not included in this repository (size and license terms); the preparation scripts are in `datasets/`.

## 3.3 Detectors and training protocol

| Model | Epochs | Patience | Batch | Checkpoint rule |
|---|---|---|---|---|
| YOLOv8n | 60 | 15 | 16 | best validation fitness (mAP50-95) |
| RT-DETR-l | 60 | 15 | 8 | best validation fitness (mAP50-95) |
| Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`), reported numbers | 15 | none | 4 | last epoch (**under-trained**) |
| Faster R-CNN, retraining in progress (`frcnn60/`) | 60 | 15 | 4 | best validation mAP50-95 |

Training arguments of the first two models are stored in `results_v3/train_logs/*/args.yaml`. All methods are scored with the same code: torchmetrics mAP, confidence ≥ 0.02, top 100 detections per image.

> [!WARNING]
> **Faster R-CNN status.** The Faster R-CNN numbers below come from a 15-epoch run that kept the last epoch, while the other two models were trained for up to 60 epochs with best-checkpoint selection. This protocol mismatch makes it incomparable. A retraining under the same protocol is in progress; every ensemble, stacking and cascade number that depends on it is marked † and will be recomputed. The old numbers stay as the "under-trained" baseline.

## 3.4 Ensembles, stacking and the cascade

- **Ensembles over detections:** NMS union, weighted box fusion (WBF), soft average, top-1.
- **Stacking:** meta-learners (LR, HGB, RF, SVM, MLP, KNN) trained on cross-validated detections. The meta-learner is chosen by a rule fixed beforehand (best CV mAP50): `LR`.
- **Cascade:** the light model (YOLOv8n) answers; images whose light-model confidence falls in an uncertainty band are sent whole to the heavy model (RT-DETR-l). The policy is the smallest escalation rate whose CV mAP50 stays within 0.01 of the heavy model alone.

```mermaid
flowchart TD
  I["Image"] --> L["Light model<br/>YOLOv8n"]
  L -->|"confident"| O1["Output: light detections"]
  L -->|"uncertain band"| H["Heavy model<br/>RT-DETR-l"]
  H --> O2["Output: heavy detections"]
```

Frozen protocol (2026-10-28 14:33:48): meta=LR, heavy=rtdetr_l, cascade=band<0.7 (fallback=False)

## 3.5 Evaluation protocol and result labels

| Label | Meaning |
|---|---|
| **DEV** | a choice (checkpoint, threshold, design) was made while looking at the same data; optimistic |
| **FROZEN-ONCE** | rules written to `protocol_frozen.json` before the holdout was touched; holdout evaluated once |
| **POST-HOC** | computed after the holdout had been consumed, or defined after seeing results |
| **CARLA** | measured in the CARLA simulator on one junction; never a real-world measurement |
| **abstract queue model** | output of `phase2_bridge/toy_sim.py`; figures carry the watermark "abstract queue model - not CARLA" |

The holdout is consumed (`holdout_done.flag`); any further evaluation on it is POST-HOC, and a new protocol must be frozen before looking at results.

## 3.6 Control study (Phase 2)

```mermaid
flowchart LR
  S["CARLA world<br/>vehicles and traffic lights"] --> P["Perception<br/>ground truth, injected noise or detector"]
  P --> C["Controller<br/>fixed, preempt, max-pressure, fuzzy"]
  C -->|"phase and timing"| S
  S --> M["Metrics from simulator ground truth"]
```

- **Simulator and junction:** CARLA 0.9.16, map `Town05_Opt`, junction index 14: four approaches (A0 to A3), two lanes each, two signal phases (A0+A2, A1+A3).
- **Scenario:** 160 s per run, 20 s warm-up, 1 s control step, 0.05 s simulation step, background traffic with inflow 0.25, 2 s all-red. One ambulance on approach A0, 25 to 38 m upstream of the stop line, spawned at a random time, identical across policies for the same seed.
- **Controllers:**

| Policy | Idea |
|---|---|
| `fixed` | fixed cycle (30 s green, 3 s yellow) |
| `preempt` | fixed cycle plus simple preemption when an emergency vehicle is detected (confidence ≥ 0.6, track age ≥ 1 s) |
| `maxpressure` | queue-pressure phase selection, green 10 to 60 s |
| `fuzzy` | fuzzy queue pressure plus emergency arbitration that minimises expected weighted delay; three modes (normal, degraded, static) driven by a perception-health signal |

- **Perception regimes:** (1) ground truth; (2) injected noise with swept parameters (not measured); (3) injected noise with rates measured offline; (4) **detector in the loop**: queues and distances come from the simulator, the emergency class from a detector run on the camera frame (hybrid mode; vision-only perception is not built).
- **Metrics (from simulator ground truth):** ambulance stopped time, ordinary-vehicle stopped time per vehicle, maximum wait, phase switches per hour, throughput, and false-preemption time. Ambulance time-to-cross is survivor-biased.
- **Reproducibility defect found and fixed:** the noise generator drew random numbers in the order of the simulator's actor list, which changes between runs. After sorting inputs with a stable key and auditing, noisy results reproduce exactly; earlier noisy results are discarded and not reported.

---

# Chapter 4 - Results

## 4.1 Phase 1: accuracy of detectors, ensembles and stacking (DEV)

<p align="center"><img src="docs/readme_figures/fig_cv_map50.png" alt="Cross-validation mAP50 of detectors, ensembles and stacking" width="860"></p>

RT-DETR-l reaches **0.9078** mAP50 and YOLOv8n **0.8815**; the under-trained Faster R-CNN reaches 0.8424. Ensembles and stacking stay at or below RT-DETR-l.

<details>
<summary>Full cross-validation table (mean ± std over 5 folds)</summary>

| method | mAP50 | mAP50-95 | AP50 police | AP50 ambulance |
|---|---|---|---|---|
| single_yolov8n | 0.8815±0.0203 | 0.7593±0.0155 | 0.9695±0.0283 | 0.9666±0.0202 |
| single_rtdetr_l | 0.9078±0.0172 | 0.7923±0.0173 | 0.9842±0.0186 | 0.9783±0.0171 |
| single_fasterrcnn | 0.8424±0.0333 | 0.6977±0.0284 | 0.9585±0.0381 | 0.9502±0.0193 |
| nms_union | 0.8773±0.0307 | 0.7229±0.0252 | 0.9705±0.0257 | 0.9798±0.0154 |
| wbf | 0.8988±0.0250 | 0.7821±0.0216 | 0.9793±0.0227 | 0.9802±0.0159 |
| top1_ensemblenet_style | 0.8716±0.0296 | 0.7210±0.0247 | 0.9555±0.0240 | 0.9719±0.0156 |
| softavg | 0.9031±0.0224 | 0.7467±0.0175 | 0.9793±0.0227 | 0.9799±0.0157 |
| meta_LR | 0.9064±0.0221 | 0.7406±0.0194 | 0.9694±0.0293 | 0.9778±0.0168 |
| meta_HGB | 0.8933±0.0246 | 0.7283±0.0242 | 0.9327±0.0465 | 0.9772±0.0166 |
| meta_RF | 0.9002±0.0243 | 0.7423±0.0212 | 0.9699±0.0276 | 0.9784±0.0159 |
| meta_SVM | 0.8784±0.0351 | 0.7093±0.0260 | 0.9504±0.0520 | 0.9696±0.0200 |
| meta_MLP | 0.9029±0.0171 | 0.7423±0.0149 | 0.9627±0.0193 | 0.9784±0.0159 |
| meta_KNN | 0.8900±0.0240 | 0.7308±0.0192 | 0.9710±0.0268 | 0.9696±0.0161 |

</details>

## 4.2 Cascade: accuracy versus cost (DEV)

<p align="center"><img src="docs/readme_figures/fig_cascade_tradeoff.png" alt="Cascade accuracy versus inference cost" width="760"></p>

The frozen policy (`band<0.7`) escalates 25% of the images and reaches **0.8981** mAP50 at **11.9 ms** per image, against 5.2 ms for the light model and 26.7 ms for the heavy model alone.

<details>
<summary>Cascade policies (cross-validation)</summary>

| policy | escalation | mAP50 | expected ms/img |
|---|---|---|---|
| light_only | 0.00 | 0.8815±0.0203 | 5.2 |
| heavy_only | 1.00 | 0.9078±0.0172 | 26.7 |
| band<0.4 | 0.15 | 0.8892±0.0142 | 9.3 |
| band<0.5 | 0.19 | 0.8942±0.0165 | 10.4 |
| band<0.6 | 0.22 | 0.8973±0.0170 | 11.1 |
| band<0.7 | 0.25 | 0.8981±0.0165 | 11.9 |
| band<0.8 | 0.28 | 0.8999±0.0159 | 12.6 |
| band<0.9 | 0.35 | 0.9036±0.0189 | 14.7 |

</details>

## 4.3 Model cost

<p align="center"><img src="docs/readme_figures/fig_model_cost.png" alt="Parameters, GFLOPs and latency of the three detectors" width="860"></p>

<details>
<summary>Cost table</summary>

| model | params (M) | GFLOPs | ms/img median | FPS |
|---|---|---|---|---|
| yolov8n | 3.0 | 8.1 | 5.2 | 190.6 |
| rtdetr_l | 32.8 | 109.2 | 26.7 | 37.4 |
| fasterrcnn | 43.3 | 559.9 | 66.6 | 15.0 |

</details>

## 4.4 Holdout, evaluated once (FROZEN-ONCE)

<p align="center"><img src="docs/readme_figures/fig_holdout_map50.png" alt="Holdout mAP50 with 95 percent intervals" width="860"></p>

<p align="center"><img src="docs/readme_figures/fig_holdout_paired.png" alt="Paired holdout differences with 95 percent intervals" width="820"></p>

Paired differences (95% group-bootstrap intervals):

- Cascade - RT-DETR-l: **-0.0215**, 95% interval [-0.0420, -0.0035]
- Cascade - YOLOv8n: **+0.0232**, 95% interval [+0.0043, +0.0451]
- Stacking (LR) - RT-DETR-l: **+0.0004**, 95% interval [-0.0055, +0.0059]
- Soft average - RT-DETR-l: **-0.0055**, 95% interval [-0.0242, +0.0041]
- WBF - RT-DETR-l: **-0.0073**, 95% interval [-0.0277, +0.0029]
- Stacking (LR) - WBF: **+0.0077**, 95% interval [+0.0019, +0.0244]

Reading: stacking brings **no measurable gain** over RT-DETR-l (+0.0004 [-0.0055, +0.0059]); the cascade is **worse than the heavy model** (-0.0215 [-0.0420, -0.0035]) and **better than the light model** (+0.0232 [+0.0043, +0.0451]), so it is a speed/accuracy trade-off, not an accuracy gain.

<details>
<summary>Holdout table</summary>

| method | mAP50 | CI95 (group bootstrap) | AP50 police | police CI95 |
|---|---|---|---|---|
| single_yolov8n | 0.8618 | [0.828, 0.944] | 0.972 | [0.921, 1.000] |
| single_rtdetr_l | 0.9208 | [0.909, 0.964] | 0.986 | [0.960, 1.000] |
| single_fasterrcnn | 0.7921 | - | 0.984 | - |
| nms_union | 0.8931 | [0.858, 0.955] | 0.984 | [0.956, 1.000] |
| wbf | 0.9140 | [0.887, 0.959] | 0.985 | [0.957, 1.000] |
| top1_ensemblenet_style | 0.8694 | - | 0.958 | - |
| softavg | 0.9159 | [0.889, 0.961] | 0.985 | [0.957, 1.000] |
| meta_LR | 0.9200 | [0.906, 0.963] | 0.986 | [0.960, 1.000] |
| cascade_frozen | 0.8917 | [0.868, 0.960] | 0.972 | [0.921, 1.000] |

</details>

The image-level cascade table in `results_v3/image_cascade_holdout_final.json` (emergency-class mAP50 with risk-controlled thresholds) is **POST-HOC**: the holdout had already been used and the emergency metric was defined after seeing results; its parameters were frozen on validation data.

## 4.5 Phase 2: policies without perception error (CARLA)

<p align="center"><img src="phase2_bridge/docs/figures/fig_clean.png" alt="Four policies on one CARLA junction without perception error" width="900"></p>

| Policy | Ambulance stopped (s) | Stopped per ordinary vehicle (s) | Max wait (s) | Switches / h | Ordinary vehicles / min |
|---|---|---|---|---|---|
| fixed | 7.30 | 7.74 | 33.30 | 102.9 | 7.37 |
| preempt | 2.00 | 7.16 | 32.00 | 113.1 | 7.24 |
| maxpressure | 5.80 | **2.73** | **13.50** | 156.9 | 6.64 |
| fuzzy | 2.20 | 4.41 | 31.60 | 100.3 | 6.64 |

*10 seeds; means. Fuzzy minus preempt, paired by seed (95% bootstrap interval): ambulance stopped +0.20 s [+0.00, +0.50]; stopped per ordinary vehicle -2.75 s [-4.45, -1.23].*

## 4.6 Phase 2: detectors on CARLA frames (domain gap)

<p align="center"><img src="phase2_bridge/docs/figures/fig_detection_validation.png" alt="Detector found rates on the CARLA validation capture" width="860"></p>

- A detector trained on real images only (`yolov8n_fold0`) finds few vehicles in CARLA frames (validation capture, conf ≥ 0.5, found rate: ambulance 0.07, police 0.16, firetruck 0.01, car 0.03).
- After fine-tuning on CARLA frames, the validation capture gives ambulance 1.00, police 0.93, firetruck 1.00, car 0.87 (checkpoint selected on it: **DEV**). On the held-out capture (new camera pose and seeds; same junction and vehicle models): ambulance 0.99, police 0.88, firetruck 0.99, car 0.75.

<p align="center"><img src="phase2_bridge/docs/figures/fig_detection_heldout.png" alt="Detector found rates on the held-out CARLA capture" width="860"></p>

## 4.7 Phase 2: injected noise versus detector in the loop (CARLA)

<p align="center"><img src="phase2_bridge/docs/figures/fig_injected_measured_noise.png" alt="Injected measured-rate noise" width="860"></p>

<p align="center"><img src="phase2_bridge/docs/figures/fig_detector_in_loop.png" alt="Detector in the loop" width="860"></p>

- With the CARLA-fine-tuned detector in the loop the ambulance is recognised in nearly all visible frames; **fuzzy shows no advantage over preempt** for the ambulance (+0.5 to +0.6 s, lower interval bound 0.0 s). Ordinary vehicles stop about 2.0 to 2.5 s less per vehicle under fuzzy (interval excludes 0 in all four weather conditions; the four weather conditions share the same traffic scenes, so these are not four independent confirmations).
- With the detector trained on real images only, the ambulance is almost never recognised in the loop.
- False emergency action is about 23 to 75 s per run with injected measured-rate noise and about 0 to 5 s with the detector in the loop. Injected noise is a **pessimistic stress test, not a prediction**; the cause is not tested (hypothesis: injected false alarms persist 1 to 5 s by assumption).

## 4.8 Simulator renders

<p align="center"><img src="phase2_bridge/docs/figures/fig_sim_junction_phases.png" alt="The two signal phases at the junction" width="900"></p>

<p align="center"><img src="phase2_bridge/docs/figures/fig_sim_signal_check.png" alt="Signal-head colours per approach and phase" width="900"></p>

*Row 1: phase 0 (A0 and A2 green); row 2: phase 1 (A1 and A3 green). The mapping between lights and approaches is checked by stop-line distance (mismatch ≤ 0.01 m), by behaviour, by pixel colour counts and by eye.*

The full Phase 2 report (all tables, protocols, per-weather results) is in [`phase2_bridge/README.md`](phase2_bridge/README.md).

---

# Chapter 5 - Discussion, summary and future work

## 5.1 Phase 1

- **The cascade is a cost lever, not an accuracy lever.** It recovers most of the gap between the light and heavy model at roughly half of the heavy model's cost in cross-validation, but on the holdout it stays below the heavy model alone.
- **Stacking adds nothing measurable** on the emergency data: the paired difference to RT-DETR-l is +0.0004 [-0.0055, +0.0059] on the holdout.
- **Faster R-CNN was under-trained** by protocol mismatch (15 epochs and last checkpoint versus 60 epochs, patience 15 and best checkpoint). In the logs of folds 0 and 3, training loss plateaus near 0.097 during epochs 4 to 8 and drops after the learning-rate step in epoch 9, validation loss is lowest at epoch 9 and rises afterwards, and the logs contain no mAP. All Faster R-CNN-dependent results are provisional until the retraining finishes.

## 5.2 Phase 2

- **Priority.** Simple preemption already reaches the floor of about 2 s of ambulance stopped time; the fuzzy controller matches it but does not beat it, with or without perception error.
- **Ordinary traffic.** Max-pressure is clearly best for ordinary traffic in the clean regime; fuzzy improves on preempt (-2.75 s per vehicle) but not on max-pressure.
- **Perception matters more than the controller.** The detector trained on real images fails on CARLA frames, and fine-tuning closes the gap only within the same junction and vehicle models.
- **Method.** An ordering defect in the noise generator made noisy runs irreproducible; finding and fixing it changed which results can be reported.

## 5.3 Threats to validity

- One junction on one map; the held-out CARLA capture uses the same junction and the same vehicle models, so it is not a new environment.
- Weather conditions share the same traffic scene for equal seeds; they are not independent repetitions. Weather-visibility proxy is uncalibrated.
- The fine-tuned detector's validation-set numbers are DEV and optimistic.
- Injected-noise experiments are stress tests; their parameters are not measurements of a real system.
- Car labels in the validation and holdout data are incomplete, so car AP is not a clean metric. `holdout_v3` and `holdout_v4` contain the same 381 image names and 49 label files differ.
- The holdout is consumed. A near-duplicate audit of `holdout_v3` against `folds_v3` has not been run yet; exact-duplicate (md5) overlap is 0 of 381. Police has very few holdout boxes, so any police number needs its interval.
- GPU timing on the laptop is unstable; use it for ordering, not for precise ratios.
- No real-world data: the simulation-to-reality gap is not measured here.

## 5.4 Summary

A heavy detector (RT-DETR-l) is the most accurate model; a light/heavy cascade trades a small accuracy loss for about half of the cost; stacking brings no gain. In closed loop in CARLA, a fuzzy emergency-aware controller matches simple preemption for the ambulance and does not exceed it, perception quality in simulation is dominated by the domain gap, and injected noise gives pessimistic but unvalidated stress tests.

## 5.5 Future work

1. Finish the Faster R-CNN retraining with the shared protocol, then recompute ensembles, stacking, the cascade policy and the holdout tables (POST-HOC), keeping the 15-epoch results as an appendix baseline.
2. Run the near-duplicate audit of the consumed holdout against the training folds and freeze a new protocol before any further holdout evaluation.
3. Extend the CARLA study to several junctions with two-hop message passing (implemented and run only in the abstract queue model so far).
4. Build vision-only perception (queues and distances from images) and calibrate the weather-visibility proxy.
5. Validate on real footage and a second map before making any claim about real traffic.

---

# Repository map and reproduction

| Path | Content |
|---|---|
| `datasets/` | dataset preparation, fold generation and training scripts (no data) |
| `results_v3/` | frozen protocol, final report, holdout results, cross-validation JSON, training logs |
| `stacking/` | detection dumps, ensembles and stacking, cascade, audits, benchmark scripts |
| `figures_pipeline/` | figure scripts and `build_root_readme.py`, which regenerates this README |
| `docs/readme_figures/` | figures in this README (generated from `results_v3/final_report.md`) |
| `phase2_bridge/` | controllers, CARLA bridge, abstract queue model, tests, Phase 2 results and report |
| `frcnn60/` | Faster R-CNN 60-epoch retraining scripts (training in progress) |
| `requirements_frozen.txt` | frozen package versions |

Datasets and model weights are not committed.

```bash
pip install -r requirements_frozen.txt
python figures_pipeline/build_root_readme.py   # rebuilds the figures and this README from results_v3/final_report.md
```

Phase 2 reproduction (CARLA 0.9.16, one run per process, seeds and gates) is described in [`phase2_bridge/README.md`](phase2_bridge/README.md).
