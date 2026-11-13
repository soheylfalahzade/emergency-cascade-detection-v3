# Emergency-vehicle detection and adaptive signal control

Two connected studies:

1. **Phase 1 - perception.** Four-class detector (ambulance, police, firetruck, car): YOLOv8n, RT-DETR-l and Faster R-CNN, stacking/ensembles, and a light/heavy cascade, with a frozen evaluation protocol and a leakage audit.
2. **Phase 2 - control.** A fuzzy junction controller with emergency arbitration, evaluated in an abstract queue model and in closed loop in CARLA (one junction), with the Phase 1 style detector in the loop. Details: [`phase2_bridge/README.md`](phase2_bridge/README.md).

## Label conventions

Every number in this repository carries one of these labels.

| Label | Meaning |
|---|---|
| DEV | Used for model or policy selection (cross-validation, validation sets). Optimistic. |
| FROZEN-ONCE | Protocol written and frozen before the holdout was evaluated; holdout evaluated once. |
| POST-HOC | Evaluated after the holdout had already been consumed, or defined after seeing results. |
| CARLA | Measured in the CARLA simulator (one junction, one map). |
| abstract queue model | Output of `phase2_bridge/toy_sim.py`; never evidence about CARLA or real traffic. Figures carry the watermark "abstract queue model - not CARLA". |

## Phase 1: perception

### Method

- Classes: ambulance, police, firetruck, car. 5-fold cross-validation on `datasets/folds_v3` (per fold: 28,099 train / 689 validation images).
- YOLOv8n and RT-DETR-l (Ultralytics): 60 epochs, patience 15, `imgsz 640`, `seed 0`, deterministic, SGD/auto optimizer settings in `results_v3/train_logs/*/args.yaml`; batch 16 and 8. The checkpoint `best.pt` is selected by Ultralytics fitness, i.e. mAP50-95 on the validation split.
- Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`): the numbers in `results_v3/` come from an earlier **15-epoch run that used the last epoch** (`datasets/train_fasterrcnn_v3.py`). They are labelled "under-trained" below. A retraining that follows the same protocol as the other two models (60 epochs, patience 15, best epoch by validation mAP50-95) is in progress; scripts are in `frcnn60/`.
- Ensembles and stacking over detections: NMS-union, WBF, soft average, top-1, and meta-learners (LR, HGB, RF, SVM, MLP, KNN). Rules for choosing the meta-learner, the heavy model and the cascade policy were fixed in `results_v3/protocol_frozen.json` before the holdout was evaluated.
- Cascade: a light model (YOLOv8n) answers when confident, otherwise the image is escalated to the heavy model (RT-DETR-l); the escalation band is chosen on cross-validation.

### Results

Cross-validation mAP50, mean over 5 folds (DEV). Faster R-CNN and anything that uses it are from the under-trained 15-epoch run.

| Method | mAP50 |
|---|---|
| YOLOv8n | 0.8815 |
| RT-DETR-l | 0.9078 |
| Faster R-CNN (under-trained, 15 epochs) | 0.8424 |
| WBF | 0.8988 |
| Soft average | 0.9031 |
| Stacking, LR meta-learner (uses Faster R-CNN) | 0.9064 |
| Cascade, band < 0.7 (escalation 0.25) | 0.8981 |

Cost (laptop RTX 5060, 100 images, fold 0; timing on this GPU is unstable): YOLOv8n 5.2 ms/image, RT-DETR-l 26.7 ms/image, Faster R-CNN 66.6 ms/image. Cascade, band < 0.7: 11.9 ms/image.

Holdout, 381 images, mean of the 5 fold models (FROZEN-ONCE, `results_v3/holdout_final.json`), mAP50:

| Method | mAP50 |
|---|---|
| YOLOv8n | 0.8618 |
| RT-DETR-l | 0.9208 |
| Faster R-CNN (under-trained) | 0.7921 |
| NMS-union | 0.8931 |
| WBF | 0.9140 |
| Soft average | 0.9159 |
| Stacking, LR (uses under-trained Faster R-CNN) | 0.9200 |
| Cascade (frozen policy) | 0.8917 |

Paired differences with 95% group-bootstrap intervals (FROZEN-ONCE):

- Stacking minus RT-DETR-l: +0.0004 [-0.0055, +0.0059]. No gain over the heavy model alone.
- Cascade minus heavy-only: -0.0215 [-0.0420, -0.0035]. The cascade is worse than the heavy model.
- Cascade minus light-only: +0.0232 [+0.0043, +0.0451].

The image-level cascade table in `results_v3/image_cascade_holdout_final.json` (emergency-class mAP50, risk-controlled thresholds) is **POST-HOC**: the holdout had already been used and the emergency metric was defined after seeing results, with parameters frozen on validation data.

### Discussion and limitations

- The cascade is a speed/accuracy trade-off, not an accuracy improvement: it recovers part of the gap between the light and heavy model at roughly half of the heavy model's cost in cross-validation.
- Stacking brings no measurable gain over RT-DETR-l on the emergency data.
- Faster R-CNN was under-trained by protocol mismatch (15 epochs and last checkpoint versus 60 epochs, patience 15 and best checkpoint). The logs of folds 0 and 3 show training loss plateauing near 0.097 during epochs 4-8 and dropping after the learning-rate step in epoch 9, with validation loss minimal at epoch 9 and rising afterwards; the logs contain no mAP. Every Faster R-CNN-dependent number must be recomputed after the retraining.
- Car labels in the validation and holdout data are incomplete, so car AP is not a clean metric. `holdout_v3` and `holdout_v4` contain the same 381 image names; 49 label files differ.
- The holdout has been consumed (`holdout_done.flag` and related flags); any further evaluation on it is POST-HOC. The near-duplicate audit of `holdout_v3` against `folds_v3` has not been run; exact-duplicate (md5) overlap is 0 of 381. The holdout is effectively one large group in the grouping used for cross-validation.
- Timing on the laptop GPU is unstable; use it for ordering, not for precise ratios.

## Abstract queue model

`phase2_bridge/toy_sim.py` is a queue model of a corridor with message passing between junctions, including emergency notices up to two junctions ahead. Results from it are labelled "abstract queue model" and say nothing about CARLA or real traffic. Two-hop messaging has been implemented and run **only in this abstract model**; it has not been run in CARLA. Unit tests: `phase2_bridge/test_phase2.py`, `test_toy_sim.py`, `test_multihop.py`.

## Phase 2: CARLA

Single junction (Town05_Opt, junction 14), four approaches, two phases, 160 s episodes. Policies: fixed time, fixed time with simple preemption, max-pressure, fuzzy controller. Perception inputs: ground truth, injected noise (swept and measured rates), and a detector in the loop (hybrid mode: queue and distance from the simulator API, emergency class from a detector on camera frames). Full method, tables and figures: [`phase2_bridge/README.md`](phase2_bridge/README.md).

Findings, in short (CARLA):

- Without noise, preemption policies reach the floor of about 2 s for the ambulance's stopped time; max-pressure is best for ordinary traffic.
- With the fine-tuned detector in the loop, the fuzzy controller shows no advantage over simple preemption for ambulance stopped time (the lower bound of the interval is 0.0) and costs ordinary vehicles about 2.0-2.5 s each, with intervals excluding zero in all four weather conditions.
- The Phase 1 detector without fine-tuning on CARLA frames almost never recognised the ambulance in the loop.
- Noise results produced before the noise-stream reproducibility fix are not reproducible and are not reported.

## Limitations (summary)

- One junction on one map; held-out CARLA captures use the same junction and vehicle models, so they are not a new environment.
- The four weather conditions share the same traffic scene for equal seeds, so they are not four independent repetitions.
- The fine-tuned detector's validation-set numbers are DEV and optimistic.
- Injected-noise experiments are stress tests, not predictions; the visibility proxy is uncalibrated.
- No real-world data; the sim-to-real gap is not measured here.
- Ambulance time-to-cross has survival bias (only vehicles that crossed are counted).

## Repository layout

| Path | Content |
|---|---|
| `datasets/` | dataset preparation, fold generation and training scripts (no data) |
| `results_v3/` | frozen protocol, reports, holdout results, cross-validation JSON, training logs |
| `stacking/` | detection dumps, ensemble and stacking evaluation, cascade, audits |
| `figures_pipeline/` | scripts that build the figures |
| `phase2_bridge/` | controller, CARLA bridge, abstract queue model, tests, Phase 2 results |
| `frcnn60/` | Faster R-CNN 60-epoch retraining scripts (training in progress) |
| `requirements_frozen.txt` | frozen package versions |

Datasets and model weights are not committed.

## Status

- Faster R-CNN retraining in progress; the stacking, ensemble and cascade tables above will be recomputed afterwards, with the previous numbers kept as the under-trained 15-epoch baseline.
- Planned: leakage audit of the consumed holdout against the training folds, and a protocol frozen before any further holdout evaluation.