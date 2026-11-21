# Final results (auto-generated)

Frozen protocol (2026-11-20 21:59:31): meta=LR, heavy=rtdetr_l, cascade=band<0.7 (fallback=False)

## 5-fold CV on val images (leave-one-fold-out meta), mean±std over folds

| method | mAP50 | mAP50-95 | AP50 police | AP50 ambulance |
|---|---|---|---|---|
| single_yolov8n | 0.8815±0.0203 | 0.7593±0.0155 | 0.9695±0.0283 | 0.9666±0.0202 |
| single_rtdetr_l | 0.9078±0.0172 | 0.7923±0.0173 | 0.9842±0.0186 | 0.9783±0.0171 |
| single_fasterrcnn | 0.8508±0.0278 | 0.7035±0.0236 | 0.9579±0.0462 | 0.9538±0.0200 |
| nms_union | 0.8824±0.0228 | 0.7315±0.0223 | 0.9723±0.0317 | 0.9814±0.0145 |
| wbf | 0.9007±0.0223 | 0.7820±0.0197 | 0.9805±0.0262 | 0.9819±0.0140 |
| top1_ensemblenet_style | 0.8754±0.0261 | 0.7270±0.0252 | 0.9586±0.0355 | 0.9736±0.0114 |
| softavg | 0.9063±0.0187 | 0.7523±0.0195 | 0.9807±0.0265 | 0.9813±0.0146 |
| meta_LR | 0.9076±0.0186 | 0.7453±0.0166 | 0.9757±0.0296 | 0.9781±0.0165 |
| meta_HGB | 0.8947±0.0139 | 0.7359±0.0187 | 0.9390±0.0322 | 0.9775±0.0168 |
| meta_RF | 0.9013±0.0216 | 0.7446±0.0213 | 0.9768±0.0314 | 0.9781±0.0163 |
| meta_SVM | 0.8824±0.0257 | 0.7164±0.0258 | 0.9702±0.0340 | 0.9708±0.0194 |
| meta_MLP | 0.9023±0.0199 | 0.7439±0.0220 | 0.9600±0.0495 | 0.9796±0.0150 |
| meta_KNN | 0.8924±0.0233 | 0.7366±0.0244 | 0.9745±0.0355 | 0.9747±0.0212 |

## Cascade (light=yolov8n, heavy=rtdetr_l), CV; image-level escalation, full-image heavy model

| policy | escalation | mAP50 | expected ms/img |
|---|---|---|---|
| light_only | 0.00 | 0.8815±0.0203 | 5.2 |
| heavy_only | 1.00 | 0.9078±0.0172 | 27.3 |
| band<0.4 | 0.15 | 0.8892±0.0142 | 9.4 |
| band<0.5 | 0.19 | 0.8942±0.0165 | 10.5 |
| band<0.6 | 0.22 | 0.8973±0.0170 | 11.2 |
| band<0.7 | 0.25 | 0.8981±0.0165 | 12.0 |
| band<0.8 | 0.28 | 0.8999±0.0159 | 12.8 |
| band<0.9 | 0.35 | 0.9036±0.0189 | 14.9 |

## Cost (GPU NVIDIA GeForce RTX 5060 Laptop GPU, 100 images, fold0)

| model | params (M) | GFLOPs | ms/img median | FPS |
|---|---|---|---|---|
| yolov8n | 3.0 | 8.1 | 5.2 | 191.1 |
| rtdetr_l | 32.8 | 109.2 | 27.3 | 36.7 |
| fasterrcnn | 43.3 | 559.9 | 71.4 | 14.0 |

## Holdout (POST-HOC: this holdout had already been consumed by the earlier 15-epoch run; evaluated once under this run's frozen protocol; 381 images, boxes {'ambulance': 191, 'police': 15, 'firetruck': 221, 'car': 259}); mean over 5 fold-models

| method | mAP50 | CI95 (group bootstrap) | AP50 police | police CI95 |
|---|---|---|---|---|
| single_yolov8n | 0.8618 | [0.828, 0.944] | 0.972 | [0.921, 1.000] |
| single_rtdetr_l | 0.9208 | [0.909, 0.964] | 0.986 | [0.960, 1.000] |
| single_fasterrcnn | 0.7709 | - | 0.976 | - |
| nms_union | 0.8830 | [0.844, 0.950] | 0.985 | [0.959, 1.000] |
| wbf | 0.9116 | [0.883, 0.957] | 0.985 | [0.957, 1.000] |
| top1_ensemblenet_style | 0.8590 | - | 0.958 | - |
| softavg | 0.9136 | [0.884, 0.960] | 0.985 | [0.957, 1.000] |
| meta_LR | 0.9211 | [0.906, 0.965] | 0.991 | [0.972, 1.000] |
| cascade_frozen | 0.8917 | [0.868, 0.960] | 0.972 | [0.921, 1.000] |

Paired differences (bootstrap):
- cascade_minus_heavy: -0.0215 CI95 [-0.04201138943433762, -0.003497161865234391]
- cascade_minus_light: +0.0232 CI95 [0.004284983575344109, 0.045057410299778]
- meta_minus_heavy: +0.0016 CI95 [-0.006849190890789031, 0.007916251718997984]
- softavg_minus_heavy: -0.0076 CI95 [-0.030484830439090706, 0.00398722916841511]
- wbf_minus_heavy: -0.0097 CI95 [-0.03280593991279601, 0.0018044492602348828]
- meta_minus_wbf: +0.0113 CI95 [0.005231912136077868, 0.028884100914001384]

## Caveats (fixed text)
- best.pt/early-stop for YOLO, RT-DETR and the retrained Faster R-CNN (60 epochs, patience 15, fitness mAP50-95) were selected on each fold's val.
- Metrics: torchmetrics mAP, conf>=0.02, top-100 per image, same code for all methods.
- Police has very few boxes; any police number needs its interval.
- Val car labels (UA-DETRAC) are incomplete; car AP is not a clean metric.
- Cascade escalation is image-level with the full image sent to the heavy model.
- No emergency-vehicle video ground truth exists; video claims are not supported.