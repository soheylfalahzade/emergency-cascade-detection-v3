# Final results (auto-generated)

Frozen protocol (2026-10-28 14:33:48): meta=LR, heavy=rtdetr_l, cascade=band<0.7 (fallback=False)

## 5-fold CV on val images (leave-one-fold-out meta), mean±std over folds

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

## Cascade (light=yolov8n, heavy=rtdetr_l), CV; image-level escalation, full-image heavy model

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

## Cost (GPU NVIDIA GeForce RTX 5060 Laptop GPU, 100 images, fold0)

| model | params (M) | GFLOPs | ms/img median | FPS |
|---|---|---|---|---|
| yolov8n | 3.0 | 8.1 | 5.2 | 190.6 |
| rtdetr_l | 32.8 | 109.2 | 26.7 | 37.4 |
| fasterrcnn | 43.3 | 559.9 | 66.6 | 15.0 |

## Holdout (evaluated once; 381 images, boxes {'ambulance': 191, 'police': 15, 'firetruck': 221, 'car': 259}); mean over 5 fold-models

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

Paired differences (bootstrap):
- cascade_minus_heavy: -0.0215 CI95 [-0.04201138943433762, -0.003497161865234391]
- cascade_minus_light: +0.0232 CI95 [0.004284983575344109, 0.045057410299778]
- meta_minus_heavy: +0.0004 CI95 [-0.005474227666854859, 0.0058858746290206465]
- softavg_minus_heavy: -0.0055 CI95 [-0.02423423945903779, 0.004106527864933048]
- wbf_minus_heavy: -0.0073 CI95 [-0.02765033900737757, 0.002927975356578827]
- meta_minus_wbf: +0.0077 CI95 [0.0019459182024001958, 0.024410107433795847]

## Caveats (fixed text)
- best.pt/early-stop for YOLO and RT-DETR were selected on each fold's val; Faster R-CNN uses the last epoch.
- Metrics: torchmetrics mAP, conf>=0.02, top-100 per image, same code for all methods.
- Police has very few boxes; any police number needs its interval.
- Val car labels (UA-DETRAC) are incomplete; car AP is not a clean metric.
- Cascade escalation is image-level with the full image sent to the heavy model.
- No emergency-vehicle video ground truth exists; video claims are not supported.