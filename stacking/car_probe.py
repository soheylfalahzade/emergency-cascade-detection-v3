import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
from stackcore import iou_mat

R = Path.home() / "Portfolio/Smart_Traffic_System"
VAL = R / "datasets/folds_v3/fold_0/val/images"
ours = YOLO(str(R / "results_v3/yolov8n_fold0/weights/best.pt"))
coco = YOLO(str(R / "datasets/yolov8n.pt"))
imgs = sorted(p for p in VAL.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
S = defaultdict(lambda: {k: [0, 0, 0] for k in ("ours", "coco")})
ngt = defaultdict(int)


def gt_car(p, W, H):
    lp = p.parent.parent / "labels" / (p.stem + ".txt")
    a = np.loadtxt(lp, ndmin=2) if lp.exists() else np.zeros((0, 5))
    a = a[a[:, 0] == 3] if a.size else a
    if not len(a):
        return np.zeros((0, 4))
    xc, yc, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)


def match(pb, ps, gb):
    tp, used = 0, set()
    if len(pb) and len(gb):
        iou = iou_mat(pb, gb)
        for i in np.argsort(-ps):
            row = iou[i].copy()
            for j in used:
                row[j] = -1
            j = int(row.argmax())
            if row[j] >= 0.5:
                tp += 1; used.add(j)
    return tp, len(pb) - tp, len(gb) - tp


for i in range(0, len(imgs), 16):
    ch = imgs[i:i + 16]
    ro = ours.predict([str(p) for p in ch], conf=0.25, imgsz=640, device="cpu", verbose=False)
    rc = coco.predict([str(p) for p in ch], conf=0.25, imgsz=640, device="cpu", verbose=False)
    for p, a, b in zip(ch, ro, rc):
        H, W = a.orig_shape
        gb = gt_car(p, W, H)
        src = p.name.split("__")[0]
        ngt[src] += len(gb)
        for key, r, keep in (("ours", a, lambda k: k == 3), ("coco", b, lambda k: k in (2, 5, 7))):
            k = r.boxes.cls.cpu().numpy().astype(int)
            m = np.array([keep(x) for x in k], bool) if len(k) else np.zeros(0, bool)
            pb = r.boxes.xyxy.cpu().numpy()[m]; ps = r.boxes.conf.cpu().numpy()[m]
            tp, fp, fn = match(pb, ps, gb)
            for t, v in enumerate((tp, fp, fn)):
                S[src][key][t] += v
print(f"{'source':<26}{'GTcar':>7} | {'ours P':>7}{'R':>6} | {'COCO P':>7}{'R':>6}")
for s in sorted(S):
    row = []
    for key in ("ours", "coco"):
        tp, fp, fn = S[s][key]
        row += [tp / max(tp + fp, 1), tp / max(tp + fn, 1)]
    print(f"{s:<26}{ngt[s]:>7} | {row[0]:>7.2f}{row[1]:>6.2f} | {row[2]:>7.2f}{row[3]:>6.2f}")
