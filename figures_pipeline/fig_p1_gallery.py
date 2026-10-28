import random

import cv2

from common import *
from det_utils import *
from stackcore import iou_mat

setup_style()
folds = available_folds(MODELS)
if not folds:
    skip("no dets")
f = folds[0]
rows = {m: load_rows(m, f) for m in MODELS}
names = [r["image"] for r in rows[MODELS[0]]]
for m in MODELS:
    assert [r["image"] for r in rows[m]] == names
IMG = ROOT / f"datasets/folds_v3/fold_{f}/val/images"
THR, W, H = 0.25, 300, 210


def n_matched(r):
    k = r["conf"] >= THR
    b, c = r["boxes"][k], r["cls"][k]
    n = 0
    for g in range(len(r["gt_cls"])):
        if r["gt_cls"][g] < 3 and len(b):
            iou = iou_mat(r["gt_boxes"][g:g + 1], b)[0]
            if ((c == r["gt_cls"][g]) & (iou >= 0.5)).any():
                n += 1
    return n


em_idx = [i for i in range(len(names)) if (rows[MODELS[0]][i]["gt_cls"] < 3).any()]
score = {i: [n_matched(rows[m][i]) for m in MODELS] for i in em_idx}
dis = [i for i in sorted(em_idx, key=lambda i: -(max(score[i]) - min(score[i]))) if max(score[i]) != min(score[i])][:4]
rest = [i for i in em_idx if i not in dis]
random.Random(0).shuffle(rest)
sel = dis + rest[:4]


def cell(i, which):
    im = cv2.imread(str(IMG / names[i]))
    if im is None:
        return np.full((H, W, 3), 255, np.uint8)
    h0, w0 = im.shape[:2]
    im = cv2.resize(im, (W, H))
    sx, sy = W / w0, H / h0
    if which == "gt":
        r = rows[MODELS[0]][i]
        boxes, cls, conf = r["gt_boxes"], r["gt_cls"], None
    else:
        rr = rows[which][i]
        k = rr["conf"] >= THR
        boxes, cls, conf = rr["boxes"][k], rr["cls"][k], rr["conf"][k]
    for j, b in enumerate(boxes):
        x1, y1, x2, y2 = int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)
        col = hex_to_bgr(CLASS_COL[CLASSES[int(cls[j])]])
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        if conf is not None:
            cv2.putText(im, f"{CLASSES[int(cls[j])][:3]} {conf[j]:.2f}", (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return im


body = np.vstack([np.hstack([cell(i, w) for w in ["gt"] + MODELS]) for i in sel])
head = np.full((26, body.shape[1], 3), 255, np.uint8)
for k, t in enumerate(["ground truth"] + [MODEL_LABEL[m] for m in MODELS]):
    cv2.putText(head, t, (k * W + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(head, "blue=ambulance orange=police red=firetruck green=car; first rows: models disagree",
            (3 * W - 40, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)
out = P1_OUT / f"p1_gallery_fold{f}.png"
cv2.imwrite(str(out), np.vstack([head, body]))
print("saved", out)
