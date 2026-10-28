"""Detection-level helpers on the stored per-image detections (results_v3/dets/*.pkl)."""
import pickle

import numpy as np

from common import RES
from stackcore import iou_mat

DETS = RES / "dets"


def available_folds(models, split="val"):
    return [f for f in range(5) if all((DETS / f"{m}_fold{f}_{split}.pkl").exists() for m in models)]


def load_rows(m, f, split="val"):
    return pickle.load(open(DETS / f"{m}_fold{f}_{split}.pkl", "rb"))


def match_class(rows, cls, min_conf=0.001, iou_thr=0.5):
    """Greedy per-class matching. Returns (conf, is_tp, n_gt)."""
    confs, tps, n_gt = [], [], 0
    for r in rows:
        gi = np.where(r["gt_cls"] == cls)[0]
        n_gt += len(gi)
        di = np.where((r["cls"] == cls) & (r["conf"] >= min_conf))[0]
        if len(di) == 0:
            continue
        di = di[np.argsort(-r["conf"][di])]
        iou = iou_mat(r["boxes"][di], r["gt_boxes"][gi]) if len(gi) else np.zeros((len(di), 0))
        used = np.zeros(len(gi), bool)
        for n, i in enumerate(di):
            hit = False
            if len(gi):
                row = np.where(used, -1.0, iou[n])
                j = int(row.argmax())
                if row[j] >= iou_thr:
                    used[j] = True
                    hit = True
            confs.append(float(r["conf"][i]))
            tps.append(hit)
    return np.array(confs), np.array(tps, bool), n_gt


def pr_curve(conf, tp, n_gt):
    """Precision-recall curve and all-point-interpolated AP (own implementation, not torchmetrics)."""
    if len(conf) == 0 or n_gt == 0:
        return np.array([0.0]), np.array([1.0]), 0.0
    o = np.argsort(-conf)
    tpc, fpc = np.cumsum(tp[o]), np.cumsum(~tp[o])
    rec, prec = tpc / n_gt, tpc / np.maximum(tpc + fpc, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return rec, prec, float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def confusion(rows, thr=0.25):
    """Rows: true class; columns: predicted class (best overlapping detection) + missed."""
    M = np.zeros((4, 5), int)
    for r in rows:
        k = r["conf"] >= thr
        b, s, c = r["boxes"][k], r["conf"][k], r["cls"][k]
        gb, gc = r["gt_boxes"], r["gt_cls"]
        if len(gb) == 0:
            continue
        iou = iou_mat(gb, b) if len(b) else np.zeros((len(gb), 0))
        for g in range(len(gb)):
            cand = np.where(iou[g] >= 0.5)[0]
            if len(cand) == 0:
                M[gc[g], 4] += 1
            else:
                M[gc[g], int(c[cand[np.argmax(s[cand])]])] += 1
    return M
