import numpy as np

NC, BG = 4, 4
CLASSES = ["ambulance", "police", "firetruck", "car"]


def iou_mat(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def _empty_cl(M):
    return dict(boxes=np.zeros((0, 4), np.float32), scores=np.zeros((0, M, NC)),
                anchor_conf=np.zeros(0), anchor_cls=np.zeros(0, int))


def _empty_det():
    return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)


def cluster_image(per_model, models, iou_thr=0.55, min_conf=0.02):
    M = len(models)
    B, C, K, Mi = [], [], [], []
    for mi, m in enumerate(models):
        d = per_model[m]
        conf = np.asarray(d["conf"], np.float64)
        keep = conf >= min_conf
        B.append(np.asarray(d["boxes"], np.float32).reshape(-1, 4)[keep])
        C.append(conf[keep])
        K.append(np.asarray(d["cls"], int)[keep])
        Mi.append(np.full(int(keep.sum()), mi))
    boxes, conf = np.concatenate(B), np.concatenate(C)
    cls, mid = np.concatenate(K), np.concatenate(Mi)
    n = len(conf)
    if n == 0:
        return _empty_cl(M)
    assert (cls >= 0).all() and (cls < NC).all(), "class id out of range"
    o = np.argsort(-conf, kind="stable")
    boxes, conf, cls, mid = boxes[o], conf[o], cls[o], mid[o]
    iou = iou_mat(boxes, boxes)
    assigned = np.full(n, -1); anchors = []
    for i in range(n):
        if assigned[i] >= 0:
            continue
        idx = len(anchors)
        assigned[(assigned < 0) & (iou[i] >= iou_thr)] = idx
        assigned[i] = idx
        anchors.append(i)
    scores = np.zeros((len(anchors), M, NC))
    for j in range(n):
        k = assigned[j]
        if conf[j] > scores[k, mid[j], cls[j]]:
            scores[k, mid[j], cls[j]] = conf[j]
    a = np.array(anchors)
    return dict(boxes=boxes[a], scores=scores, anchor_conf=conf[a], anchor_cls=cls[a])


def features(cl, W, H, use_geom=False):
    K, M, _ = cl["scores"].shape
    f = [cl["scores"].reshape(K, M * NC), (cl["scores"].max(2) > 0).sum(1, keepdims=True)]
    if use_geom:
        b = cl["boxes"]
        w = np.clip((b[:, 2] - b[:, 0]) / W, 1e-4, None); h = np.clip((b[:, 3] - b[:, 1]) / H, 1e-4, None)
        f.append(np.stack([np.log(w * h), np.log(w / h)], 1))
    return np.concatenate(f, 1)


def label_candidates(boxes, gt_boxes, gt_cls, thr=0.5):
    y = np.full(len(boxes), BG)
    gt_boxes = np.asarray(gt_boxes).reshape(-1, 4); gt_cls = np.asarray(gt_cls)
    if len(boxes) and len(gt_boxes):
        iou = iou_mat(boxes, gt_boxes)
        j = iou.argmax(1)
        ok = iou[np.arange(len(boxes)), j] >= thr
        y[ok] = gt_cls[j[ok]]
    return y


def build_split(per_model_lists, models, use_geom=False, iou_thr=0.55, min_conf=0.02):
    ref = per_model_lists[models[0]]
    X, Y, IMG, BOX, CL = [], [], [], [], []
    for i, r in enumerate(ref):
        pm = {m: per_model_lists[m][i] for m in models}
        cl = cluster_image(pm, models, iou_thr, min_conf)
        CL.append(cl)
        K = len(cl["boxes"])
        if K == 0:
            continue
        X.append(features(cl, r["W"], r["H"], use_geom))
        Y.append(label_candidates(cl["boxes"], r["gt_boxes"], r["gt_cls"]))
        IMG.append(np.full(K, i)); BOX.append(cl["boxes"])
    F = len(models) * NC + 1 + (2 if use_geom else 0)
    if not X:
        return dict(X=np.zeros((0, F)), y=np.zeros(0, int), img=np.zeros(0, int), boxes=np.zeros((0, 4), np.float32), cl=CL)
    return dict(X=np.concatenate(X), y=np.concatenate(Y), img=np.concatenate(IMG), boxes=np.concatenate(BOX), cl=CL)


def dets_from_probs(boxes, P, score_floor=0.001, nms_iou=0.5, max_det=100):
    import torch, torchvision
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return _empty_det()
    P = np.asarray(P)[:, :NC]
    kk, cc = np.nonzero(P >= score_floor)
    if len(kk) == 0:
        return _empty_det()
    b = torch.from_numpy(np.ascontiguousarray(boxes[kk]))
    s = torch.from_numpy(P[kk, cc].astype(np.float32))
    l = torch.from_numpy(cc.astype(np.int64))
    keep = torchvision.ops.batched_nms(b, s, l, nms_iou)[:max_det]
    return b[keep].numpy(), s[keep].numpy(), l[keep].numpy()


def fuse_top1(cl, max_det=100):
    o = np.argsort(-cl["anchor_conf"])[:max_det]
    return cl["boxes"][o], cl["anchor_conf"][o].astype(np.float32), cl["anchor_cls"][o].astype(np.int64)


def fuse_softavg(cl):
    if len(cl["boxes"]) == 0:
        return _empty_det()
    return dets_from_probs(cl["boxes"], cl["scores"].mean(1))


def per_image(built, n, P):
    idx = built["img"]
    st = np.searchsorted(idx, np.arange(n), "left"); en = np.searchsorted(idx, np.arange(n), "right")
    return [dets_from_probs(built["boxes"][s:e], P[s:e]) for s, e in zip(st, en)]


class SubsampledSVM:
    def __init__(self, cap=15000, seed=0):
        self.cap, self.seed = cap, seed

    def fit(self, X, y):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        if len(y) > self.cap:
            idx = np.random.RandomState(self.seed).choice(len(y), self.cap, replace=False)
            X, y = X[idx], y[idx]
        self.m = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=0)).fit(X, y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def make_meta(name):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=8, random_state=0)
    if name == "HGB":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
    if name == "SVM":
        return SubsampledSVM()
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=200, early_stopping=True, random_state=0))
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return make_pipeline(StandardScaler(), KNeighborsClassifier(25, weights="distance"))
    raise ValueError(name)


def proba5(clf, X):
    P = np.zeros((len(X), NC + 1))
    if len(X) == 0:
        return P
    pr = clf.predict_proba(X)
    for j, c in enumerate(clf.classes_):
        P[:, int(c)] = pr[:, j]
    return P


def evaluate_map(dets, gts, chunk=256):
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    m_all = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    m_50 = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5], class_metrics=True)
    for i in range(0, len(dets), chunk):
        P = [dict(boxes=torch.as_tensor(np.asarray(b, np.float32).reshape(-1, 4)),
                  scores=torch.as_tensor(np.asarray(s, np.float32)),
                  labels=torch.as_tensor(np.asarray(l, np.int64))) for b, s, l in dets[i:i + chunk]]
        T = [dict(boxes=torch.as_tensor(np.asarray(gb, np.float32).reshape(-1, 4)),
                  labels=torch.as_tensor(np.asarray(gc, np.int64))) for gb, gc in gts[i:i + chunk]]
        m_all.update(P, T); m_50.update(P, T)
    ra, r5 = m_all.compute(), m_50.compute()
    out = dict(map50=float(ra["map_50"]), map=float(ra["map"]))
    cls = np.atleast_1d(r5["classes"].numpy()); ap = np.atleast_1d(r5["map_per_class"].numpy())
    for c, v in zip(cls, ap):
        out[f"ap50_{CLASSES[int(c)]}"] = float(v)
    return out
