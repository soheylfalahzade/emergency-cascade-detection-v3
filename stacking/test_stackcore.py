import numpy as np
from stackcore import *


def rec(boxes, conf, cls):
    return dict(boxes=np.array(boxes, np.float32).reshape(-1, 4), conf=np.array(conf, np.float32), cls=np.array(cls, int))


def test_iou():
    r = iou_mat([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])[0]
    assert abs(r[0] - 1) < 1e-6 and abs(r[1] - 1 / 3) < 1e-6 and r[2] == 0


def test_cluster_features_labels():
    models = ["a", "b"]
    pm = {"a": rec([[0, 0, 100, 100], [300, 300, 400, 400]], [0.9, 0.5], [0, 3]),
          "b": rec([[2, 2, 102, 102]], [0.8], [2])}
    cl = cluster_image(pm, models)
    assert len(cl["boxes"]) == 2
    k0 = int(np.argmax(cl["anchor_conf"])); k1 = 1 - k0
    assert abs(cl["scores"][k0, 0, 0] - 0.9) < 1e-6 and abs(cl["scores"][k0, 1, 2] - 0.8) < 1e-6
    assert cl["scores"][k0, 0, 2] == 0
    assert abs(cl["scores"][k1, 0, 3] - 0.5) < 1e-6 and cl["scores"][k1, 1].sum() == 0
    assert features(cl, 800, 480).shape == (2, 9) and features(cl, 800, 480, True).shape == (2, 11)
    y = label_candidates(cl["boxes"], [[0, 0, 100, 100]], [1])
    assert y[k0] == 1 and y[k1] == BG
    assert len(cluster_image({"a": rec([], [], []), "b": rec([], [], [])}, models)["boxes"]) == 0


def test_dets_nms():
    b, s, l = dets_from_probs([[0, 0, 10, 10], [0, 0, 10, 10]], np.array([[0.9, 0, 0, 0], [0.8, 0, 0, 0]]))
    assert len(b) == 1 and abs(s[0] - 0.9) < 1e-6
    b, s, l = dets_from_probs([[0, 0, 10, 10]], np.array([[0.9, 0, 0, 0.8]]))
    assert len(b) == 2 and set(l.tolist()) == {0, 3}
    assert len(dets_from_probs(np.zeros((0, 4)), np.zeros((0, 5)))[0]) == 0
    cl = cluster_image({"a": rec([[0, 0, 10, 10]], [0.3], [1]), "b": rec([[0, 0, 10, 10]], [0.9], [2])}, ["a", "b"])
    b, s, l = fuse_top1(cl)
    assert l[0] == 2 and abs(s[0] - 0.9) < 1e-6


def test_proba5_missing_class():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 5); y = rng.choice([0, 2, 3, 4], 300)
    P = proba5(make_meta("LR").fit(X, y), X)
    assert P.shape == (300, 5) and (P[:, 1] == 0).all() and abs(P.sum(1) - 1).max() < 1e-6


def make_split(n_img, seed, models):
    rng = np.random.RandomState(seed)
    lists = {m: [] for m in models}
    for i in range(n_img):
        k = rng.randint(1, 4)
        gb = np.array([[50 + 150 * j, 50, 150 + 150 * j, 150] for j in range(k)], np.float32)
        gc = rng.randint(0, 4, k)
        for m in models:
            b, c, kc = [], [], []
            for j in range(k):
                t = gc[j]
                p = t if rng.rand() < 0.6 else rng.choice([x for x in range(4) if x != t])
                b.append(gb[j] + rng.randn(4) * 2); c.append(rng.uniform(0.5, 0.95)); kc.append(p)
            b.append([600, 300, 700, 400]); c.append(rng.uniform(0.02, 0.3)); kc.append(rng.randint(0, 4))
            lists[m].append(dict(image=f"{i}.jpg", W=800, H=480, boxes=np.array(b, np.float32),
                                 conf=np.array(c, np.float32), cls=np.array(kc), gt_boxes=gb, gt_cls=gc))
    return lists


def test_end_to_end():
    models = ["m0", "m1", "m2"]
    tr = build_split(make_split(600, 1, models), models)
    te = build_split(make_split(300, 2, models), models)
    assert (tr["y"] == BG).any() and (tr["y"] != BG).any()
    clf = make_meta("LR").fit(tr["X"], tr["y"])
    P = proba5(clf, te["X"]); nb = te["y"] != BG
    acc_meta = (P[nb, :NC].argmax(1) == te["y"][nb]).mean()
    acc0 = (te["X"][nb, :NC].argmax(1) == te["y"][nb]).mean()
    assert acc_meta > acc0 + 0.03, (acc_meta, acc0)
    lists = make_split(100, 3, models)
    gts = [(r["gt_boxes"], r["gt_cls"]) for r in lists["m0"]]
    r = evaluate_map([(gb, np.ones(len(gb)), gc) for gb, gc in gts], gts)
    assert r["map50"] > 0.99, r
    bt = build_split(lists, models)
    r2 = evaluate_map(per_image(bt, 100, proba5(clf, bt["X"])), gts)
    assert 0 <= r2["map50"] <= 1, r2


if __name__ == "__main__":
    test_iou(); test_cluster_features_labels(); test_dets_nms(); test_proba5_missing_class(); test_end_to_end()
    print("ALL STACK TESTS PASSED")
