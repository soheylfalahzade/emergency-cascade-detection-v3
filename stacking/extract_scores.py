import os, sys, yaml
import numpy as np, pandas as pd
from pathlib import Path
from ultralytics import YOLO, RTDETR

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
BASE, RES = ROOT / "datasets/folds", ROOT / "results"
OUT = RES / "stacking"; OUT.mkdir(parents=True, exist_ok=True)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = os.environ.get("DEVICE", "0")
BS = int(os.environ.get("BS", 16))


def val_images(fold):
    y = BASE / f"fold_{fold}" / "data.yaml"
    d = yaml.safe_load(open(y))
    root = Path(d.get("path", y.parent))
    if not root.is_absolute():
        root = y.parent / root
    v = Path(d["val"])
    if not v.is_absolute():
        v = root / v
    if v.is_dir():
        imgs = sorted(p for p in v.rglob("*") if p.suffix.lower() in IMG_EXT)
    else:
        imgs = [Path(l.strip()) for l in open(v) if l.strip()]
    assert imgs, f"no val images resolved for fold {fold} from {v}"
    return imgs, d["names"]


def label_path(img):
    s = str(img)
    i = s.rfind("/images/")
    assert i >= 0, f"no /images/ in path: {s}"
    return Path(s[:i] + "/labels/" + Path(s[i + 8:]).with_suffix(".txt").as_posix())


def read_gt(img, W, H):
    p = label_path(img)
    if not p.exists():
        return np.zeros((0,), int), np.zeros((0, 4))
    a = np.loadtxt(p, ndmin=2)
    if a.size == 0:
        return np.zeros((0,), int), np.zeros((0, 4))
    c = a[:, 0].astype(int)
    xc, yc, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return c, np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)


def iou_mat(a, b):
    if len(b) == 0:
        return np.zeros((len(a), 0))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def main(model_name, fold):
    out = OUT / f"scores_{model_name}_fold{fold}.csv"
    if out.exists():
        print(f"skip (exists): {out}"); return
    w = RES / f"{model_name}_fold{fold}" / "weights" / "best.pt"
    assert w.exists(), f"missing weights: {w}"
    ctor = RTDETR if model_name.startswith("rtdetr") else YOLO
    model = ctor(str(w))
    imgs, names = val_images(fold)
    nc = len(names)
    print(f"{model_name} fold{fold}: {len(imgs)} val images, nc={nc}, names={names}, device={DEVICE}", flush=True)
    rows = []
    for i in range(0, len(imgs), BS):
        chunk = imgs[i:i + BS]
        res = model.predict([str(p) for p in chunk], conf=0.001, imgsz=640, device=DEVICE,
                            verbose=False)
        for p, r in zip(chunk, res):
            H, W = r.orig_shape
            gc, gb = read_gt(p, W, H)
            if len(gc) == 0:
                continue
            db = r.boxes.xyxy.cpu().numpy(); dc = r.boxes.conf.cpu().numpy(); dk = r.boxes.cls.cpu().numpy().astype(int)
            iou = iou_mat(gb, db)
            for g in range(len(gc)):
                s = np.zeros(nc)
                for c in range(nc):
                    m = (dk == c) & (iou[g] >= 0.5) if len(dk) else np.zeros(0, bool)
                    s[c] = dc[m].max() if m.any() else 0.0
                row = dict(fold=fold, image=p.name, gt_idx=g, gt_cls=int(gc[g]),
                           iou_best=float(iou[g].max()) if iou.shape[1] else 0.0,
                           gt_area=float((gb[g, 2] - gb[g, 0]) * (gb[g, 3] - gb[g, 1]) / (W * H)))
                row.update({f"s{c}": s[c] for c in range(nc)})
                rows.append(row)
        if (i // BS) % 50 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out} ({len(rows)} GT boxes)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
