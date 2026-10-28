import os, sys, pickle, argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torchvision.transforms import functional as F
from ultralytics import YOLO, RTDETR

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
DS, RES = ROOT / "datasets", ROOT / "results_v3"
OUT = RES / "dets"; OUT.mkdir(parents=True, exist_ok=True)
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = os.environ.get("DEVICE", "cpu")
BS = int(os.environ.get("BS", 16))


def img_dir(fold, split):
    return DS / "holdout_v3/images" if split == "holdout" else DS / f"folds_v3/fold_{fold}/val/images"


def read_gt(img, W, H):
    lp = img.parent.parent / "labels" / (img.stem + ".txt")
    a = np.loadtxt(lp, ndmin=2) if lp.exists() else np.zeros((0, 5))
    if a.size == 0:
        return np.zeros((0, 4)), np.zeros(0, int)
    xc, yc, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1), a[:, 0].astype(int)


def complete(model, fold):
    if model == "fasterrcnn":
        p = RES / f"fasterrcnn_fold{fold}" / "train_log.txt"
        return p.exists() and "epoch 15/15 DONE" in p.read_text()
    w = RES / f"{model}_fold{fold}" / "weights" / "best.pt"
    if not w.exists():
        return False
    return torch.load(w, map_location="cpu", weights_only=False).get("optimizer") is None


def load_predictor(model, fold):
    if model == "fasterrcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        net = fasterrcnn_resnet50_fpn_v2(weights=None)
        net.roi_heads.box_predictor = FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features, 5)
        net.load_state_dict(torch.load(RES / f"fasterrcnn_fold{fold}" / "last.pt", map_location="cpu"))
        net.roi_heads.score_thresh = 0.001; net.roi_heads.detections_per_img = 300
        dev = torch.device("cpu" if DEVICE == "cpu" else "cuda")
        net.to(dev).eval()

        def pred(paths):
            out = []
            for p in paths:
                im = Image.open(p).convert("RGB"); W0, H0 = im.size
                s = min(1.0, 800 / max(W0, H0))
                if s < 1.0:
                    im = im.resize((max(1, int(W0 * s)), max(1, int(H0 * s))), Image.BILINEAR)
                with torch.no_grad():
                    o = net([F.to_tensor(im).to(dev)])[0]
                out.append((o["boxes"].cpu().numpy() / s, o["scores"].cpu().numpy(), o["labels"].cpu().numpy() - 1, (H0, W0)))
            return out
        return pred
    ctor = RTDETR if model.startswith("rtdetr") else YOLO
    net = ctor(str(RES / f"{model}_fold{fold}" / "weights" / "best.pt"))

    def pred(paths):
        rs = net.predict([str(p) for p in paths], conf=0.001, imgsz=640, device=DEVICE, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), tuple(r.orig_shape)) for r in rs]
    return pred


def run(model, fold, splits):
    todo = [s for s in splits if not (OUT / f"{model}_fold{fold}_{s}.pkl").exists()]
    if not todo:
        print(f"skip (exist): {model} fold{fold}"); return
    if not complete(model, fold):
        print(f"SKIP {model} fold{fold}: training NOT complete (or weights missing)"); return
    pred = load_predictor(model, fold)
    for split in todo:
        d = img_dir(fold, split)
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)
        rows = []
        for i in range(0, len(imgs), BS):
            chunk = imgs[i:i + BS]
            for p, (b, c, k, (H, W)) in zip(chunk, pred(chunk)):
                gb, gc = read_gt(p, W, H)
                rows.append(dict(image=p.name, W=W, H=H, boxes=np.asarray(b, np.float32).reshape(-1, 4),
                                 conf=np.asarray(c, np.float32), cls=np.asarray(k, np.int64),
                                 gt_boxes=gb.astype(np.float32), gt_cls=gc.astype(np.int64)))
            if (i // BS) % 20 == 0:
                print(f"  {model} fold{fold} {split}: {i}/{len(imgs)}", flush=True)
        out = OUT / f"{model}_fold{fold}_{split}.pkl"
        tmp = out.with_suffix(".tmp")
        pickle.dump(rows, open(tmp, "wb")); os.replace(tmp, out)
        print(f"saved {out} ({len(rows)} images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("folds", nargs="+", type=int)
    ap.add_argument("--splits", nargs="+", default=["val", "holdout"])
    a = ap.parse_args()
    for f in a.folds:
        run(a.model, f, a.splits)
