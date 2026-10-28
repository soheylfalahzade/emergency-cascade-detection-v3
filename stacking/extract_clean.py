import sys, os
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_scores as es

es.BASE = es.ROOT / "datasets/folds_clean"
es.RES = es.ROOT / "results_clean"
es.OUT = es.RES / "stacking"
es.OUT.mkdir(parents=True, exist_ok=True)
MAX_SIDE = 800


class _B:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls


class _R:
    def __init__(self, shape, b):
        self.orig_shape, self.boxes = shape, b


class FRCNNWrap:
    def __init__(self, path):
        m = fasterrcnn_resnet50_fpn_v2(weights=None)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 5)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.roi_heads.score_thresh = 0.001
        m.roi_heads.detections_per_img = 300
        self.m = m.eval()

    def predict(self, paths, conf=0.001, imgsz=640, device="0", verbose=False, **kw):
        dev = torch.device("cpu" if str(device) == "cpu" else "cuda")
        self.m.to(dev)
        out = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            W0, H0 = im.size
            s = min(1.0, MAX_SIDE / max(W0, H0))
            if s < 1.0:
                im = im.resize((max(1, int(W0 * s)), max(1, int(H0 * s))), Image.BILINEAR)
            with torch.no_grad():
                o = self.m([F.to_tensor(im).to(dev)])[0]
            out.append(_R((H0, W0), _B((o["boxes"].cpu() / s), o["scores"].cpu(), (o["labels"].cpu() - 1))))
        return out


if __name__ == "__main__":
    model, folds = sys.argv[1], [int(f) for f in sys.argv[2:]]
    if model == "fasterrcnn":
        es.YOLO = FRCNNWrap
        for f in folds:
            d = es.RES / f"fasterrcnn_fold{f}" / "weights"
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "best.pt").exists():
                os.symlink((es.RES / f"fasterrcnn_fold{f}" / "last.pt").resolve(), d / "best.pt")
    for f in folds:
        es.main(model, f)
