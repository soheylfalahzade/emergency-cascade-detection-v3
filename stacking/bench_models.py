import os, json, time
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image
from torchvision.transforms import functional as F
from ultralytics import YOLO, RTDETR

R = Path.home() / "Portfolio/Smart_Traffic_System"
RES = R / "results_v3"
FOLD = int(os.environ.get("FOLD", 0)); N = int(os.environ.get("NIMG", 100))
assert torch.cuda.is_available(), "GPU required"
dev = torch.device("cuda")
VAL = R / f"datasets/folds_v3/fold_{FOLD}/val/images"
files = sorted(p for p in VAL.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})[:N]
arrs = [cv2.imread(str(p)) for p in files]
assert all(a is not None for a in arrs) and len(arrs) == N, "unreadable images"
pils = [Image.open(p).convert("RGB") for p in files]


def timeit(fn, items, warm=10):
    for i in range(warm):
        fn(items[i % len(items)])
    torch.cuda.synchronize()
    ts = []
    for it in items:
        t = time.perf_counter(); fn(it); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.mean(ts))


def flops_of(call):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            call()
        return fc.get_total_flops() / 1e9
    except Exception as e:
        print("flops failed:", repr(e)[:200]); return None


out = {"gpu": torch.cuda.get_device_name(0), "n_images": N, "fold": FOLD,
       "note": "end-to-end per image, batch 1, includes preprocessing/NMS, excludes disk read; FLOPs by torch FlopCounterMode (1 MAC = 2 FLOPs)"}
for name, ctor in (("yolov8n", YOLO), ("rtdetr_l", RTDETR)):
    y = ctor(str(RES / f"{name}_fold{FOLD}/weights/best.pt"))
    params = sum(p.numel() for p in y.model.parameters())
    med, mean = timeit(lambda a: y.predict(a, imgsz=640, conf=0.02, device=0, verbose=False), arrs)
    net = y.model.to(dev).eval().float()
    fl = flops_of(lambda: net(torch.zeros(1, 3, 640, 640, device=dev)))
    out[name] = dict(params_M=params / 1e6, flops_G=fl, ms_median=med, ms_mean=mean, fps_median=1000 / med)
    print(name, out[name], flush=True)

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
net = fasterrcnn_resnet50_fpn_v2(weights=None)
net.roi_heads.box_predictor = FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features, 5)
net.load_state_dict(torch.load(RES / f"fasterrcnn_fold{FOLD}/last.pt", map_location="cpu"))
net.roi_heads.score_thresh = 0.02; net.roi_heads.detections_per_img = 100
net.to(dev).eval()
params = sum(p.numel() for p in net.parameters())


def fr(im):
    W0, H0 = im.size; s = min(1.0, 800 / max(W0, H0))
    if s < 1.0:
        im = im.resize((max(1, int(W0 * s)), max(1, int(H0 * s))), Image.BILINEAR)
    with torch.no_grad():
        net([F.to_tensor(im).to(dev)])


med, mean = timeit(fr, pils)
fl = flops_of(lambda: net([torch.zeros(3, 800, 800, device=dev)]))
out["fasterrcnn"] = dict(params_M=params / 1e6, flops_G=fl, ms_median=med, ms_mean=mean, fps_median=1000 / med)
print("fasterrcnn", out["fasterrcnn"])
json.dump(out, open(RES / "bench.json", "w"), indent=1)
print("saved bench.json")
