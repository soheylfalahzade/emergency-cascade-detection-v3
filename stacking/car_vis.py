from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

R = Path.home() / "Portfolio/Smart_Traffic_System"
VAL = R / "datasets/folds_v3/fold_0/val/images"
m = YOLO(str(R / "results_v3/yolov8n_fold0/weights/best.pt"))
ims = [p for p in sorted(VAL.iterdir()) if p.name.startswith("uadetrac_yolo")][::max(1, 129 // 8)][:8]
W, H = 640, 360
sheet = Image.new("RGB", (2 * W, 4 * H), "white")
for i, p in enumerate(ims):
    im = Image.open(p).convert("RGB"); w0, h0 = im.size
    d = ImageDraw.Draw(im)
    lp = p.parent.parent / "labels" / (p.stem + ".txt")
    for l in open(lp):
        c, xc, yc, w, h = map(float, l.split()[:5])
        if int(c) == 3:
            d.rectangle([(xc - w / 2) * w0, (yc - h / 2) * h0, (xc + w / 2) * w0, (yc + h / 2) * h0], outline=(0, 255, 0), width=3)
    r = m.predict(str(p), conf=0.25, imgsz=640, device="cpu", verbose=False)[0]
    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        if int(c) == 3:
            d.rectangle(list(b), outline=(255, 0, 0), width=2)
    sheet.paste(im.resize((W, H)), ((i % 2) * W, (i // 2) * H))
out = Path.home() / "car_vis.png"
sheet.save(out); print("saved", out)
