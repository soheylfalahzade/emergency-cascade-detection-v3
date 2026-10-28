import sys
from pathlib import Path
from collections import Counter
import numpy as np
from ultralytics import YOLO
from stackcore import iou_mat

R = Path.home() / "Portfolio/Smart_Traffic_System"
ver = sys.argv[1]
m = YOLO(str(R / f"results_{ver}/yolov8n_fold{__import__('os').environ.get('FOLD','0')}/weights/best.pt"))
coco = YOLO("yolov8m.pt")
names = {0: "ambulance", 1: "police", 2: "firetruck"}
tally, tot = Counter(), 0
for v in sys.argv[2:]:
    n = 0
    for r in m.predict(source=v, stream=True, conf=0.5, imgsz=int(__import__("os").environ.get("SZ",960)), device=0, vid_stride=10, verbose=False):
        n += 1
        em = [(b, int(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()) if int(c) != 3]
        if em:
            rc = coco.predict(r.orig_img, conf=0.25, imgsz=int(__import__("os").environ.get("SZ",960)), device=0, verbose=False)[0]
            cb = rc.boxes.xyxy.cpu().numpy(); ck = rc.boxes.cls.cpu().numpy().astype(int)
            for b, c in em:
                tot += 1
                lab = "no_coco_match"
                if len(cb):
                    iou = iou_mat(b[None], cb)[0]; j = int(iou.argmax())
                    if iou[j] >= 0.5:
                        lab = coco.names[ck[j]]
                tally[(names[c], lab)] += 1
        if n >= 150:
            break
print(f"{ver}: emergency dets conf>=0.5 over the 3 videos: {tot} (assumed all false)")
for (k, lab), c in tally.most_common():
    print(f"  {k:<10} looks like COCO '{lab}': {c}")
