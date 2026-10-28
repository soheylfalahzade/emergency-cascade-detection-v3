import sys
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO
R = Path.home() / "Portfolio/Smart_Traffic_System"
m = YOLO(str(R / "results_v4/yolov8n_fold0/weights/best.pt"))
coco = YOLO("yolov8m.pt")
out = Path.home() / "fp_crops"; out.mkdir(exist_ok=True)
for v in sys.argv[1:]:
    print("==", Path(v).name)
    ref, n = 0, 0
    for r in coco.predict(source=v, stream=True, conf=0.3, imgsz=1920, device=0, vid_stride=10, classes=[2, 5, 7], verbose=False):
        ref += len(r.boxes); n += 1
    print(f"  COCO yolov8m@1920 vehicles/frame: {ref/max(n,1):.1f}")
    for sz in (960, 1280, 1920):
        car = emg = k = 0; sizes = []
        for r in m.predict(source=v, stream=True, conf=0.25, imgsz=sz, device=0, vid_stride=10, verbose=False):
            k += 1
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if int(c) == 3: car += 1; sizes.append(max(b[2]-b[0], b[3]-b[1]))
                elif s >= 0.5:
                    emg += 1
                    if sz == 960 and emg <= 12:
                        x1, y1, x2, y2 = map(int, b); cv2.imwrite(str(out / f"{Path(v).stem}_{k}_{int(c)}_{s:.2f}.jpg"), r.orig_img[y1:y2, x1:x2])
        print(f"  imgsz={sz}: car/frame={car/max(k,1):.1f} emergency(>=.5)/frame={emg/max(k,1):.2f} median car px={np.median(sizes) if sizes else 0:.0f}")
