import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

W = Path.home() / "Portfolio/Smart_Traffic_System/results_v5/yolov8n_fold0/weights/best.pt"
NAMES = {0: "amb", 1: "pol", 2: "fire"}
m = YOLO(str(W))
items = []
for v in sys.argv[1:]:
    k = 0
    for r in m.predict(source=v, stream=True, conf=0.25, imgsz=1920, device=0, vid_stride=5, verbose=False):
        k += 1
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if int(c) in NAMES:
                x1, y1, x2, y2 = map(int, b)
                cr = r.orig_img[max(y1 - 20, 0):y2 + 20, max(x1 - 20, 0):x2 + 20]
                if cr.size:
                    items.append((float(s), Path(v).stem[:8], k, NAMES[int(c)], cr))
items.sort(key=lambda t: -t[0])
print("emergency dets conf>=0.25 @1920:", len(items))
S, COLS = 220, 6
sel = items[:30]
rows = max((len(sel) + COLS - 1) // COLS, 1)
sheet = np.full((rows * S, COLS * S, 3), 255, np.uint8)
for i, (s, vn, k, cn, cr) in enumerate(sel):
    h, w = cr.shape[:2]
    f = (S - 4) / max(h, w)
    cr = cv2.resize(cr, (max(int(w * f), 1), max(int(h * f), 1)))
    y, x = (i // COLS) * S, (i % COLS) * S
    sheet[y:y + cr.shape[0], x:x + cr.shape[1]] = cr
    cv2.putText(sheet, f"{cn} {s:.2f} {vn} f{k}", (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
out = Path.home() / "ev_candidates.png"
cv2.imwrite(str(out), sheet)
print("saved", out)
