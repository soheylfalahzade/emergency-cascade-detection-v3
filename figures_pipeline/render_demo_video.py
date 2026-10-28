"""Qualitative cascade demo video: light detector + tracker, escalation to the heavy detector on uncertain frames.
Not an evaluation. Use --device 0 after training for speed."""
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import RTDETR, YOLO

from common import *

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(P1_OUT / "p1_demo_cascade.mp4"))
ap.add_argument("--light", default=str(RES / "yolov8n_fold0/weights/best.pt"))
ap.add_argument("--heavy", default=str(RES / "rtdetr_l_fold0/weights/best.pt"))
ap.add_argument("--no_heavy", action="store_true")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--band", type=float, default=0.5)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--max_frames", type=int, default=300)
ap.add_argument("--device", default="cpu")
ap.add_argument("--tracker", default="bytetrack.yaml")
ap.add_argument("--width", type=int, default=1280)
a = ap.parse_args()

light = YOLO(a.light)
heavy = None if a.no_heavy else RTDETR(a.heavy)
cap = cv2.VideoCapture(a.video)
fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()
fps = float(min(max(fps_src / a.stride, 5.0), 15.0))
trail = defaultdict(lambda: deque(maxlen=30))
writer, n, esc_n = None, 0, 0
gen = light.track(source=a.video, tracker=a.tracker, persist=True, stream=True, conf=a.conf, imgsz=a.imgsz,
                  device=a.device, vid_stride=a.stride, verbose=False)
for r in gen:
    n += 1
    if n > a.max_frames:
        break
    H0, W0 = r.orig_img.shape[:2]
    s = a.width / W0
    im = cv2.resize(r.orig_img, (a.width, int(H0 * s)))
    b = r.boxes
    xyxy = b.xyxy.cpu().numpy() * s
    cl = b.cls.cpu().numpy().astype(int)
    cf = b.conf.cpu().numpy()
    ids = b.id.cpu().numpy().astype(int) if b.id is not None else -np.ones(len(cl), int)
    esc = bool(len(cf) and (cf < a.band).any())
    esc_n += int(esc)
    for k in range(len(cl)):
        col = hex_to_bgr(CLASS_COL[CLASSES[cl[k]]])
        x1, y1, x2, y2 = [int(v) for v in xyxy[k]]
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        cv2.putText(im, f"{ids[k] if ids[k] >= 0 else '-'} {CLASSES[cl[k]][:3]} {cf[k]:.2f}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if ids[k] >= 0:
            trail[int(ids[k])].append((int((x1 + x2) / 2), int(y2)))
            pts = np.array(trail[int(ids[k])], np.int32).reshape(-1, 1, 2)
            if len(pts) > 1:
                cv2.polylines(im, [pts], False, col, 1)
    if esc and heavy is not None:
        rh = heavy.predict(r.orig_img, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
        hb = rh.boxes.xyxy.cpu().numpy() * s
        for k in range(len(hb)):
            x1, y1, x2, y2 = [int(v) for v in hb[k]]
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(im, f"H {CLASSES[int(rh.boxes.cls[k])][:3]} {float(rh.boxes.conf[k]):.2f}", (x1, y2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(im, f"frame {n}  tracks {len(set(ids[ids >= 0].tolist()))}  escalated {esc_n}/{n} ({esc_n / n:.0%})"
                    f"  band<{a.band}{'  [heavy ON this frame]' if esc and heavy is not None else ''}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(im, "Qualitative demo on a development video - not an evaluation", (8, im.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if writer is None:
        writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (im.shape[1], im.shape[0]))
    writer.write(im)
if writer is not None:
    writer.release()
print("saved", a.out, "| frames", min(n, a.max_frames), "| escalation rate", f"{esc_n / max(min(n, a.max_frames), 1):.2f}")
