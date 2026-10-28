import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from stackcore import iou_mat

R = Path.home() / "Portfolio/Smart_Traffic_System"
m = YOLO(str(R / "results_v3/yolov8n_fold0/weights/best.pt"))
names = ["ambulance", "police", "firetruck"]
THR = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

vf = {}
for v in sys.argv[1:]:
    n, det = 0, []
    for r in m.predict(source=v, stream=True, conf=0.05, imgsz=960, device=0, vid_stride=10, verbose=False):
        n += 1
        for c, p in zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            if c != 3:
                det.append((int(c), float(p)))
        if n >= 150:
            break
    vf[Path(v).name] = (n, det)

VAL = R / "datasets/folds_v3/fold_0/val/images"
imgs = []
for p in sorted(VAL.iterdir()):
    lp = p.parent.parent / "labels" / (p.stem + ".txt")
    a = np.loadtxt(lp, ndmin=2) if lp.exists() else np.zeros((0, 5))
    if a.size and (a[:, 0] < 3).any():
        imgs.append((p, a))
tp = {0: [], 1: [], 2: []}
for i in range(0, len(imgs), 32):
    ch = imgs[i:i + 32]
    rs = m.predict([str(p) for p, _ in ch], conf=0.05, imgsz=640, device=0, verbose=False)
    for (p, a), r in zip(ch, rs):
        H, W = r.orig_shape
        db = r.boxes.xyxy.cpu().numpy(); dc = r.boxes.conf.cpu().numpy(); dk = r.boxes.cls.cpu().numpy().astype(int)
        for row in a[a[:, 0] < 3]:
            c = int(row[0]); xc, yc, w, h = row[1] * W, row[2] * H, row[3] * W, row[4] * H
            gb = np.array([[xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]])
            best = 0.0
            if len(db):
                ok = (dk == c) & (iou_mat(gb, db)[0] >= 0.5)
                if ok.any():
                    best = float(dc[ok].max())
            tp[c].append(best)

allv = np.concatenate([np.array(v) for v in tp.values()])
nf = sum(n for n, _ in vf.values())
print("val GT emergency boxes:", {names[c]: len(v) for c, v in tp.items()}, "| video frames used:", nf)
print("ASSUMES none of the videos contains a real emergency vehicle; counts are per-frame detections, not tracks")
print(f"{'thr':>5} | {'valTPR':>7} {'amb':>5} {'pol':>5} {'fire':>5} | video FP dets per 100 frames: {'amb':>6} {'pol':>6} {'fire':>6} {'total':>7}")
for t in THR:
    per = [(np.array(tp[c]) >= t).mean() for c in (0, 1, 2)]
    fp = [sum(1 for _, d in vf.values() for cc, pp in d if cc == c and pp >= t) * 100 / max(nf, 1) for c in (0, 1, 2)]
    print(f"{t:>5} | {(allv >= t).mean():>7.3f} {per[0]:>5.2f} {per[1]:>5.2f} {per[2]:>5.2f} | {'':>31}{fp[0]:>6.1f} {fp[1]:>6.1f} {fp[2]:>6.1f} {sum(fp):>7.1f}")
for k, (n, d) in vf.items():
    print(f"{k}: frames={n} emergency dets conf>=0.5: {sum(1 for _, p in d if p >= 0.5)}")
