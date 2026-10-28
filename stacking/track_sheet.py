import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

R = Path.home() / "Portfolio/Smart_Traffic_System"
video = sys.argv[1]; tracker = sys.argv[2] if len(sys.argv) > 2 else "bytetrack.yaml"
maxf = int(sys.argv[3]) if len(sys.argv) > 3 else 200
names = ["ambulance", "police", "firetruck", "car"]
m = YOLO(str(R / "results_v3/yolov8n_fold0/weights/best.pt"))
hist, best = defaultdict(list), {}
for fi, r in enumerate(m.track(source=video, tracker=tracker, persist=True, stream=True, conf=0.25,
                               imgsz=960, device=0, vid_stride=2, verbose=False)):
    if fi >= maxf // 2:
        break
    if r.boxes.id is None:
        continue
    for b, t, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int),
                          r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        hist[t].append((int(c), float(p)))
        if t not in best or p > best[t][0]:
            x1, y1, x2, y2 = [int(v) for v in b]
            H, W = r.orig_img.shape[:2]
            pad = int(0.15 * max(x2 - x1, y2 - y1))
            crop = r.orig_img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
            if crop.size:
                best[t] = (float(p), cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
rows = []
for t, h in hist.items():
    if len(h) < 3 or t not in best:
        continue
    cnt = Counter(c for c, _ in h); v = cnt.most_common(1)[0][0]
    rows.append((t, v, cnt[v] / len(h), len(h), best[t][0], best[t][1]))
non = [r for r in rows if r[1] != 3]
car = [r for r in rows if r[1] == 3][:8]
sel = non + car
print(f"tracks(min 3 obs): {len(rows)} | non-car: {len(non)} | shown: {len(sel)}")
for r in non:
    print(f"  id={r[0]} vote={names[r[1]]} stability={r[2]:.2f} n={r[3]} best_conf={r[4]:.2f}")
S = 220; cols = 4
sheet = Image.new("RGB", (cols * S, ((len(sel) + cols - 1) // cols) * (S + 22)), "white")
for i, (t, v, st, n, p, crop) in enumerate(sel):
    im = Image.fromarray(crop); im.thumbnail((S, S))
    x, y = (i % cols) * S, (i // cols) * (S + 22)
    sheet.paste(im, (x, y + 22))
    ImageDraw.Draw(sheet).text((x + 3, y + 4), f"id{t} {names[v]} st={st:.2f} n={n}", fill=(255, 0, 0) if v != 3 else (0, 120, 0))
out = Path.home() / "track_sheet.png"; sheet.save(out); print("saved", out)
