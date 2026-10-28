import random
from pathlib import Path
from PIL import Image, ImageDraw

D = Path.home() / "Portfolio/Smart_Traffic_System/datasets/folds_v5/fold_0/train/images"
COL = {0: (0, 90, 255), 1: (255, 200, 0), 2: (255, 40, 40), 3: (0, 200, 0)}
S, COLS, ROWS = 256, 6, 5
random.seed(0)
names = sorted(p.name for p in D.iterdir())


def sheet(files, out):
    sh = Image.new("RGB", (COLS * S, ROWS * S), "white")
    for i, n in enumerate(files[:COLS * ROWS]):
        p = D / n
        im = Image.open(p).convert("RGB"); W, H = im.size
        d = ImageDraw.Draw(im)
        lp = D.parent / "labels" / (p.stem + ".txt")
        if lp.exists():
            for l in open(lp):
                v = l.split()
                if len(v) < 5:
                    continue
                c = int(v[0]); cx, cy, w, h = [float(x) for x in v[1:5]]
                d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H],
                            outline=COL.get(c, (255, 255, 255)), width=max(3, W // 150))
        im.thumbnail((S, S)); sh.paste(im, ((i % COLS) * S, (i // COLS) * S))
    sh.save(out); print("saved", out, len(files[:COLS * ROWS]))


for src in ["ev_bahrain_v6", "ev_yolo_v1", "ev_police_v1", "emergency_vehicles_raw"]:
    fs = [n for n in names if n.startswith(src + "__") and not n.rsplit(".", 1)[0].endswith("_aug") and "_dup" not in n]
    print(src, "originals in train:", len(fs))
    sheet(random.sample(fs, min(30, len(fs))), Path.home() / f"sheet_{src}.png")
vt = [n for n in names if n.startswith("vid_") and n.endswith("_r0.jpg")]
print("video tiles (r0):", len(vt))
sheet(random.sample(vt, min(30, len(vt))), Path.home() / "sheet_vid.png")
