import os, sys, shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from stackcore import iou_mat

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
DS = ROOT / "datasets"
P3, P4 = DS / "pool_v3/labels", DS / "pool_v4/labels"
EMERG = ["emergency_vehicles_raw", "ev_bahrain_v6", "ev_yolo_v1", "ev_police_v1"]
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CONF = float(os.environ.get("CONF", 0.5))
MODE = sys.argv[1] if len(sys.argv) > 1 else "dry"
assert MODE in ("dry", "build")


def find_img(name):
    src, split, stem = name.split("__", 2)
    d = DS / src / split / "images"
    for e in EXT:
        if (d / (stem + e)).exists():
            return d / (stem + e)
    return None


def flip_lines(lines):
    out = []
    for l in lines:
        c, cx, cy, w, h = l.split()[:5]
        out.append(f"{c} {1.0 - float(cx):.6f} {cy} {w} {h}")
    return out


def load_model():
    w = os.environ.get("COCO_W", "yolov8m.pt")
    try:
        return YOLO(w), w
    except Exception as e:
        print("could not load", w, "->", e, "| fallback datasets/yolov8n.pt")
        return YOLO(str(DS / "yolov8n.pt")), "yolov8n.pt"


names = sorted(p.stem for p in P3.glob("*.txt"))
origs = [n for n in names if n.split("__")[0] in EMERG and not n.endswith("_aug")]
print(f"pool_v3 labels: {len(names)} | emergency-source originals: {len(origs)} | CONF={CONF}")
model, wname = load_model()
print("COCO model:", wname)
new_lines, stats, miss, sheet_items = {}, Counter(), 0, []
for i in range(0, len(origs), 32):
    pairs = [(n, find_img(n)) for n in origs[i:i + 32]]
    ok = [(n, p) for n, p in pairs if p is not None]
    miss += len(pairs) - len(ok)
    if not ok:
        continue
    rs = model.predict([str(p) for _, p in ok], conf=0.25, imgsz=640, device=0, verbose=False)
    for (n, p), r in zip(ok, rs):
        H, W = r.orig_shape
        lines = [l for l in open(P3 / (n + ".txt")).read().splitlines() if l.strip()]
        em = []
        for l in lines:
            c, cx, cy, w, h = [float(x) for x in l.split()[:5]]
            em.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        em = np.array(em).reshape(-1, 4)
        db = r.boxes.xyxy.cpu().numpy(); dc = r.boxes.conf.cpu().numpy(); dk = r.boxes.cls.cpu().numpy().astype(int)
        added = []
        for b, c, k in zip(db, dc, dk):
            if k != 2 or c < CONF or min(b[2] - b[0], b[3] - b[1]) < 12:
                continue
            if len(em):
                iou = iou_mat(b[None], em)[0]
                ix1 = np.maximum(b[0], em[:, 0]); iy1 = np.maximum(b[1], em[:, 1])
                ix2 = np.minimum(b[2], em[:, 2]); iy2 = np.minimum(b[3], em[:, 3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                ab = (b[2] - b[0]) * (b[3] - b[1]); ae = (em[:, 2] - em[:, 0]) * (em[:, 3] - em[:, 1])
                if (iou >= 0.2).any() or (inter / ab >= 0.3).any() or (inter / np.maximum(ae, 1e-9) >= 0.5).any():
                    stats["skipped_overlap_emergency"] += 1
                    continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            added.append(f"3 {cx:.6f} {cy:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}")
        src = n.split("__")[0]
        stats[f"imgs_{src}"] += 1
        if added:
            stats[f"imgs_modified_{src}"] += 1; stats[f"car_added_{src}"] += len(added)
            new_lines[n] = lines + added
            if len(sheet_items) < 12:
                sheet_items.append((p, em, [[float(v) for v in a.split()[1:]] for a in added]))
for k in sorted(stats):
    print(f"  {k}: {stats[k]}")
tot = sum(v for k, v in stats.items() if k.startswith("car_added_"))
print("images not found:", miss, "| modified originals:", len(new_lines), "| car boxes added:", tot)
S = 320
sh = Image.new("RGB", (4 * S, 3 * S), "white")
for i, (p, em, add) in enumerate(sheet_items):
    im = Image.open(p).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    for b in em:
        d.rectangle([float(v) for v in b], outline=(0, 200, 0), width=max(2, W // 200))
    for cx, cy, w, h in add:
        d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], outline=(0, 80, 255), width=max(2, W // 200))
    im.thumbnail((S, S)); sh.paste(im, ((i % 4) * S, (i // 4) * S))
out = Path.home() / "relabel_check.png"; sh.save(out); print("saved", out, "(green=existing emergency label, blue=ADDED car)")
if MODE == "dry":
    sys.exit("dry run: nothing written")

if (DS / "pool_v4").exists():
    shutil.rmtree(DS / "pool_v4")
P4.mkdir(parents=True)
for f in P3.glob("*.txt"):
    shutil.copy(f, P4 / f.name)
for n, lines in new_lines.items():
    (P4 / (n + ".txt")).write_text("\n".join(lines) + "\n")
    if (P3 / (n + "_aug.txt")).exists():
        (P4 / (n + "_aug.txt")).write_text("\n".join(flip_lines(lines)) + "\n")
print("pool_v4 labels written:", len(list(P4.glob("*.txt"))))
for old, new in (("folds_v3", "folds_v4"), ("holdout_v3", "holdout_v4")):
    if (DS / new).exists():
        shutil.rmtree(DS / new)
    n_link = 0
    for root, dirs, files in os.walk(DS / old):
        for fn in files:
            sp = Path(root) / fn
            dp = DS / new / sp.relative_to(DS / old)
            dp.parent.mkdir(parents=True, exist_ok=True)
            if fn == "data.yaml":
                dp.write_text(sp.read_text().replace(old, new))
            elif sp.is_symlink():
                tgt = os.readlink(sp)
                if "/labels/" in str(sp) and fn.endswith(".txt"):
                    tgt = str(P4 / Path(tgt).name)
                os.symlink(tgt, dp); n_link += 1
    print(old, "->", new, "symlinks:", n_link)
print("BUILD OK")
