"""Detections of three COCO-pretrained detectors on COCO val2017 restricted to four traffic classes.
Classes: 0 car, 1 motorcycle, 2 bus, 3 truck (matches the four-class layout of stackcore). Crowd annotations are excluded.
The dev/test split (seed 42, 2500/2500) is written to results_coco/split.json."""
import os, json, pickle
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
CB = Path.home() / "Portfolio/datasets/coco_bench"
OUT = ROOT / "results_coco"; (OUT / "dets").mkdir(parents=True, exist_ok=True)
DEVICE, BS = os.environ.get("DEVICE", "0"), 16
CAT = {3: 0, 4: 1, 6: 2, 8: 3}   # COCO category id -> class
YMAP = {2: 0, 3: 1, 5: 2, 7: 3}  # ultralytics 80-class index -> class
TMAP = {3: 0, 4: 1, 6: 2, 8: 3}  # torchvision 91-class index -> class
NAMES = ["car", "motorcycle", "bus", "truck"]

ann = json.load(open(CB / "annotations/instances_val2017.json"))
imgs = sorted(ann["images"], key=lambda x: x["id"])
gt = {im["id"]: ([], []) for im in imgs}
ncrowd = ntiny = 0
for a in ann["annotations"]:
    if a["category_id"] not in CAT:
        continue
    if a["iscrowd"]:
        ncrowd += 1; continue
    x, y, w, h = a["bbox"]
    if w < 1 or h < 1:
        ntiny += 1; continue
    gt[a["image_id"]][0].append([x, y, x + w, y + h]); gt[a["image_id"]][1].append(CAT[a["category_id"]])
cnt = {NAMES[c]: sum(g[1].count(c) for g in gt.values()) for c in range(4)}
print(f"images={len(imgs)} GT boxes={cnt} excluded: crowd={ncrowd} tiny(<1px)={ntiny}")
spf = OUT / "split.json"
if not spf.exists():
    perm = np.random.RandomState(42).permutation(len(imgs)); h = len(imgs) // 2
    json.dump(dict(seed=42, dev=sorted(int(imgs[i]["id"]) for i in perm[:h]), test=sorted(int(imgs[i]["id"]) for i in perm[h:])), open(spf, "w"))
sp = json.load(open(spf)); print("split:", len(sp["dev"]), "dev /", len(sp["test"]), "test", "| dev&test overlap:", len(set(sp["dev"]) & set(sp["test"])))


def rows_from(preds):
    rows = []
    for im, (b, c, k) in zip(imgs, preds):
        gb, gc = gt[im["id"]]
        rows.append(dict(image=im["file_name"], image_id=im["id"], W=im["width"], H=im["height"],
                         boxes=np.asarray(b, np.float32).reshape(-1, 4), conf=np.asarray(c, np.float32), cls=np.asarray(k, np.int64),
                         gt_boxes=np.asarray(gb, np.float32).reshape(-1, 4), gt_cls=np.asarray(gc, np.int64)))
    return rows


def pred_ultra(model):
    from ultralytics import YOLO, RTDETR
    ck = "rtdetr-l.pt" if model.startswith("rtdetr") else "yolov8n.pt"
    net = (RTDETR if model.startswith("rtdetr") else YOLO)(str(ROOT / "datasets" / ck))
    assert len(net.names) == 80 and net.names[2] == "car" and net.names[7] == "truck", "unexpected class names"
    paths = [str(CB / "val2017" / im["file_name"]) for im in imgs]; out = []
    for i in range(0, len(paths), BS):
        for r in net.predict(paths[i:i + BS], conf=0.001, imgsz=640, device=DEVICE, verbose=False):
            c = r.boxes.cls.cpu().numpy().astype(int); k = np.isin(c, list(YMAP))
            out.append((r.boxes.xyxy.cpu().numpy()[k], r.boxes.conf.cpu().numpy()[k], np.array([YMAP[v] for v in c[k]], np.int64)))
        if (i // BS) % 40 == 0:
            print(f"  {model}: {i}/{len(paths)}", flush=True)
    return out


def pred_frcnn():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    net = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=0.001, box_detections_per_img=300)
    dev = torch.device("cuda"); net.to(dev).eval(); out = []
    for i, im in enumerate(imgs):
        x = F.to_tensor(Image.open(CB / "val2017" / im["file_name"]).convert("RGB")).to(dev)
        with torch.no_grad():
            o = net([x])[0]
        l = o["labels"].cpu().numpy(); k = np.isin(l, list(TMAP))
        out.append((o["boxes"].cpu().numpy()[k], o["scores"].cpu().numpy()[k], np.array([TMAP[v] for v in l[k]], np.int64)))
        if i % 500 == 0:
            print(f"  fasterrcnn: {i}/{len(imgs)}", flush=True)
    return out


for m in ("yolov8n", "rtdetr_l", "fasterrcnn"):
    p = OUT / "dets" / f"{m}_val2017.pkl"
    if p.exists():
        print("skip (exists):", m); continue
    rows = rows_from(pred_frcnn() if m == "fasterrcnn" else pred_ultra(m))
    pickle.dump(rows, open(p.with_suffix(".tmp"), "wb")); os.replace(p.with_suffix(".tmp"), p)
    print(f"saved {p} | images {len(rows)} | detections {sum(len(r['conf']) for r in rows)}", flush=True)
