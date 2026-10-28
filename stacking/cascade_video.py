import sys, json, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, torch
from PIL import Image
import cv2
from torchvision.transforms import functional as F
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_core import Track, CascadePolicy
import extract_clean as ec

CLASSES = ["ambulance", "police", "firetruck", "car"]
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--light", required=True, help="path to YOLO weights, or 'coco' for raw yolov8n COCO (cars only)")
ap.add_argument("--heavy", default=None, help="Faster R-CNN state_dict (last.pt); omit to disable escalation")
ap.add_argument("--tracker", default="botsort.yaml")
ap.add_argument("--device", default="cpu")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--conf_thr", type=float, default=0.6)
ap.add_argument("--stab_thr", type=float, default=0.7)
ap.add_argument("--min_obs", type=int, default=3)
ap.add_argument("--no_verify_rare", action="store_true")
ap.add_argument("--max_esc_per_track", type=int, default=3)
ap.add_argument("--esc_gap", type=int, default=5)
ap.add_argument("--out", default=None)
a = ap.parse_args()

coco = a.light == "coco"
light = YOLO("yolov8n.pt" if coco else a.light)
kw = dict(classes=[2, 5, 7]) if coco else {}
heavy = None
if a.heavy:
    heavy = ec.FRCNNWrap(a.heavy)
    heavy.m.to(torch.device("cpu" if a.device == "cpu" else "cuda"))
dev = torch.device("cpu" if a.device == "cpu" else "cuda")
pol = CascadePolicy(a.conf_thr, a.stab_thr, a.min_obs, verify_rare=not a.no_verify_rare)


def heavy_classify(bgr, box, pad=0.1):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(W, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(H, y2 + pad * bh))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    im = Image.fromarray(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    s = min(1.0, 800 / max(im.size))
    if s < 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)))
    with torch.no_grad():
        o = heavy.m([F.to_tensor(im).to(dev)])[0]
    if len(o["scores"]) == 0:
        return None
    i = int(o["scores"].argmax())
    return int(o["labels"][i]) - 1


tracks, esc_log, n_det = {}, Counter(), 0
hvotes, last_esc = defaultdict(list), {}
esc_count, frames, t_light, t_heavy = Counter(), 0, 0.0, 0.0
t0 = time.time()
gen = light.track(source=a.video, tracker=a.tracker, persist=True, stream=True, conf=a.conf,
                  imgsz=a.imgsz, device=a.device, vid_stride=a.stride, verbose=False, **kw)
for r in gen:
    frames += 1
    if a.max_frames and frames > a.max_frames:
        break
    if r.boxes.id is None:
        continue
    for box, tid, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int),
                              r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        c = 3 if coco else int(c)
        tr = tracks.setdefault(int(tid), Track(int(tid)))
        tr.update(c, p); n_det += 1
        act, lab, why = pol.decide(tr, c, p)
        if act == "escalate":
            for w in why:
                esc_log[w] += 1
            if (heavy is not None and tr.heavy_cls is None and esc_count[tid] < a.max_esc_per_track
                    and tr.n >= a.min_obs and tr.n - last_esc.get(int(tid), -10**9) >= a.esc_gap):
                th = time.time()
                hc = heavy_classify(r.orig_img, box)
                t_heavy += time.time() - th
                esc_count[tid] += 1
                last_esc[int(tid)] = tr.n
                if hc is not None:
                    hvotes[int(tid)].append(hc)
                if esc_count[tid] >= a.max_esc_per_track and hvotes[int(tid)]:
                    tr.heavy_cls = Counter(hvotes[int(tid)]).most_common(1)[0][0]

total = time.time() - t0
for _t, _v in hvotes.items():
    if _v and tracks[_t].heavy_cls is None:
        tracks[_t].heavy_cls = Counter(_v).most_common(1)[0][0]
final = {}
for tid, tr in tracks.items():
    v, s = tr.vote()
    final[tid] = dict(light=v, stability=s, heavy=tr.heavy_cls, n_obs=tr.n,
                      final=tr.heavy_cls if tr.heavy_cls is not None else v)
long_ = {k: v for k, v in final.items() if v["n_obs"] >= a.min_obs}
dist_light = Counter(CLASSES[v["light"]] for v in long_.values())
dist_final = Counter(CLASSES[v["final"]] for v in long_.values())
summary = dict(frames=frames - (1 if a.max_frames and frames > a.max_frames else 0),
               tracks_total=len(final), tracks_min_obs=len(long_), detections=n_det,
               light_only=dict(dist_light), after_cascade=dict(dist_final),
               escalated_tracks=len(esc_count), escalation_reasons=dict(esc_log),
               sec_total=round(total, 1), sec_heavy=round(t_heavy, 1), device=a.device,
               config=vars(a))
print(json.dumps(summary, indent=1, ensure_ascii=False))
if a.out:
    json.dump(dict(summary=summary, tracks=final), open(a.out, "w"), indent=1, ensure_ascii=False)
