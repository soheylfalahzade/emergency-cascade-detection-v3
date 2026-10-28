from collections import Counter, defaultdict
from pathlib import Path

L = Path.home() / "Portfolio/Smart_Traffic_System/datasets/folds_v5/fold_0/train/labels"
P = ["emergency_vehicles_raw", "ev_bahrain_v6", "ev_yolo_v1", "ev_police_v1", "ev_zk8pk_v1",
     "kitti_yolo", "miotcd_yolo", "uadetrac_yolo", "visdrone_yolo", "vid_", "syn_"]
NM = ["amb", "pol", "fire", "car"]
imgs, orig, box = Counter(), Counter(), defaultdict(Counter)
for f in L.iterdir():
    if f.suffix != ".txt":
        continue
    s = next((p for p in P if f.name.startswith(p)), "OTHER")
    imgs[s] += 1
    if "_aug" not in f.stem and "_dup" not in f.stem:
        orig[s] += 1
        for ln in f.read_text().splitlines():
            t = ln.split()
            if len(t) >= 5 and 0 <= int(t[0]) < len(NM):
                box[s][NM[int(t[0])]] += 1
print("boxes counted on ORIGINAL images only (no _aug/_dup)")
print(f"{'source':26s}{'imgs':>7s}{'orig':>7s}" + "".join(f"{n:>8s}" for n in NM))
for s in sorted(imgs, key=lambda k: -imgs[k]):
    print(f"{s:26s}{imgs[s]:>7d}{orig[s]:>7d}" + "".join(f"{box[s][n]:>8d}" for n in NM))
