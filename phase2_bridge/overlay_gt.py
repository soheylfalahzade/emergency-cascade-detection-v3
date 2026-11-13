"""Draw the recorded ground-truth boxes on saved frames so the projection can be checked by eye.
Usage: python overlay_gt.py <capture_dir> [n_frames=6]   (writes <capture_dir>/overlay/*.jpg)."""
import json
import os
import sys

import cv2

d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rows = [json.loads(l) for l in open(os.path.join(d, "gt.jsonl"), encoding="utf-8") if l.strip()]
if not rows:
    raise SystemExit("empty gt.jsonl")
os.makedirs(os.path.join(d, "overlay"), exist_ok=True)
pick = [rows[int(i * (len(rows) - 1) / max(n - 1, 1))] for i in range(n)]
emr = [r for r in rows if any(v["cls"] != "car" for v in r["vehicles"])]
if emr:
    k = max(n // 2, 1)
    pick = [emr[int(i * (len(emr) - 1) / max(k - 1, 1))] for i in range(min(k, len(emr)))] + pick[: n - min(k, len(emr))]
COL = {"ambulance": (0, 0, 255), "firetruck": (0, 128, 255), "police": (255, 0, 0), "car": (0, 200, 0)}
for r in pick:
    img = cv2.imread(os.path.join(d, r["file"]))
    if img is None:
        print("missing", r["file"])
        continue
    for v in r["vehicles"]:
        x1, y1, x2, y2 = [int(round(c)) for c in v["bbox"]]
        vis = v.get("inst_px", -1) > 0
        c = COL.get(v["cls"], (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2 if vis else 1)
        cv2.putText(img, "%s %d %.0fm%s" % (v["cls"], v["id"], v["dist_m"], "" if vis else " (no inst px)"),
                    (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    out = os.path.join(d, "overlay", r["file"].replace(".png", ".jpg"))
    cv2.imwrite(out, img)
    print(out, "vehicles:", len(r["vehicles"]), "with instance pixels:", sum(1 for v in r["vehicles"] if v.get("inst_px", -1) > 0))
