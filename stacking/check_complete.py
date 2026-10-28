import sys
sys.path.insert(0, ".")
import dump_dets as d
miss = [f"{m}:fold{f}" for m in ("yolov8n", "rtdetr_l", "fasterrcnn") for f in range(5) if not d.complete(m, f)]
print("missing:", miss if miss else "none")
sys.exit(1 if miss else 0)
