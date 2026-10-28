import os, sys, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch
from ultralytics import YOLO

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
BASE, RESULTS = ROOT / "datasets/folds_clean", ROOT / "results_clean"
PRETRAINED = ROOT / "datasets/yolov8n.pt"
STOP = Path.home() / "STOP_TRAINING"
K, EPOCHS, PATIENCE = 5, 60, 15
BATCH, WORKERS = int(os.environ.get("BATCH", 16)), int(os.environ.get("WORKERS", 4))


def state(run):
    p = run / "results.csv"
    df = pd.read_csv(p) if p.exists() else None
    if df is not None:
        df.columns = [c.strip() for c in df.columns]
    n = 0 if df is None or not len(df) else len(df)
    last = run / "weights" / "last.pt"
    if n == 0 and not last.exists():
        return "fresh", "no checkpoint"
    if not last.exists():
        return "anomaly", f"{n} rows, no last.pt"
    ck = torch.load(last, map_location="cpu", weights_only=False)
    stripped = ck.get("optimizer") is None
    info = f"rows={n}, stripped={stripped}"
    if n >= EPOCHS:
        return "complete", info
    if not stripped:
        return "resume", info
    fit = 0.1 * df["metrics/mAP50(B)"].values + 0.9 * df["metrics/mAP50-95(B)"].values
    best = len(fit) - int(np.argmax(fit[::-1]))
    if n - best >= PATIENCE:
        return "complete", info + f", early-stopped (best_epoch={best})"
    return "anomaly", info + f", stripped but not early-stop (best_epoch={best})"


def cb(trainer):
    if STOP.exists():
        print(f"=== STOP seen after epoch {trainer.epoch + 1}; exiting ===", flush=True)
        os._exit(42)


def main():
    bad = []
    for f in range(K):
        if STOP.exists():
            sys.exit(42)
        run = RESULTS / f"yolov8n_fold{f}"
        st, info = state(run)
        print(f"=== fold{f}: {st} | {info} ===", flush=True)
        if st == "complete":
            continue
        if st == "anomaly":
            bad.append(f); continue
        try:
            if st == "resume":
                m = YOLO(str(run / "weights" / "last.pt")); m.add_callback("on_model_save", cb)
                m.train(resume=True, batch=BATCH, workers=WORKERS, device=0)
            else:
                m = YOLO(str(PRETRAINED)); m.add_callback("on_model_save", cb)
                m.train(data=str(BASE / f"fold_{f}" / "data.yaml"), epochs=EPOCHS, imgsz=640, batch=BATCH,
                        workers=WORKERS, device=0, project=str(RESULTS), name=f"yolov8n_fold{f}",
                        patience=PATIENCE, exist_ok=True)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        st2, info2 = state(run)
        print(f"=== fold{f} after run: {st2} | {info2} ===", flush=True)
        if st2 != "complete":
            sys.exit(1)
    if bad:
        print(f"=== NEEDS DECISION: folds {bad} ==="); sys.exit(3)
    print("=== ALL FOLDS COMPLETE ===")


if __name__ == "__main__":
    main()
