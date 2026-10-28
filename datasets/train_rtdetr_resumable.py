import os, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import RTDETR

HOME = Path.home()
ROOT = HOME / "Portfolio/Smart_Traffic_System"
BASE = ROOT / "datasets/folds"
RESULTS = ROOT / "results"
PRETRAINED = ROOT / "datasets/rtdetr-l.pt"
STOP_FILE = HOME / "STOP_TRAINING"
K, EPOCHS, IMGSZ, PATIENCE = 5, 60, 640, 15
BATCH = int(os.environ.get("BATCH", 8))
WORKERS = int(os.environ.get("WORKERS", 8))


def read_results(run_dir):
    p = run_dir / "results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df if len(df) else None


def fold_state(run_dir):
    df = read_results(run_dir)
    n = 0 if df is None else len(df)
    last = run_dir / "weights" / "last.pt"
    if n == 0 and not last.exists():
        return "fresh", "no checkpoint, no results"
    if not last.exists():
        return "anomaly", f"{n} csv rows but no last.pt"
    ck = torch.load(last, map_location="cpu", weights_only=False)
    stripped = ck.get("optimizer") is None
    info = f"csv_rows={n}, ckpt_epoch={ck.get('epoch')}, stripped={stripped}"
    if n >= EPOCHS:
        return "complete", info
    if not stripped:
        return "resume", info
    # stripped checkpoint with < EPOCHS rows: complete only if early-stop explains it
    fit = 0.1 * df["metrics/mAP50(B)"].values + 0.9 * df["metrics/mAP50-95(B)"].values
    best_epoch = len(fit) - int(np.argmax(fit[::-1]))  # 1-indexed, last max (ties like ultralytics)
    if n - best_epoch >= PATIENCE:
        return "complete", info + f", early-stopped (best_epoch={best_epoch})"
    return "anomaly", info + f", stripped but NOT explained by early-stop (best_epoch={best_epoch})"


def on_model_save(trainer):
    # runs right after last.pt is written for this epoch
    if STOP_FILE.exists():
        print(f"=== STOP_TRAINING seen after epoch {trainer.epoch + 1}; last.pt saved; exiting ===", flush=True)
        os._exit(42)


def run_fold(fold, state):
    run_name = f"rtdetr_l_fold{fold}"
    last = RESULTS / run_name / "weights" / "last.pt"
    if state == "resume":
        model = RTDETR(str(last))
        model.add_callback("on_model_save", on_model_save)
        model.train(resume=True, batch=BATCH, device=0)
    else:
        model = RTDETR(str(PRETRAINED))
        model.add_callback("on_model_save", on_model_save)
        model.train(
            data=str(BASE / f"fold_{fold}" / "data.yaml"),
            epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, workers=WORKERS, device=0,
            project=str(RESULTS), name=run_name, patience=PATIENCE, exist_ok=True,
        )


def main():
    anomalies = []
    for fold in range(K):
        if STOP_FILE.exists():
            print("=== STOP_TRAINING present, not starting new fold ===", flush=True)
            sys.exit(42)
        run_dir = RESULTS / f"rtdetr_l_fold{fold}"
        state, info = fold_state(run_dir)
        print(f"=== fold{fold}: {state} | {info} ===", flush=True)
        if state == "complete":
            continue
        if state == "anomaly":
            anomalies.append(fold)
            continue
        try:
            run_fold(fold, state)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        state2, info2 = fold_state(run_dir)
        print(f"=== fold{fold} after run: {state2} | {info2} ===", flush=True)
        if state2 != "complete":
            sys.exit(1)
    if anomalies:
        print(f"=== NEEDS DECISION: folds {anomalies} (not resumable, not complete) ===", flush=True)
        sys.exit(3)
    print("=== ALL FOLDS COMPLETE ===", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
