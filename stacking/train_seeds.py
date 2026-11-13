"""Re-train the light detector with additional seeds on the same folds, using the hyper-parameters stored in the seed-0 runs (args.yaml).
Each (fold, seed) run is a separate process with retries and resume; detections are dumped with dump_dets.py after each seed."""
import os, sys, time, subprocess
from pathlib import Path
import yaml, torch
from ultralytics import YOLO

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"; RES = ROOT / "results_v3"
SEEDS, FOLDS, DEV = (1, 2), range(5), os.environ.get("DEVICE", "0")
DROP = {"model", "task", "mode", "project", "name", "seed", "resume", "exist_ok", "device", "save_dir"}


def complete(d):
    w = d / "weights" / "best.pt"
    return w.exists() and torch.load(w, map_location="cpu", weights_only=False).get("optimizer") is None


def child(f, s, mode):
    d = RES / f"yolov8n_s{s}_fold{f}"
    if mode == "resume":
        YOLO(str(d / "weights" / "last.pt")).train(resume=True); return
    base = yaml.safe_load(open(RES / f"yolov8n_fold{f}" / "args.yaml"))
    assert base.get("seed") != s, "seed equals the existing run"
    m, data = str(ROOT / "datasets" / "yolov8n.pt"), str(base["data"])
    assert Path(m).exists(), m
    if not Path(data).exists():
        data = str(ROOT / "datasets" / data)
    assert Path(data).exists(), data
    args = {k: v for k, v in base.items() if k not in DROP}
    args.update(data=data, project=str(RES), name=d.name, seed=s, exist_ok=True, device=DEV)
    print("ARGS", {k: args.get(k) for k in ("data", "epochs", "patience", "batch", "imgsz", "seed", "name", "pretrained", "workers")}, "model=", m, flush=True)
    YOLO(m).train(**args)


def orchestrate():
    for s in SEEDS:
        for f in FOLDS:
            d = RES / f"yolov8n_s{s}_fold{f}"
            for att in range(9):
                if os.path.exists(os.path.expanduser("~/STOP_TRAINING")):
                    print("=== STOPPED", flush=True); return
                if complete(d):
                    break
                mode = "resume" if ((d / "weights" / "last.pt").exists() and att < 3) else "fresh"
                t0 = time.time()
                rc = subprocess.run([sys.executable, __file__, "one", str(f), str(s), mode]).returncode
                print(f"{d.name} attempt {att} mode={mode} rc={rc} {time.time() - t0:.0f}s", flush=True)
                if not complete(d):
                    time.sleep(20)
            if not complete(d):
                raise SystemExit(f"{d.name}: failed after retries")
            print(f"=== DONE {d.name}", flush=True)
        subprocess.run([sys.executable, "dump_dets.py", f"yolov8n_s{s}", "0", "1", "2", "3", "4"], cwd=str(ROOT / "stacking"),
                       env={**os.environ, "DEVICE": DEV}, check=True)
        print(f"=== seed {s}: detections dumped", flush=True)
    print("=== ALL SEEDS DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        child(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        orchestrate()
