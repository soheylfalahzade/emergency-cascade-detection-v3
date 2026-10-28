#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env || exit 1
R=~/Portfolio/Smart_Traffic_System; d=$R/results_v5/yolov8n_fold0; EP=25
rows() { if [ -f $d/results.csv ]; then echo $(( $(wc -l < $d/results.csv) - 1 )); else echo 0; fi; }
check() { python - << PY
import torch, sys
ck = torch.load("$d/weights/last.pt", map_location="cpu", weights_only=False)
n = $(rows)
ok = ck.get("optimizer") is not None and abs(int(ck.get("epoch")) - (n - 1)) <= 1
print("check: csv_rows=%d ckpt_epoch=%s has_optimizer=%s ok=%s" % (n, ck.get("epoch"), ck.get("optimizer") is not None, ok), flush=True)
sys.exit(0 if ok else 1)
PY
}
fails=0; prev=$(rows)
while [ "$(rows)" -lt $EP ]; do
  check || { echo "=== checkpoint check FAILED, not resuming"; exit 3; }
  yolo detect train resume model=$d/weights/last.pt
  now=$(rows)
  if [ "$now" -gt "$prev" ]; then fails=0; else fails=$((fails+1)); fi
  prev=$now
  [ "$fails" -ge 3 ] && { echo "=== GAVE UP after 3 fruitless retries"; exit 1; }
  [ "$(rows)" -lt $EP ] && { echo "=== crashed at $now epochs, retry in 30s"; sleep 30; }
done
echo "=== V5 DONE ($(rows) epochs)"
