#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env || exit 1
which python
cd ~/Portfolio/Smart_Traffic_System/datasets || exit 1
rm -f ~/STOP_TRAINING
LOG=~/Portfolio/Smart_Traffic_System/rtdetr_run_$(date +%Y%m%d_%H%M).log
total() { cat ../results/rtdetr_l_fold*/results.csv 2>/dev/null | wc -l; }
fails=0; prev=$(total)
while true; do
  python train_rtdetr_resumable.py 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  case $rc in
    0)  echo "=== ALL DONE ==="; exit 0;;
    3)  echo "=== DONE except folds needing decision ==="; exit 3;;
    42) echo "=== STOPPED BY USER ==="; exit 42;;
  esac
  now=$(total)
  if [ "$now" -gt "$prev" ]; then fails=0; else fails=$((fails+1)); fi
  prev=$now
  if [ "$fails" -ge 5 ]; then echo "=== GAVE UP after 5 fruitless retries ==="; exit 1; fi
  echo "=== crash rc=$rc, retry in 30s (fruitless=$fails/5) ==="; sleep 30
done
