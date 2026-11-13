#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env || exit 1
cd ~/Portfolio/Smart_Traffic_System/datasets || exit 1
rm -f ~/STOP_TRAINING
LOG=~/Portfolio/Smart_Traffic_System/results_v4/yolo_v4.log
total() { cat ../results_v4/yolov8n_fold*/results.csv 2>/dev/null | wc -l; }
fails=0; prev=$(total)
while true; do
  python train_yolo_v4.py 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  case $rc in 0) exit 0;; 3) exit 3;; 42) exit 42;; esac
  now=$(total)
  if [ "$now" -gt "$prev" ]; then fails=0; else fails=$((fails+1)); fi
  prev=$now
  [ "$fails" -ge 5 ] && { echo "=== GAVE UP ==="; exit 1; }
  echo "=== crash rc=$rc, retry in 30s ==="; sleep 30
done
