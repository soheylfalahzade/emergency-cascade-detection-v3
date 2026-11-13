#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate traffic_env || exit 1
cd ~/frcnn60 || exit 1
export NW=${NW:-2}
R=~/Portfolio/Smart_Traffic_System/results_v3_frcnn60
for k in 0 1 2 3 4; do
  [ -f $R/fasterrcnn_fold$k/done_fold$k.txt ] && { echo "skip fold $k"; continue; }
  python train_frcnn60.py $k; rc=$?
  [ $rc -eq 42 ] && { echo "STOPPED by STOP_TRAINING"; exit 42; }
  [ $rc -ne 0 ] && { echo "FAILED fold $k rc=$rc"; exit $rc; }
done
echo "ALL FOLDS DONE $(date)"