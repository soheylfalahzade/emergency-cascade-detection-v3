#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env || exit 1
R=~/Portfolio/Smart_Traffic_System; S=$R/stacking; RES=$R/results_v3; LG=$RES/post_logs
cd $S || exit 1
[ -f $RES/protocol_frozen.json ] || { echo "no frozen protocol; run post_train.sh first"; exit 1; }
[ -f $RES/holdout_done.flag ] && { echo "holdout already evaluated (delete the flag knowingly to repeat)"; exit 1; }
BOOT=${BOOT:-100} python holdout_final.py > $LG/holdout.log 2>&1 || { echo "=== HOLDOUT FAILED"; tail -15 $LG/holdout.log; exit 1; }
python make_report.py > $LG/report.log 2>&1 || { echo "=== REPORT FAILED"; tail -15 $LG/report.log; exit 1; }
cat $RES/final_report.md
