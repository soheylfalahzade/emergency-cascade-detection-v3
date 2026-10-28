#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env || exit 1
R=~/Portfolio/Smart_Traffic_System; S=$R/stacking; LOG=$R/results_v3/heavy.log
until grep -q "=== rtdetr fold1 done" $LOG; do sleep 120; done
[ -f ~/STOP_TRAINING ] && { echo "STOP file present, abort"; exit 0; }
pkill -f run_heavy_v3.sh; pkill -f run_frcnn_fold.py; pkill -f train_rtdetr_v3.py
sleep 20
cd $S
for m in yolov8n rtdetr_l fasterrcnn; do DEVICE=0 python dump_dets.py $m 0 1 --splits val 2>&1 | grep -v -i warning | tail -3; done
ls $R/results_v3/dets/
python compare_single3.py 0
python compare_single3.py 1
python stack_eval.py cv --models yolov8n rtdetr_l fasterrcnn > $R/results_v3/stack_cv_2fold.txt 2>&1
tail -25 $R/results_v3/stack_cv_2fold.txt
FOLD_LIST="0 1 2 3 4" nohup $R/datasets/run_heavy_v3.sh >> $LOG 2>&1 &
echo "training restarted"
