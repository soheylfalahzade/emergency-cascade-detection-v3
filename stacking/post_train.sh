#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env || exit 1
R=~/Portfolio/Smart_Traffic_System; S=$R/stacking; RES=$R/results_v3; ST=$RES/post_state; LG=$RES/post_logs
mkdir -p $ST $LG; cd $S || exit 1
step() { n=$1; shift
  [ -f $ST/$n.done ] && { echo "skip $n"; return; }
  echo "=== START $n $(date +%H:%M)"; "$@" > $LG/$n.log 2>&1; rc=$?
  [ $rc -ne 0 ] && { echo "=== FAILED $n rc=$rc (see $LG/$n.log)"; tail -5 $LG/$n.log; exit 1; }
  touch $ST/$n.done; echo "=== DONE $n"; }
miss=0
while true; do
  if pgrep -f "run_heavy_v3|run_frcnn_fold|train_rtdetr_v3|train_yolo" > /dev/null; then miss=0; sleep 600; continue; fi
  if python check_complete.py > $LG/check.log 2>&1; then break; fi
  miss=$((miss+1))
  [ $miss -ge 3 ] && { echo "=== training not running and incomplete:"; cat $LG/check.log; exit 2; }
  sleep 60
done
echo "=== all 15 models complete $(date +%H:%M)"
for m in yolov8n rtdetr_l fasterrcnn; do step dets_$m env DEVICE=0 python dump_dets.py $m 0 1 2 3 4 --splits val holdout; done
step bench env FOLD=0 python bench_models.py
step cv python stack_eval.py cv --models yolov8n rtdetr_l fasterrcnn --metas LR HGB RF SVM MLP KNN
step cascade_cv python cascade_cv.py
step freeze python freeze_protocol.py
echo "=== POST-TRAIN DONE. Review CV + frozen protocol, then run run_holdout_once.sh"
tail -30 $LG/cv.log; tail -25 $LG/cascade_cv.log; cat $RES/protocol_frozen.json | head -30
