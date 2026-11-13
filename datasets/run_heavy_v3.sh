#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env || exit 1
R=~/Portfolio/Smart_Traffic_System; cd $R/datasets || exit 1
rm -f ~/STOP_TRAINING
FOLD_LIST="${FOLD_LIST:-0 1 2}"
prog() { cat $R/results_v3/fasterrcnn_fold*/train_log.txt $R/results_v3/rtdetr_l_fold*/results.csv 2>/dev/null | wc -l; }
stage() {
  label=$1; shift; fails=0; prev=$(prog)
  while true; do
    [ -f ~/STOP_TRAINING ] && { echo "=== STOP requested before $label"; exit 42; }
    "$@"; rc=$?
    case $rc in 0) echo "=== $label done"; return 0;; 3) echo "=== $label NEEDS DECISION"; exit 3;; 42) echo "=== STOPPED in $label"; exit 42;; esac
    now=$(prog)
    if [ "$now" -gt "$prev" ]; then fails=0; else fails=$((fails+1)); fi
    prev=$now
    [ $fails -ge 5 ] && { echo "=== GAVE UP in $label"; exit 1; }
    echo "=== $label crashed rc=$rc, retry in 30s"; sleep 30
  done
}
for f in $FOLD_LIST; do
  stage "frcnn fold$f" python run_frcnn_fold.py $f
  stage "rtdetr fold$f" env FOLDS=$f python train_rtdetr_v3.py
done
echo "=== ALL REQUESTED FOLDS DONE"
