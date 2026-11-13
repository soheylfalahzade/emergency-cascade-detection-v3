#!/bin/bash
rm -f ~/STOP_TRAINING
cd ~/frcnn60 || exit 1
NW=${1:-2} nohup setsid bash run_frcnn60.sh >> ~/frcnn60/train.log 2>&1 &
sleep 2; echo "launched NW=${1:-2}"; pgrep -af run_frcnn60