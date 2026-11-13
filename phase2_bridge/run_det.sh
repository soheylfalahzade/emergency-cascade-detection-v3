#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate traffic_env
cd /mnt/c/Users/Admin/Documents/phase2_bridge
FT=$HOME/carla_ft_runs/carla_ft/weights/best.pt
P1=$HOME/Portfolio/Smart_Traffic_System/results_v3/yolov8n_fold0/weights/best.pt
test -f "$FT" || { echo "missing $FT"; exit 1; }
test -f "$P1" || { echo "missing $P1"; exit 1; }
python det_server.py --weights "$FT" --port 5601 --imgsz 960 --device cpu > v12/det_ft.log 2>&1 &
python det_server.py --weights "$P1" --port 5602 --imgsz 960 --device cpu > v12/det_p1.log 2>&1 &
wait