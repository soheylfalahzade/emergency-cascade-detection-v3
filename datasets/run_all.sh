#!/bin/bash
LOG=~/Portfolio/Smart_Traffic_System/results/run_all.log
cd ~/Portfolio/Smart_Traffic_System/datasets

echo "STEP: YOLOv8n (resumable, 5 folds)" | tee -a $LOG
python train_yolo_rtdetr.py yolo 2>&1 | tee -a $LOG

echo "STEP: RT-DETR-l (resumable, 5 folds)" | tee -a $LOG
python train_yolo_rtdetr.py rtdetr 2>&1 | tee -a $LOG

echo "STEP: Faster R-CNN (resumable, 5 folds)" | tee -a $LOG
python train_fasterrcnn.py 2>&1 | tee -a $LOG

echo "ALL DONE" | tee -a $LOG
