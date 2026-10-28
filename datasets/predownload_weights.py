from pathlib import Path
import torch
from ultralytics import YOLO, RTDETR
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import resnet50, ResNet50_Weights

WORK = Path.home() / "Portfolio/Smart_Traffic_System/datasets"

print(">> yolov8n.pt")
YOLO(str(WORK / "yolov8n.pt") if (WORK / "yolov8n.pt").exists() else "yolov8n.pt")

print(">> rtdetr-l.pt")
RTDETR(str(WORK / "rtdetr-l.pt") if (WORK / "rtdetr-l.pt").exists() else "rtdetr-l.pt")

print(">> resnet50 backbone weights (torch hub cache)")
_ = resnet50(weights=ResNet50_Weights.DEFAULT)

print(">> fasterrcnn_resnet50_fpn_v2 full weights (torch hub cache)")
_ = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

print("ALL WEIGHTS CACHED LOCALLY — no internet needed from here on.")
