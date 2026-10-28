from pathlib import Path
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image

BASE = Path.home() / "Portfolio/Smart_Traffic_System/datasets/folds"
RESULTS = Path.home() / "Portfolio/Smart_Traffic_System/results"
CLASSES = ["ambulance", "police", "firetruck", "car"]
K = 5
EPOCHS = 15
BATCH = 4
MAX_SIDE = 800
PRINT_EVERY = 20

class YoloDetDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.imgs = sorted(self.img_dir.glob("*"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size

        # cap the largest side at MAX_SIDE before anything touches the GPU
        scale = min(1.0, MAX_SIDE / max(w0, h0))
        if scale < 1.0:
            img = img.resize((max(1, int(w0 * scale)), max(1, int(h0 * scale))), Image.BILINEAR)
        w, h = img.size

        boxes, labels = [], []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    cid = int(parts[0]); cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
                    boxes.append([x1, y1, x2, y2]); labels.append(cid + 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }
        return F.to_tensor(img), target

def collate_fn(batch):
    return tuple(zip(*batch))

def build_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    return model

def train_fold(fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_path = BASE / f"fold_{fold}"
    train_ds = YoloDetDataset(fold_path / "train" / "images", fold_path / "train" / "labels")
    val_ds = YoloDetDataset(fold_path / "val" / "images", fold_path / "val" / "labels")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn, num_workers=2, persistent_workers=True)

    run_dir = RESULTS / f"fasterrcnn_fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    log_path = run_dir / "train_log.txt"

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"fold {fold}: resuming from epoch {start_epoch}", flush=True)

    if start_epoch >= EPOCHS:
        print(f"fold {fold}: already complete ({start_epoch}/{EPOCHS}), skipping", flush=True)
        return

    n_batches = len(train_loader)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if (i + 1) % PRINT_EVERY == 0 or (i + 1) == n_batches:
                elapsed = time.time() - t0
                print(f"fold {fold} epoch {epoch+1}/{EPOCHS} batch {i+1}/{n_batches} "
                      f"loss={loss.item():.4f} avg_loss={total_loss/(i+1):.4f} elapsed={elapsed:.0f}s", flush=True)
        scheduler.step()

        with torch.no_grad():
            model.train()
            val_loss = 0.0
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss_dict = model(imgs, targets)
                    val_loss += sum(loss_dict.values()).item()

        msg = f"fold {fold} epoch {epoch+1}/{EPOCHS} DONE train_loss={total_loss/max(n_batches,1):.4f} val_loss={val_loss/max(len(val_loader),1):.4f}"
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }, ckpt_path)
        torch.save(model.state_dict(), run_dir / "last.pt")

if __name__ == "__main__":
    for fold in range(K):
        train_fold(fold)
