import os, sys, time, csv, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
for _p in (str(Path.home() / "eb_local"), str(Path.home() / "eb_local/numba_stub"), str(ROOT / "stacking")):
    if _p not in sys.path: sys.path.append(_p)
from stackcore import evaluate_map

BASE = ROOT / "datasets/folds_v3"
RESULTS = ROOT / os.environ.get("OUT_NAME", "results_v3_frcnn60")
CLASSES = ["ambulance", "police", "firetruck", "car"]
EPOCHS = int(os.environ.get("EPOCHS", 60)); PATIENCE = int(os.environ.get("PATIENCE", 15))
BATCH = 4; MAX_SIDE = 800; PRINT_EVERY = 200
SEED = int(os.environ.get("SEED", 0)); NW = int(os.environ.get("NW", 2))
LOGCOLS = ["epoch", "train_loss", "val_loss", "map50_95", "map50", "lr", "epoch_time_s", "eval_time_s", "best_epoch"]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _resize(img):
    w0, h0 = img.size
    s = min(1.0, MAX_SIDE / max(w0, h0))
    if s < 1.0:
        img = img.resize((max(1, int(w0 * s)), max(1, int(h0 * s))), Image.BILINEAR)
    return img, s, w0, h0


class TrainDS(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.lbl_dir = Path(lbl_dir); self.imgs = sorted(Path(img_dir).glob("*"))
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        p = self.imgs[idx]; lp = self.lbl_dir / (p.stem + ".txt")
        img, s, w0, h0 = _resize(Image.open(p).convert("RGB")); w, h = img.size
        boxes, labels = [], []
        if lp.exists():
            for line in open(lp):
                q = line.split()
                if not q: continue
                cid = int(q[0]); cx, cy, bw, bh = map(float, q[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]); labels.append(cid + 1)
        t = {"boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
             "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)}
        return F.to_tensor(img), t


class EvalDS(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.lbl_dir = Path(lbl_dir); self.imgs = sorted(Path(img_dir).glob("*"))
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        p = self.imgs[idx]; lp = self.lbl_dir / (p.stem + ".txt")
        img, s, w0, h0 = _resize(Image.open(p).convert("RGB"))
        a = np.loadtxt(lp, ndmin=2) if lp.exists() else np.zeros((0, 5))
        if a.size == 0:
            gb, gc = np.zeros((0, 4)), np.zeros(0, int)
        else:
            xc, yc, w, h = a[:, 1] * w0, a[:, 2] * h0, a[:, 3] * w0, a[:, 4] * h0
            gb = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1); gc = a[:, 0].astype(int)
        return F.to_tensor(img), (s, gb, gc)


def collate(b): return tuple(zip(*b))


def build_model():
    m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1)
    return m


def evaluate(model, loader, device):
    model.eval(); st, dp = model.roi_heads.score_thresh, model.roi_heads.detections_per_img
    model.roi_heads.score_thresh = 0.001; model.roi_heads.detections_per_img = 300
    dets, gts = [], []
    with torch.no_grad():
        for imgs, metas in loader:
            outs = model([i.to(device) for i in imgs])
            for o, (s, gb, gc) in zip(outs, metas):
                b = o["boxes"].cpu().numpy() / s; sc = o["scores"].cpu().numpy(); l = o["labels"].cpu().numpy() - 1
                k = sc >= 0.02; b, sc, l = b[k], sc[k], l[k]
                top = np.argsort(-sc)[:100]
                dets.append((b[top], sc[top], l[top])); gts.append((gb, gc))
    model.roi_heads.score_thresh, model.roi_heads.detections_per_img = st, dp
    model.train()
    r = evaluate_map(dets, gts)
    return r["map"], r["map50"]


def train_fold(fold):
    seed_all(SEED + fold)
    dev = torch.device("cuda")
    fp = BASE / f"fold_{fold}"
    run = RESULTS / f"fasterrcnn_fold{fold}"; run.mkdir(parents=True, exist_ok=True)
    done = run / f"done_fold{fold}.txt"
    if done.exists():
        print(f"fold {fold}: done already ({done.read_text().strip()})", flush=True); return
    tr = TrainDS(fp / "train/images", fp / "train/labels")
    va_loss_ds = TrainDS(fp / "val/images", fp / "val/labels")
    va_ds = EvalDS(fp / "val/images", fp / "val/labels")
    val_loader = DataLoader(va_loss_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=NW)
    eval_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=NW)

    model = build_model().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ck = run / "checkpoint.pt"; logp = run / "results.csv"
    start, best_fit, best_ep = 0, -1.0, 0
    if ck.exists():
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"]); sch.load_state_dict(c["scheduler"])
        scaler.load_state_dict(c["scaler"]); start = c["epoch"] + 1; best_fit = c["best_fit"]; best_ep = c["best_epoch"]
        print(f"fold {fold}: resume at epoch {start+1}, best_fit={best_fit:.4f} (epoch {best_ep})", flush=True)
    if not logp.exists():
        csv.writer(open(logp, "w", newline="")).writerow(LOGCOLS)
    json.dump(dict(seed=SEED + fold, epochs=EPOCHS, patience=PATIENCE, batch=BATCH, max_side=MAX_SIDE, lr0=0.005, momentum=0.9,
                   wd=5e-4, step=8, gamma=0.1, amp_train=True, fitness="map50_95 (stackcore.evaluate_map, conf>=0.02, top100)",
                   num_workers=NW), open(run / "config.json", "w"), indent=1)

    for ep in range(start, EPOCHS):
        gen = torch.Generator(); gen.manual_seed(SEED * 1000 + fold * 100 + ep)
        loader = DataLoader(tr, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=NW, generator=gen)
        model.train(); tl = 0.0; t0 = time.time(); nb = len(loader)
        for i, (imgs, tg) in enumerate(loader):
            imgs = [x.to(dev) for x in imgs]; tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
            with torch.amp.autocast("cuda"):
                loss = sum(model(imgs, tg).values())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item()
            if (i + 1) % PRINT_EVERY == 0 or i + 1 == nb:
                print(f"fold {fold} ep {ep+1}/{EPOCHS} b {i+1}/{nb} avg_loss={tl/(i+1):.4f} {time.time()-t0:.0f}s", flush=True)
        lr_used = opt.param_groups[0]["lr"]; sch.step(); t_train = time.time() - t0
        t1 = time.time(); vl = 0.0
        with torch.no_grad():
            model.train()
            for imgs, tg in val_loader:
                imgs = [x.to(dev) for x in imgs]; tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
                with torch.amp.autocast("cuda"):
                    vl += sum(model(imgs, tg).values()).item()
        vl /= max(len(val_loader), 1)
        m5095, m50 = evaluate(model, eval_loader, dev); t_eval = time.time() - t1
        if m5095 >= best_fit:
            best_fit, best_ep = m5095, ep + 1
            torch.save(model.state_dict(), run / "best.pt")
        torch.save(model.state_dict(), run / "last.pt")
        with open(logp, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, f"{tl/nb:.5f}", f"{vl:.5f}", f"{m5095:.5f}", f"{m50:.5f}", f"{lr_used:.6g}", f"{t_train:.0f}", f"{t_eval:.0f}", best_ep])
        print(f"fold {fold} ep {ep+1} DONE train={tl/nb:.4f} val_loss={vl:.4f} mAP50-95={m5095:.4f} mAP50={m50:.4f} best_ep={best_ep}", flush=True)
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sch.state_dict(), scaler=scaler.state_dict(),
                        epoch=ep, best_fit=best_fit, best_epoch=best_ep), ck)
        stop = None
        if (ep + 1) - best_ep >= PATIENCE: stop = f"early_stop epoch={ep+1} best_epoch={best_ep} best_map50_95={best_fit:.5f}"
        elif ep + 1 == EPOCHS: stop = f"max_epochs epoch={ep+1} best_epoch={best_ep} best_map50_95={best_fit:.5f}"
        if stop:
            done.write_text(stop + "\n"); print(f"fold {fold}: {stop}", flush=True); return
        if (Path.home() / "STOP_TRAINING").exists():
            print("=== STOP seen after epoch", ep + 1, flush=True); os._exit(42)


if __name__ == "__main__":
    for f in [int(x) for x in sys.argv[1:]] or range(5):
        train_fold(f)