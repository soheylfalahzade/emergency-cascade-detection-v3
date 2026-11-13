import os, sys, time, json
sys.path.insert(0, os.path.expanduser("~/frcnn60"))
import torch
from torch.utils.data import DataLoader
import train_frcnn60 as T
dev = torch.device("cuda")
fp = T.BASE / "fold_0"
ds = T.TrainDS(fp / "train/images", fp / "train/labels")
model = T.build_model().to(dev); model.train()
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=5e-4)
scaler = torch.amp.GradScaler("cuda")
def step(imgs, tg):
    imgs = [x.to(dev) for x in imgs]; tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
    with torch.amp.autocast("cuda"): loss = sum(model(imgs, tg).values())
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
n_iter = len(ds) // T.BATCH
res = {"iters_per_epoch": n_iter}
b0 = next(iter(DataLoader(ds, batch_size=4, collate_fn=T.collate)))
for _ in range(10): step(*b0)
torch.cuda.synchronize(); t = time.time()
for _ in range(40): step(*b0)
torch.cuda.synchronize(); g = (time.time() - t) / 40
res["gpu_only_s_per_iter"] = g; res["gpu_only_epoch_h"] = g * n_iter / 3600; print(res, flush=True)
for nw in (2, 6, 12):
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=T.collate, num_workers=nw)
    it = iter(dl)
    for _ in range(10): step(*next(it))
    torch.cuda.synchronize(); t = time.time()
    for _ in range(60): step(*next(it))
    torch.cuda.synchronize(); s = (time.time() - t) / 60
    res[f"nw{nw}_s_per_iter"] = s; res[f"nw{nw}_epoch_h"] = s * n_iter / 3600
    print(nw, s, s * n_iter / 3600, flush=True); del it, dl
json.dump(res, open(os.path.expanduser("~/frcnn60/bench_speed.json"), "w"), indent=1)
print("BENCH DONE")