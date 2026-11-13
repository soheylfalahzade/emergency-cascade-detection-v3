import json, pickle
import numpy as np, torch, torchvision
import crop_cascade as cc

P = json.load(open(cc.RES / "crop_protocol_frozen.json"))
lo, hc, hi_img, m, s, tau = P["lo"], P["hi_crop"], P["hi_image"], P["margin"], P["size"], P["tau"]
keys = ("map50", "emerg_map50", "ap50_ambulance", "ap50_police", "ap50_firetruck", "ap50_car", "emerg_recall")


def union(rg, row):
    b, c, l, rid, reps = rg
    B, S, L = [b], [c], [l]
    for q, (hb, hs, hl) in row["heavy"].items():
        if not (lo <= c[reps[q]] < hc):
            continue
        x1, y1, x2, y2 = b[reps[q]]
        cx, cy = (hb[:, 0] + hb[:, 2]) / 2, (hb[:, 1] + hb[:, 3]) / 2
        k = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        B.append(hb[k]); S.append(hs[k]); L.append(hl[k])
    B, S, L = np.concatenate(B), np.concatenate(S), np.concatenate(L)
    if len(B):
        kk = torchvision.ops.batched_nms(torch.from_numpy(B.astype(np.float32)), torch.from_numpy(S.astype(np.float32)),
                                         torch.from_numpy(L.astype(np.int64)), 0.5).numpy()[:100]
        B, S, L = B[kk], S[kk], L[kk]
    return B.astype(np.float32).reshape(-1, 4), S.astype(np.float32), L.astype(np.int64)


acc = {}
for f in range(5):
    L = cc.load(cc.LIGHT, f, "val"); H = cc.load(cc.HEAVY, f, "val")
    rows = pickle.load(open(cc.CC / f"val_m{m}_s{s}_fold{f}.pkl", "rb"))
    RG = [cc.regions(r) for r in L]
    GT = [(r["gt_boxes"], r["gt_cls"]) for r in L]
    LD = [cc.cf(r) for r in L]; HD = [cc.cf(r) for r in H]
    E = [bool(((r["conf"] >= lo) & (r["conf"] < hi_img)).any()) for r in L]
    systems = {"light_only": LD, "heavy_only": HD,
               "image_cascade": [HD[i] if E[i] else LD[i] for i in range(len(L))],
               "crop_replace": cc.crop_dets(RG, rows, hc, lo)[0],
               "crop_union": [union(RG[i], rows[i]) for i in range(len(L))]}
    for k, D in systems.items():
        r = cc.metrics(D, GT, tau)
        acc.setdefault(k, []).append([r.get(q, np.nan) for q in keys])
    print(f"fold{f} done", flush=True)
for k, v in acc.items():
    print(f"{k:<14}", dict(zip(keys, np.round(np.nanmean(np.array(v), 0), 4))))
