import numpy as np, pandas as pd, cv2
from pathlib import Path
from collections import Counter
import crop_cascade as cc
from stack_eval import aligned

G, THR = "731", 0.25
man = pd.read_csv(cc.ROOT / "datasets/folds_v3_manifest.csv"); gmap = dict(zip(man["id"], man["group"]))
st = man.groupby(["set", "group"]).size().reset_index(name="n")
for s_, x in st.groupby("set"):
    print(s_, "groups", len(x), "images", int(x["n"].sum()), "largest", x.sort_values("n", ascending=False).head(5)[["group", "n"]].values.tolist())
pm = {f: aligned(cc.MODELS, f, "holdout") for f in range(5)}
names = [r["image"] for r in pm[0]["yolov8n"]]
gid = np.array([str(gmap.get(Path(n).stem, "")) for n in names])
sel = np.where(gid == G)[0]
oth = [i for i in range(len(names)) if gid[i] != G and (pm[0]["yolov8n"][i]["gt_cls"] == 2).any()]
print(f"holdout images {len(names)} | group {G}: {len(sel)} | GT classes in group:",
      dict(Counter(int(c) for i in sel for c in pm[0]["yolov8n"][i]["gt_cls"])))


def taxonomy(model):
    cnt = Counter()
    for f in range(5):
        for i in sel:
            r = pm[f][model][i]; gb, gc = r["gt_boxes"], r["gt_cls"]; k = r["conf"] >= THR
            for q in np.where(gc == 2)[0]:
                if not k.any():
                    cnt["no detection"] += 1; continue
                I = cc.iou_mat(gb[q:q + 1], r["boxes"][k])[0]; c = r["cls"][k]
                if ((I >= 0.5) & (c == 2)).any(): cnt["detected"] += 1
                elif ((I >= 0.5) & (c != 2)).any(): cnt["wrong class (IoU>=0.5)"] += 1
                elif ((I >= 0.1) & (c == 2)).any(): cnt["poor localisation, same class"] += 1
                else: cnt["no detection"] += 1
    tot = sum(cnt.values())
    return {k: round(v / tot, 3) for k, v in cnt.most_common()}, tot


for m in ("yolov8n", "rtdetr_l"):
    print(m, taxonomy(m))


def rel_area(ix):
    a = []
    for i in ix:
        r = pm[0]["yolov8n"][i]
        a += [float((b[2] - b[0]) * (b[3] - b[1]) / (r["W"] * r["H"])) for b, c in zip(r["gt_boxes"], r["gt_cls"]) if c == 2]
    return np.percentile(a, [25, 50, 75]).round(4).tolist(), len(a)


print("firetruck relative box area (q25, median, q75), n:  group", G, rel_area(sel), "| other groups", rel_area(oth))
out = cc.RES / "diag"; out.mkdir(exist_ok=True)
pick = np.random.RandomState(0).choice(sel, min(12, len(sel)), replace=False); tiles = []
for i in pick:
    im = cv2.imread(str(cc.img_dir(0, "holdout") / names[i])); r = pm[0]["yolov8n"][i]; t = max(2, im.shape[1] // 300)
    for b in r["gt_boxes"]:
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), t)
    for b, s_ in zip(r["boxes"], r["conf"]):
        if s_ >= THR:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), t)
    tiles.append(cv2.resize(im, (320, 240)))
while len(tiles) < 12:
    tiles.append(np.zeros((240, 320, 3), np.uint8))
sheet = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)])
cv2.imwrite(str(out / "group731_montage.jpg"), sheet); print("saved", out / "group731_montage.jpg (green=GT, red=YOLO fold0 conf>=0.25)")
