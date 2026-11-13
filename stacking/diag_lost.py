import numpy as np
from collections import Counter
import image_cascade_final as ic
import crop_cascade as cc

d = ic.dev_data()
HI = 0.74
cat, groups = Counter(), set()
for f in range(5):
    off = sum(len(d["L"][g]) for g in range(f))
    for i, r in enumerate(d["L"][f]):
        gb, gc = d["GT"][f][i]
        if not d["hh"][f][i].any() or bool(((r["conf"] >= ic.LO) & (r["conf"] < HI)).any()):
            continue
        lost = d["hh"][f][i] & ~cc.hit(gb, gc, *d["LD"][f][i], ic.TAU)
        if not lost.any():
            continue
        groups.add(d["gid"][off + i])
        m = r["conf"] >= cc.MINC
        for q in np.where(lost)[0]:
            ok = cc.iou_mat(gb[q:q + 1], r["boxes"][m])[0] >= 0.5
            if not ok.any():
                cat["no light detection"] += 1
                continue
            j = np.where(ok)[0][np.argmax(r["conf"][m][ok])]
            c, cl = float(r["conf"][m][j]), int(r["cls"][m][j])
            cat[f"{cc.CLASSES[int(gc[q])]} -> {cc.CLASSES[cl]} (conf {'>=' if c >= HI else '<'} {HI if c >= HI else round(ic.LO, 2)})"] += 1
print("lost groups:", len(groups))
for k, v in cat.most_common():
    print(v, k)
