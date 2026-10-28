import sys, cv2
from pathlib import Path
from ultralytics import YOLO
F = Path.home() / "Portfolio/Smart_Traffic_System/datasets/folds_v5/fold_0/train"
coco = YOLO("yolov8m.pt")
TILE, STRIDE, REP, MAXF = 1280, 1000, 3, 60
def grid(L):
    m = max(L - TILE, 0); g = list(range(0, m + 1, STRIDE))
    if g[-1] != m: g.append(m)
    return g
ni = nb = 0
for v in sys.argv[1:]:
    cap = cv2.VideoCapture(v); i = used = 0
    while used < MAXF:
        ok, fr = cap.read()
        if not ok: break
        i += 1
        if i % 10: continue
        used += 1
        H, W = fr.shape[:2]
        for y0 in grid(H):
            for x0 in grid(W):
                t = fr[y0:y0 + TILE, x0:x0 + TILE]; h, w = t.shape[:2]
                r = coco.predict(t, conf=0.4, imgsz=1280, classes=[2, 5, 7], device=0, verbose=False)[0]
                L = [f"3 {(a+c)/2/w:.6f} {(b+d)/2/h:.6f} {(c-a)/w:.6f} {(d-b)/h:.6f}" for a, b, c, d in r.boxes.xyxy.cpu().numpy()]
                if not L: continue
                for k in range(REP):
                    n = f"vid_{Path(v).stem}_{i}_{y0}_{x0}_r{k}"
                    cv2.imwrite(str(F / "images" / f"{n}.jpg"), t)
                    (F / "labels" / f"{n}.txt").write_text("\n".join(L) + "\n")
                    ni += 1; nb += len(L)
print("added imgs", ni, "car boxes", nb)
