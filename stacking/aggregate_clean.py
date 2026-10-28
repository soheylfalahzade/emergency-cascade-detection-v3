import sys, os
import numpy as np, pandas as pd
R = os.path.expanduser("~/Portfolio/Smart_Traffic_System/results_clean")
model = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"
rows = []
for f in range(5):
    p = f"{R}/{model}_fold{f}/results.csv"
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    fit = (0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]).values
    b = len(fit) - 1 - int(np.argmax(fit[::-1]))
    r = df.iloc[b]
    rows.append(dict(fold=f, epochs_logged=len(df), best_epoch=b + 1,
                     P=r["metrics/precision(B)"], R=r["metrics/recall(B)"],
                     mAP50=r["metrics/mAP50(B)"], mAP50_95=r["metrics/mAP50-95(B)"]))
t = pd.DataFrame(rows)
print(t.round(4).to_string(index=False))
if len(t) >= 2:
    for c in ("P", "R", "mAP50", "mAP50_95"):
        print(f"{c}: {t[c].mean():.4f} ± {t[c].std(ddof=1):.4f}  (n_folds={len(t)})")
print("توجه: فولدی که هنوز در حال training است، عدد نهایی نیست.")
