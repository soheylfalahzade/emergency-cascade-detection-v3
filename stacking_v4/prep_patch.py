import re
from pathlib import Path
d = Path('.')


def sub(fn, old, new, count=1):
    t = (d / fn).read_text(encoding='utf-8')
    n = t.count(old)
    assert n == count, f"{fn}: expected {count} of {old!r}, found {n}"
    (d / fn).write_text(t.replace(old, new), encoding='utf-8')


for fn in ['dump_dets.py', 'bench_models.py', 'stack_eval.py', 'make_report.py', 'cascade_cv.py', 'freeze_protocol.py', 'holdout_final.py']:
    t = (d / fn).read_text(encoding='utf-8')
    assert 'results_v3' in t, fn
    (d / fn).write_text(t.replace('results_v3', 'results_v4_frcnn60'), encoding='utf-8')
sub('freeze_protocol.py', 'R / "stacking"', 'R / "stacking_v4"')
sub('dump_dets.py', '''        p = RES / f"fasterrcnn_fold{fold}" / "train_log.txt"
        return p.exists() and "epoch 15/15 DONE" in p.read_text()''',
    '''        return (RES / f"fasterrcnn_fold{fold}" / f"done_fold{fold}.txt").exists() and (RES / f"fasterrcnn_fold{fold}" / "best.pt").exists()''')
sub('dump_dets.py', '"last.pt"', '"best.pt"')
sub('bench_models.py', 'last.pt', 'best.pt')
sub('make_report.py', 'Holdout (evaluated once; ',
    "Holdout (POST-HOC: this holdout had already been consumed by the earlier 15-epoch run; evaluated once under this run's frozen protocol; ")
sub('make_report.py', '"- best.pt/early-stop for YOLO and RT-DETR were selected on each fold\'s val; Faster R-CNN uses the last epoch.",',
    '"- best.pt/early-stop for YOLO, RT-DETR and the retrained Faster R-CNN (60 epochs, patience 15, fitness mAP50-95) were selected on each fold\'s val.",')
print('patches ok')