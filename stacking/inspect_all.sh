#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env
cd ~/Portfolio/Smart_Traffic_System
{
echo "##### nproc"; nproc
echo "##### data.yaml fold0"; cat datasets/folds/fold_0/data.yaml
echo "##### fold_0 tree"; ls datasets/folds/fold_0; ls datasets/folds/fold_0/val | head
echo "##### counts"; for s in train val; do echo $s imgs: $(ls datasets/folds/fold_0/$s/images 2>/dev/null | wc -l) labels: $(ls datasets/folds/fold_0/$s/labels 2>/dev/null | wc -l); done
echo "##### sample val names"; ls datasets/folds/fold_0/val/images | shuf -n 15 --random-source=<(yes)
echo "##### aug-like names in val"; ls datasets/folds/fold_0/val/images | grep -ciE "flip|noise|sharp|aug"
echo "##### datasets dir"; ls datasets | head -50
echo "##### scripts mentioning fasterrcnn"; grep -rlE "fasterrcnn|FasterRCNN" --include=*.py . 2>/dev/null | head
echo "##### results dir"; ls results
echo "##### holdout/video candidates"; ls datasets | grep -iE "holdout|test|video"; find ~ -maxdepth 3 \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | head
echo "##### frcnn checkpoint keys"
python - << 'PY'
import torch
for n in ("last.pt","checkpoint.pt"):
    ck = torch.load(f"results/fasterrcnn_fold0/{n}", map_location="cpu", weights_only=False)
    print(n, type(ck))
    if isinstance(ck, dict):
        ks = list(ck.keys()); print(" top keys:", ks[:15])
        sd = ck.get("model_state_dict", ck.get("model", ck))
        if isinstance(sd, dict):
            k2 = list(sd.keys()); print(" n_params_tensors:", len(k2), "first:", k2[:3], "last:", k2[-3:])
PY
} 2>&1 | tee ~/inspect_out.txt
