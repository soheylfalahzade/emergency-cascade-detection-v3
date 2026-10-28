#!/bin/bash
# Safe to run during training (CPU only, low priority). Re-run after post_train.sh for the final versions.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate traffic_env || exit 1
R=~/Portfolio/Smart_Traffic_System; F=$R/figures_pipeline; P=$R/phase2_bridge
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
run() { echo "=== $*"; nice -n 19 taskset -c 0-5 "$@" 2>&1 | grep -v -i warning; }
cd $F
for s in fig_diagrams fig_p1_main fig_p1_cascade fig_p1_dets fig_p1_data fig_p1_gallery; do run python $s.py; done
cd $P
run python test_toy_sim.py
for s in fig_p2_fuzzy fig_p2_scenarios; do run python $s.py; done
run python fig_p2_sweeps.py ${QUICK:+--quick}
run python anim_p2.py
echo "=== DONE. Phase 1 figures: $R/results_v3/figures | Phase 2: $P/outputs"
ls $R/results_v3/figures $P/outputs
