#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PYTHONPATH=src:. TMPDIR=/tmp TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
# Only wait on this specific validation job, never terminate another workload.
/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python -u - <<'PY'
import json,time
from pathlib import Path
root=Path('evaluate_results/vlabench/vlabench_select_book_rothko_3cam192_wan21/heldout382_4ckpt_vae562')
files=[root/f'step{s:06d}_shard{r}.json' for s in (843,1405,1686,2529) for r in (0,1)]
print('Waiting for all eight validation shards to complete.',flush=True)
while not all(p.exists() and json.loads(p.read_text()).get('complete') for p in files):
    time.sleep(15)
for s in (843,1405,1686,2529):
    rows=[r for p in files if p.name.startswith(f'step{s:06d}_') for r in json.loads(p.read_text())['windows']]
    assert sorted(r['val_index'] for r in rows)==list(range(382))
print('Validation complete; starting robust_joint evaluation.',flush=True)
time.sleep(5)
PY
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam_vlabench/bin/python -u experiments/vlabench/run_select_book_manager.py \
 --checkpoint runs/vlabench_select_book_rothko_3cam192_wan21/2026-09-15_17-33-31/checkpoints/weights/step_002529.pt \
 --output-dir evaluate_results/vlabench/vlabench_select_book_rothko_3cam192_wan21/ckpt002529_vaeDecoderStep562_replan8_robustJointAnchor0_ep50 \
 --vae-safetensors-path runs/vlabench_select_book_vae_wan21_3cam192_bs2_ga4_lr1e-5_ep1/Wan2.1_VAE_vlabench_select_book_step000562.safetensors \
 --allow-vae-mismatch --decode-mode robust_joint --gpu-ids 0 1 2 3 4 5 6 7 \
 --workers-per-gpu 4 --episodes 50 --replan-steps 8 --gripper-threshold 0.5 --threads 2
