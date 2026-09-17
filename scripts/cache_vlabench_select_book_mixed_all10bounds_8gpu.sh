#!/usr/bin/env bash
# Dedicated mixed-representation launch; never use the generic task default.
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=$PWD/checkpoints
export TMPDIR=$PWD/evaluate_results/vlabench/mixedtmp
mkdir -p "$TMPDIR"
task=vlabench_select_book_rothko_mixed_all10bounds_3cam192_wan21
cache=data/vlabench_select_book_rothko_mixed_all10bounds_3cam192_wan21_bf16_h16_latents
python - <<'CHECK'
import subprocess
used=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).split()]
assert len(used)==8 and all(v<128 for v in used), ('GPUs busy; refusing launch',used)
CHECK
control=$(mktemp -d "$PWD/evaluate_results/vlabench/mixed_all10_cache_XXXXXXXX")
printf '%s\n' "$control"
# Raw original VAE; first verify actual online/cache encoding and training loss.
python -u scripts/precompute_visual_action_latents.py \
  --task "$task" --output-dir "$control/smoke16" \
  --benchmark-max-samples 16 --samples-per-shard 8 \
  --batch-size 2 --num-workers 1 --log-every 1 model.load_text_encoder=false \
  > "$control/smoke16.log" 2>&1
# 8 GPUs x 4 independent VAE workers, batch2 and 1 loader worker per process.
python -u scripts/precompute_vlabench_task_latents.py \
  --task-config "$task" --task-name select_book --output "$cache" \
  --gpus 0,1,2,3,4,5,6,7 --processes-per-gpu 4 \
  --batch-size 2 --num-workers 1 --shard-size 256 \
  > "$control/cache.log" 2>&1
printf 'Mixed cache finished: %s; training has not been started.\n' "$cache"
