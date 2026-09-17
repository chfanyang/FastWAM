#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PYTHONPATH=src:. TMPDIR=/tmp OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python
output=evaluate_results/vlabench/vae_audit_3cam192_select_book_all382_20260916
mkdir -p "$output"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$gpu "$python" -u experiments/vlabench/audit_three_camera_vae_reconstruction.py \
    --config runs/vlabench_select_book_rothko_3cam192_wan21/2026-09-15_17-33-31/config.yaml \
    --output "$output/shard$gpu" --all-windows --shard-rank "$gpu" --num-shards 8 \
    > "$output/shard$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" != 0 ]; then exit 1; fi
"$python" - <<'PY'
import json
from pathlib import Path
from experiments.vlabench.audit_three_camera_vae_reconstruction import summarize
p=Path('evaluate_results/vlabench/vae_audit_3cam192_select_book_all382_20260916')
rows=[]
for i in range(8): rows.extend(json.loads((p/f'shard{i}/windows.json').read_text()))
rows.sort(key=lambda r:r['val_index'])
assert [r['val_index'] for r in rows]==list(range(382))
summary=json.loads((p/'shard0/summary.json').read_text())
summary.update(windows=len(rows),complete=True,num_shards=8)
summary.pop('shard_rank')
for label,n in [('first8',8),('all16',16)]:
 summary[label]={phase:summarize(rows,phase,n) for phase in
   ('direct_vs_target','bf16_pixels_vs_direct','vae_vs_direct','vae_vs_target')}
(p/'windows.json').write_text(json.dumps(rows,indent=2))
(p/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['first8'],indent=2))
PY
