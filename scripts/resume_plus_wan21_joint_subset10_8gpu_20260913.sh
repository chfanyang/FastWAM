#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
OUT=evaluate_results/libero_plus/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/ckpt021700_vaeDecoderStep7498_replan8_ensembleOff_robustJointAnchor0_stratified10_seed42
BACKUP=$OUT/backup_before_8gpu_20260913
test ! -e "$BACKUP"
mkdir "$BACKUP"
for ITEM in manager_config.yaml tasks.jsonl worker_tasks worker_logs summary.json; do
  if [ -e "$OUT/$ITEM" ]; then cp -a "$OUT/$ITEM" "$BACKUP/"; fi
done
cp logs/libero_plus_wan21_robustjoint_subset10.log "$BACKUP/manager.log"
cp "$0" "$OUT/resume_8gpu_20260913.sh"
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
exec bash scripts/eval_libero_plus_wan21_cf05_robust_joint_subset10_4567.sh \
  'MULTIRUN.gpu_ids=[0,1,2,3,4,5,6,7]' MULTIRUN.num_gpus=8 MULTIRUN.workers_per_gpu=4 \
  > "$OUT/manager_resume_8gpu_20260913.log" 2>&1
