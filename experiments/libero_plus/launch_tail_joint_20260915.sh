#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cfy/FastWAM
case "${1:-}" in manipulation|nav) ;; *) exit 2 ;; esac
export PAIRED_MANIFEST_DIR=$ROOT/experiments/libero_plus/tail_joint_20260915
export PAIRED_OUTPUT_DIR=$ROOT/evaluate_results/libero_plus/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/ckpt021700_vaeDecoderStep7498_replan8_ensembleOff_robustJointAnchor0_tail_rebalanced_${1}_20260915
exec bash "$ROOT/experiments/libero_plus/launch_a800_paired_3090_20260913.sh" "$@"
