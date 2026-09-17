#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
clone_pid="${1:?Pass the existing conda-clone PID}"
export TMPDIR=/tmp
deadline=$((SECONDS + 3600))
while kill -0 "$clone_pid" 2>/dev/null; do
  if (( SECONDS > deadline )); then
    echo 'Clone still running after one hour; not installing into incomplete environment.' >&2
    exit 1
  fi
  sleep 10
done
prefix=/mnt/hwdata/cfy/miniconda3/envs/fastwam_vlabench
test -x "$prefix/bin/python"
test -f "$prefix/conda-meta/history"
"$prefix/bin/python" -m pip install \
  --src third_party \
  -r experiments/vlabench/requirements-eval.txt \
  -c experiments/vlabench/constraints-eval.txt \
  --report runs/vlabench_pip_install.json
"$prefix/bin/python" -m pip install --no-deps -e third_party/VLABench
"$prefix/bin/python" -m pip freeze > runs/vlabench_eval_environment_freeze.txt
MUJOCO_GL=egl PYTHONPATH=src:third_party/VLABench "$prefix/bin/python" - <<'PY'
import torch, numpy, mujoco, dm_control, open3d, mediapy
print('Model runtime:', torch.__version__, numpy.__version__)
from VLABench.evaluation.evaluator.base import Evaluator
print('VLABench evaluator import PASS; rendering/assets still require smoke test')
PY
echo ENV_SETUP_COMPLETE
