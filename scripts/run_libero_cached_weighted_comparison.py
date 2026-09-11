"""Launch eight independent cached-suite comparisons, one per physical GPU."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'evaluate_results/libero_decoder_offline'


def main():
    jobs = []
    for seed in (20260909, 20260910):
        for suite in ('libero_10', 'libero_goal', 'libero_object', 'libero_spatial'):
            cache = BASE / f'vae7498_train800_seed{seed}' / suite / 'encoded_training_windows.pt'
            out = BASE / f'vae7498_weightedJoint_v1_train800_seed{seed}' / suite
            assert cache.is_file(), cache
            assert not out.exists(), out
            jobs.append((cache, out))
    children = []
    for gpu, (cache, out) in enumerate(jobs):
        out.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='2',
                   NUMEXPR_NUM_THREADS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        log = (out.parent / f'{out.name}.log').open('x')
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/compare_libero_cached_weighted_decoder.py'),
                     '--cache', str(cache), '--output-dir', str(out)], cwd=ROOT, env=env,
                     stdout=log, stderr=subprocess.STDOUT)
        children.append((process, log, gpu))
        print(f'GPU{gpu} pid={process.pid} output={out}', flush=True)
    failures = []
    for process, log, gpu in children:
        code = process.wait()
        log.close()
        print(f'GPU{gpu} exit={code}', flush=True)
        if code:
            failures.append(gpu)
    if failures:
        raise RuntimeError(f'Failed shards: {failures}')
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_libero_decoder_samples.py'),
                    *[str(BASE / f'vae7498_weightedJoint_v1_train800_seed{s}') for s in (20260909, 20260910)],
                    '--output', str(BASE / 'vae7498_weightedJoint_v1_train1600_summary.json')], check=True)


if __name__ == '__main__':
    main()
