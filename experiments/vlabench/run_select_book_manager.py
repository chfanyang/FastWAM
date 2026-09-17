"""Configurable workers per GPU, disjoint Track-1 episodes; no tmux required."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def partition(episodes, gpu_ids, workers_per_gpu=1):
    if episodes < 1 or workers_per_gpu < 1 or not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Positive episode/worker count and distinct GPU IDs required')
    slots = list(gpu_ids) * workers_per_gpu
    return [(gpu, list(range(rank, episodes, len(slots))))
            for rank, gpu in enumerate(slots) if rank < episodes]


def summarize(output, assignments, episodes):
    records = {}
    errors = []
    for rank, (gpu, ids) in enumerate(assignments):
        worker = output / f'worker{rank}_gpu{gpu}'
        for episode in ids:
            path = worker / f'episode_{episode:03d}.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result['episode'] != episode or episode in records:
                    raise ValueError(f'Invalid/duplicate episode record: {path}')
                records[episode] = dict(result, worker_dir=str(worker.relative_to(output)))
            error_path = worker / f'episode_{episode:03d}.error.json'
            if error_path.exists():
                errors.append(json.loads(error_path.read_text()))
    completed = len(records)
    successes = sum(bool(r['success']) for r in records.values())
    complete = completed == episodes and not errors
    return dict(expected=episodes, completed=completed, successes=successes,
                complete=complete, missing=sorted(set(range(episodes)) - records.keys()),
                success_rate=successes / episodes if complete else None,
                completed_success_rate=successes / completed if completed else None,
                intention_score=sum(r['intention_score'] for r in records.values()) / completed if completed else None,
                progress_score=sum(r['progress_score'] for r in records.values()) / completed if completed else None,
                errors=errors, episodes=[records[i] for i in sorted(records)])


def stop_workers(processes):
    # Each child owns a process group; never kill unrelated training/evaluation.
    for p in processes:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for p in processes:
        try:
            p.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vae-safetensors-path', type=Path)
    parser.add_argument('--allow-vae-mismatch', action='store_true')
    parser.add_argument('--decode-mode', choices=['legacy', 'robust_joint'], default='legacy')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=list(range(8)),
                        help='Physical GPU IDs; each child sees only its assigned GPU.')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--workers-per-gpu', type=int, default=1)
    parser.add_argument('--replan-steps', type=int, required=True)
    parser.add_argument('--gripper-threshold', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.vae_safetensors_path is not None:
        args.vae_safetensors_path = args.vae_safetensors_path.resolve()
        if not args.vae_safetensors_path.is_file():
            raise FileNotFoundError(args.vae_safetensors_path)
    track_path = ROOT / 'third_party/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json'
    available = len(json.loads(track_path.read_text())['select_book'])
    if not 1 <= args.episodes <= available or args.threads < 1 or any(g < 0 for g in args.gpu_ids):
        raise ValueError('Invalid episode count, threads or GPU IDs')
    if not 1 <= args.replan_steps <= 16 or not 0 <= args.gripper_threshold <= 1:
        raise ValueError('Expected replan in [1,16] and gripper threshold in [0,1]')
    assignments = partition(args.episodes, args.gpu_ids, args.workers_per_gpu)
    commands = []
    for rank, (gpu, ids) in enumerate(assignments):
        worker_dir = args.output_dir / f'worker{rank}_gpu{gpu}'
        cmd = [sys.executable, '-u', str(ROOT / 'experiments/vlabench/eval_select_book.py'),
               '--checkpoint', str(args.checkpoint), '--output-dir', str(worker_dir),
               '--episodes', str(args.episodes), '--episode-ids', *map(str, ids),
               '--replan-steps', str(args.replan_steps),
               '--gripper-threshold', str(args.gripper_threshold), '--seed', str(args.seed)]
        if args.vae_safetensors_path:
            cmd.extend(['--vae-safetensors-path', str(args.vae_safetensors_path)])
        if args.allow_vae_mismatch:
            cmd.append('--allow-vae-mismatch')
        cmd.extend(['--decode-mode', args.decode_mode])
        commands.append(cmd)
        print(f'Worker {rank}, GPU {gpu}: episodes={ids}', flush=True)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Refusing nonempty output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = dict(checkpoint=str(args.checkpoint), episodes=args.episodes,
                gpu_ids=args.gpu_ids, workers_per_gpu=args.workers_per_gpu,
                threads=args.threads, seed=args.seed,
                replan_steps=args.replan_steps, gripper_threshold=args.gripper_threshold,
                assignments=assignments, commands=commands)
    (args.output_dir / 'manager_config.json').write_text(json.dumps(plan, indent=2))
    processes = []
    failed = False
    def terminate_handler(signum, frame):
        raise KeyboardInterrupt('Manager received termination signal')
    previous_handler = signal.signal(signal.SIGTERM, terminate_handler)
    try:
        for rank, ((gpu, _), cmd) in enumerate(zip(assignments, commands)):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl',
                       MUJOCO_EGL_DEVICE_ID=str(gpu), TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1')
            for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                env[key] = str(args.threads)
            with (args.output_dir / f'worker{rank}_gpu{gpu}.log').open('w') as log:
                processes.append(subprocess.Popen(cmd, cwd=ROOT, env=env,
                                                  stdout=log, stderr=subprocess.STDOUT,
                                                  start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                failed = True
                break
            time.sleep(2)
        failed = failed or any(p.poll() not in (None, 0) for p in processes)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        stop_workers(processes)
        summary = summarize(args.output_dir, assignments, args.episodes)
        summary['worker_returncodes'] = [p.returncode for p in processes]
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f"Completed {summary['completed']}/{args.episodes}, successes={summary['successes']}", flush=True)
    if failed or not summary['complete']:
        raise RuntimeError(f'Evaluation incomplete; inspect {args.output_dir}/summary.json and worker logs')


if __name__ == '__main__':
    main()
