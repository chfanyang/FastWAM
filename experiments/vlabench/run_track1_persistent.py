"""Track1 shared episode queue with one model load per persistent GPU worker."""
import argparse
from collections import Counter
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / 'third_party/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json'


def slot_gpus(gpus, workers):
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0 or workers < 1:
        raise ValueError('Distinct nonnegative GPUs and positive worker count required')
    return list(gpus) * workers


def jobs_for(tasks, episodes, track):
    if not tasks or len(tasks) != len(set(tasks)) or episodes < 1:
        raise ValueError('Distinct tasks and positive episode count required')
    if any(t not in track or len(track[t]) < episodes for t in tasks):
        raise ValueError('Task or episode outside Track1')
    return [(t, e) for e in range(episodes) for t in tasks]


def inherited_results(folder, jobs, args):
    """Only accept complete episode records with the same evaluation protocol."""
    records = {}
    if folder is None:
        return records
    for task, episode in jobs:
        dest = folder / task / f'episode_{episode:03d}'
        result_file = dest / f'episode_{episode:03d}.json'
        if not result_file.exists():
            continue
        identity = json.loads((dest / 'evaluation_config.json').read_text())
        expected = dict(task=task, seed=args.seed, replan_steps=args.replan_steps,
                        gripper_threshold=args.gripper_threshold, decoder=args.decode_mode,
                        vae=str(args.vae_safetensors_path.resolve()) if getattr(args, 'vae_safetensors_path', None) else 'original', max_substeps=1)
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Evaluation protocol mismatch: {dest}')
        if Path(identity['checkpoint']).resolve() != args.checkpoint.resolve():
            raise ValueError(f'Checkpoint mismatch: {dest}')
        if (dest / f'episode_{episode:03d}.error.json').exists():
            raise ValueError(f'Both result and error exist: {dest}')
        result = json.loads(result_file.read_text())
        if result['episode'] != episode or result.get('task') != task:
            raise ValueError(f'Episode identity mismatch: {dest}')
        records[(task, episode)] = dict(task=task, episode=episode, result=result,
                                        output=str(dest), inherited=True)
    return records


def consume_jobs(jobs, runner, notify, recoverable=(), on_error=None):
    """The returned runtime stays in this process across all tasks and episodes."""
    runtime = None
    while True:
        job = jobs.get()
        if job is None:
            return
        notify('started', job)
        try:
            runtime = runner(job, runtime)
        except recoverable as error:
            on_error(job, error)
        else:
            notify('completed', job)


def resume_status(path, all_jobs, args):
    """Follow saved record locations across resumes, validating every identity."""
    saved = json.loads(path.read_text())
    records = {}
    for record in saved['records']:
        key = (record['task'], record['episode'])
        if key not in all_jobs or key in records:
            raise ValueError(f'Invalid resumed result: {key}')
        records.update(inherited_results(Path(record['output']).parents[1], [key], args))
        if key not in records:
            raise ValueError(f'Missing resumed result: {key}')
    errors = saved.get('errors', [])
    if errors and not args.continue_on_physics_error:
        raise ValueError('Resuming recorded physics errors requires explicit opt-in')
    for error in errors:
        if 'dm_control.rl.control.PhysicsError:' not in error.get('traceback', ''):
            raise ValueError('Only PhysicsError can be isolated')
        key = tuple(error['job'])
        if key not in all_jobs or key in records:
            raise ValueError(f'Invalid resumed error: {key}')
    # Error-only episodes have no result identity: check the originating run too.
    previous = json.loads((path.parent / 'plan.json').read_text())['options']
    for key in ('checkpoint', 'vae_safetensors_path', 'seed', 'replan_steps',
                'gripper_threshold', 'decode_mode', 'episodes'):
        value = getattr(args, key)
        if isinstance(value, Path): value = str(value.resolve())
        if previous.get(key) != value:
            raise ValueError(f'Resume protocol mismatch: {key}')
    return records, errors


def worker(slot, gpu, jobs, events, options):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl',
                      MUJOCO_EGL_DEVICE_ID=str(gpu), TOKENIZERS_PARALLELISM='false')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = str(options['threads'])
    out = Path(options['output_dir'])
    log = os.open(out / f'worker{slot}_gpu{gpu}.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.dup2(log, 1); os.dup2(log, 2); os.close(log)
    sys.path.insert(0, str(ROOT))
    from experiments.vlabench.eval_select_book import main as evaluate
    current = None

    def notify(kind, job):
        nonlocal current
        current = job
        print(f'{kind.upper()} worker={slot} gpu={gpu} task={job[0]} episode={job[1]}', flush=True)
        events.put(dict(kind=kind, slot=slot, gpu=gpu, task=job[0], episode=job[1]))

    def run(job, runtime):
        task, episode = job
        argv = ['--task', task, '--checkpoint', options['checkpoint'],
                '--output-dir', str(out / task / f'episode_{episode:03d}'),
                '--episodes', str(options['episodes']), '--episode-ids', str(episode),
                '--replan-steps', str(options['replan_steps']),
                '--gripper-threshold', str(options['gripper_threshold']),
                '--seed', str(options['seed']), '--decode-mode', options['decode_mode']]
        if options.get('vae_safetensors_path'):
            argv += ['--vae-safetensors-path', options['vae_safetensors_path']]
        if options.get('allow_vae_mismatch'):
            argv.append('--allow-vae-mismatch')
        model_before = None if runtime is None else id(runtime.model)
        result = evaluate(argv, runtime=runtime)
        if model_before is not None and id(result.model) != model_before:
            raise RuntimeError('Persistent worker unexpectedly replaced its model')
        print(f'MODEL_REUSE worker={slot} model_id={id(result.model)} first_load={model_before is None}', flush=True)
        return result

    try:
        recoverable = ()
        if options.get('continue_on_physics_error'):
            from dm_control.rl.control import PhysicsError
            recoverable = (PhysicsError,)

        def on_error(job, error):
            trace = traceback.format_exc()
            # The official evaluator closes env only on its normal return path.
            # Close its local environment before discarding this failed episode.
            tb = error.__traceback__
            while tb is not None:
                frame = tb.tb_frame
                if frame.f_code.co_name == 'evaluate_single_episode' and 'env' in frame.f_locals:
                    frame.f_locals['env'].close()
                tb = tb.tb_next
            print(f'PHYSICS_ERROR worker={slot} task={job[0]} episode={job[1]}\n{trace}', flush=True)
            events.put(dict(kind='episode_error', slot=slot, gpu=gpu, job=job,
                            traceback=trace, output=str(out / job[0] / f'episode_{job[1]:03d}')))

        consume_jobs(jobs, run, notify, recoverable, on_error)
    except BaseException:
        events.put(dict(kind='error', slot=slot, gpu=gpu, job=current,
                        traceback=traceback.format_exc()))
        raise


def summary(tasks, episodes, records):
    rows = {}
    for task in tasks:
        valid = [v for (t, _), v in records.items() if t == task]
        successes = sum(bool(v['result']['success']) for v in valid)
        rows[task] = dict(completed=len(valid), expected=episodes, successes=successes,
                         success_rate=successes / episodes if len(valid) == episodes else None)
    complete = len(records) == len(tasks) * episodes
    return dict(complete=complete, completed=len(records), expected=len(tasks)*episodes,
                successes=sum(r['successes'] for r in rows.values()), by_task=rows,
                records=list(records.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vae-safetensors-path', type=Path)
    parser.add_argument('--allow-vae-mismatch', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--completed-from', type=Path)
    parser.add_argument('--resume-status', type=Path)
    parser.add_argument('--continue-on-physics-error', action='store_true',
                        help='Record PhysicsError separately and continue; other errors remain fatal')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--replan-steps', type=int, default=8)
    parser.add_argument('--gripper-threshold', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--decode-mode', choices=['legacy', 'robust_joint'], default='legacy')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve(); args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    if args.vae_safetensors_path is not None:
        args.vae_safetensors_path = args.vae_safetensors_path.resolve()
        if not args.vae_safetensors_path.is_file():
            raise FileNotFoundError(args.vae_safetensors_path)
    if args.threads < 1 or not 1 <= args.replan_steps <= 16 or not 0 <= args.gripper_threshold <= 1:
        raise ValueError('Invalid threads or action parameters')
    track = json.loads(TRACK.read_text())
    tasks = args.tasks or list(track)
    all_jobs = jobs_for(tasks, args.episodes, track)
    slots = slot_gpus(args.gpu_ids, args.workers_per_gpu)
    records = inherited_results(args.completed_from, all_jobs, args)
    errors = []
    if args.resume_status:
        if args.completed_from:
            raise ValueError('Use either --resume-status or --completed-from')
        records, errors = resume_status(args.resume_status, all_jobs, args)
    failed = {tuple(e['job']) for e in errors}
    pending = [j for j in all_jobs if j not in records and j not in failed]
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan = dict(options=options, slots=slots, tasks=tasks, pending=pending,
                inherited=len(records), inherited_errors=errors)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'plan.json').write_text(json.dumps(plan, indent=2))
    context = mp.get_context('spawn')
    jobs, events = context.Queue(), context.Queue()
    for job in pending: jobs.put(job)
    for _ in slots: jobs.put(None)
    processes = []
    active = {}

    def save(state):
        report = summary(tasks, args.episodes, records)
        report.update(physics_error_count=len(failed), processed=len(records)+len(failed),
                      pending=len(all_jobs)-len(records)-len(failed))
        report.update(state=state, active=list(active.values()), errors=errors,
                      workers=[dict(slot=i, gpu=slots[i], pid=p.pid, exitcode=p.exitcode)
                               for i, p in enumerate(processes)], updated=time.time())
        tmp = args.output_dir/'status.json.new'
        tmp.write_text(json.dumps(report, indent=2)); tmp.replace(args.output_dir/'status.json')
        return report

    def interrupt(signum, frame): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if pending:
            for slot, gpu in enumerate(slots):
                p = context.Process(target=worker, args=(slot, gpu, jobs, events, options))
                p.start(); processes.append(p)
        save('running')
        while len(records) + len(failed) < len(all_jobs):
            try: event = events.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes):
                    raise RuntimeError('Worker exited unexpectedly; see worker log')
                if processes and all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Workers exited before all results arrived')
                continue
            kind = event['kind']; slot = event['slot']
            if kind == 'error':
                errors.append(event); raise RuntimeError('Worker error; see status and worker log')
            if kind == 'episode_error':
                key = tuple(event['job'])
                if key in records or key in failed:
                    raise ValueError(f'Duplicate outcome {key}')
                errors.append(event); failed.add(key); active.pop(slot, None)
                save('running'); continue
            key = (event['task'], event['episode'])
            if kind == 'started': active[slot] = event
            elif kind == 'completed':
                if key in records: raise ValueError(f'Duplicate result {key}')
                dest = args.output_dir/key[0]/f'episode_{key[1]:03d}'
                result = json.loads((dest/f'episode_{key[1]:03d}.json').read_text())
                if result['episode'] != key[1] or result.get('task') != key[0]:
                    raise ValueError(f'Result identity mismatch {key}')
                records[key] = dict(task=key[0], episode=key[1], result=result,
                                    output=str(dest), inherited=False)
                active.pop(slot, None)
            save('running')
        for p in processes: p.join()
        if any(p.exitcode != 0 for p in processes): raise RuntimeError('Worker shutdown failed')
        (args.output_dir/'summary.json').write_text(json.dumps(
            save('completed_with_errors' if failed else 'completed'), indent=2))
    except BaseException:
        for p in processes:
            if p.is_alive(): p.terminate()
        deadline = time.monotonic()+15
        for p in processes:
            p.join(max(.01, deadline-time.monotonic()))
            if p.is_alive(): p.kill(); p.join()
        save('interrupted_or_failed'); raise
    finally:
        jobs.cancel_join_thread(); jobs.close()
        events.close(); events.join_thread()


if __name__ == '__main__':
    main()
