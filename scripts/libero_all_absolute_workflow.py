#!/usr/bin/env python3
"""Local preparation/launch entry for the four-suite absolute Wan2.1 experiment.

GPU stages print their exact command by default; --execute explicitly runs it.
No changes to generic cache, training, or evaluation implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASK = 'libero_all4_rothko_all_absolute_2cam224_full_wan21_1_3b_1e-4'
CACHE = ROOT / 'data/libero_all4_rothko_all_absolute_2cam224_wan21_bf16_h16_latents'
TRAIN_ENV = Path('/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin')
EVAL_ENV = Path('/mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin')
STATS_SHA = '999f9e9031ac55bb40a0b89c60dbd1389852e33e6f173d9a58ba48e0f1c216f4'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def check_inputs(require_cache=False):
    """CPU-only config/identity/coverage checks, never constructs a GPU model."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
    from fastwam.representations.libero_rothko_all_absolute import LiberoAllAbsoluteRothkoCodec
    from fastwam.representations.rothko import RothkoNormStats
    from fastwam.datasets.latent_cache import build_dataset_contract, LatentCacheReader
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['task='+TASK])
    OmegaConf.resolve(cfg)
    expected = dict(batch_size=4, gradient_accumulation_steps=4, num_epochs=10,
                    learning_rate=1e-4, warmup_ratio=.05, eval_every=1000,
                    save_every=3000, state_save_every=3000, save_at_end=True,
                    save_before_eval=False, mixed_precision='bf16', seed=42)
    for k, v in expected.items():
        if cfg.get(k) != v:
            raise ValueError(f'Unexpected {k}: {cfg.get(k)} != {v}')
    if cfg.get('save_every_epochs') or cfg.get('state_save_every_epochs'):
        raise ValueError('Epoch-based saving would override the agreed step interval')
    if cfg.get('resume') or cfg.get('init_weights') or cfg.get('max_steps') is not None:
        raise ValueError('This entry starts the agreed fresh 10-epoch run only')
    if cfg.model.vae_safetensors_path is not None or cfg.model.video_dit_config.use_gradient_checkpointing:
        raise ValueError('Expected original VAE and disabled gradient checkpointing')
    stats = Path(cfg.data.train.rothko_norm_stats)
    loaded_stats = RothkoNormStats.load(stats)
    if loaded_stats.fingerprint() != STATS_SHA:
        raise ValueError('Unexpected full-absolute norm_stats fingerprint')
    codec = LiberoAllAbsoluteRothkoCodec(
        LiberoRothkoCodecConfig(**OmegaConf.to_container(cfg.data.train.rothko_config)), loaded_stats)
    roots = list(cfg.data.train.dataset_dirs)
    counts = [json.loads((Path(d)/'meta/info.json').read_text())['total_frames'] for d in roots]
    count = sum(counts)
    if count != 277713 or len(roots) != 4:
        raise ValueError(f'Dataset coverage changed: {counts}')
    contract = build_dataset_contract(dataset_dirs=roots, dataset_length=count,
        num_frames=cfg.data.train.num_frames, video_size=list(cfg.data.train.video_size),
        raymap_representation=cfg.data.train.raymap_representation,
        raymap_codec_metadata=codec.metadata(), norm_stats_sha256=STATS_SHA)
    vae = ROOT/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
    dit = vae.parent/'diffusion_pytorch_model.safetensors'
    if not dit.is_file() or not Path(cfg.data.train.text_embedding_cache_dir).is_dir():
        raise FileNotFoundError('Original DiT weights or existing LIBERO text cache are missing')
    identity = dict(kind='original_wan21', filename=vae.name,
                    size_bytes=vae.stat().st_size, sha256=digest(vae))
    result = dict(config=expected, dataset_frames=count, suite_frames=counts,
                  norm_stats_sha256=STATS_SHA, vae_identity=identity,
                  norm_stats_file_sha256=digest(stats),
                  expected_steps=21700, expected_warmup_steps=1085,
                  cache_directory=str(CACHE), cache_checked=False)
    if require_cache:
        reader = LatentCacheReader(CACHE, expected_dataset_contract=contract)
        m = reader.metadata
        if (m.get('benchmark_only') or len(reader) != count
                or m.get('model_variant') != 'wan2.1-t2v-1.3b'
                or m.get('encoding_torch_dtype') != 'torch.bfloat16'
                or m.get('modalities') != ['rgb', 'raymap']
                or m.get('latent_shape') != [16, 5, 28, 56]
                or m.get('vae_identity') != identity
                or not m.get('verification', {}).get('training_loss_bit_exact')):
            raise ValueError('Cache identity, precision, shape or verification mismatch')
        result['cache_checked'] = True
        result['cache_metadata_sha256'] = digest(CACHE/'metadata.json')
    return result


def command(args):
    envbin = EVAL_ENV if args.stage == 'eval' else TRAIN_ENV
    if args.stage in ('cache', 'benchmark'):
        out = CACHE if args.stage == 'cache' else args.output
        cmd = [str(envbin/'torchrun'), '--standalone', '--nproc_per_node=8',
               'scripts/precompute_visual_action_latents.py', '--task', TASK,
               '--output-dir', str(out), '--batch-size', str(args.batch_size),
               '--num-workers', str(args.num_workers), '--samples-per-shard',
               '1024' if args.stage == 'cache' else '32', '--log-every', '20']
        if args.stage == 'benchmark':
            cmd += ['--benchmark-max-samples', '256']
        return cmd
    if args.stage == 'train':
        return ['bash', 'scripts/train_zero1.sh', '8', 'task='+TASK,
                'output_dir='+str(args.output)]
    if args.stage == 'audit':
        cmd = [str(envbin/'python'), 'scripts/audit_libero_all_absolute.py',
               '--output', str(args.output)]
        if args.indices_json:
            cmd += ['--indices-json', str(args.indices_json.resolve())]
        return cmd
    if args.stage == 'eval':
        run = args.run.resolve()
        return [str(envbin/'python'), 'experiments/libero/run_libero_manager.py',
                '--config-name=sim_libero_all_absolute', 'task='+TASK,
                'ckpt='+str(run/'checkpoints/weights/step_021700.pt'),
                'model.vae_safetensors_path=null', 'model.allow_vae_mismatch=false',
                'model.rothko_decode_mode=legacy', 'model.rothko_decode_anchor_alpha=0',
                'seed=42', 'EVALUATION.replan_steps=8', 'EVALUATION.num_trials=50',
                'EVALUATION.num_inference_steps=20', 'EVALUATION.use_action_ensembler=false',
                'EVALUATION.save_prediction_videos=false',
                'EVALUATION.dataset_stats_path='+str(run/'dataset_stats.json'),
                'EVALUATION.output_dir='+str(args.output),
                'MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10]',
                'MULTIRUN.num_gpus=8', 'MULTIRUN.max_tasks_per_gpu=2']
    raise ValueError(args.stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'check-cache', 'audit', 'benchmark', 'cache', 'train', 'eval'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, help='New audit/benchmark/run/evaluation directory')
    parser.add_argument('--run', type=Path, help='Completed training directory for eval')
    parser.add_argument('--indices-json', type=Path, help='Optional four-suite audit window manifest')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--batch-size', type=int, default=8, help='VAE cache batch, not training batch')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers per cache process')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT/'src'))
    if args.stage in ('check', 'check-cache'):
        print(json.dumps(check_inputs(args.stage == 'check-cache'), indent=2))
        return
    if args.stage == 'eval' and args.run is None:
        parser.error('eval requires --run; only its final step_021700 is selected')
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Invalid cache batch/worker settings')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    if args.output is None:
        base = ROOT/'runs'/TASK if args.stage == 'train' else ROOT/'evaluate_results/libero/all4_absolute'
        args.output = base/(args.stage+'_'+stamp)
    args.output = args.output.resolve()
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        print('Dry run only. Add --execute to run this stage; no GPU task started.')
        return
    gpu_ids = args.gpus.split(',')
    expected_gpus = len(gpu_ids) if args.stage == 'audit' else 8
    if len(gpu_ids) != expected_gpus or len(set(gpu_ids)) != expected_gpus or not all(x.isdigit() for x in gpu_ids):
        parser.error('This experiment requires eight distinct numeric GPU IDs')
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    used = {k.strip(): int(v) for k,v in (line.split(',') for line in usage.splitlines())}
    if any(used.get(g, 10**9) > 128 for g in gpu_ids):
        raise RuntimeError('Selected GPUs are occupied; no jobs were started or stopped')
    result = check_inputs(require_cache=args.stage == 'train')
    if args.stage == 'cache':
        if CACHE.exists():
            raise FileExistsError(f'Refusing to overwrite/resume implicitly: {CACHE}')
    elif args.output.exists():
        raise FileExistsError(args.output)
    if args.stage == 'eval':
        for f in ['checkpoints/weights/step_021700.pt', 'dataset_stats.json', 'config.yaml']:
            if not (args.run/f).is_file():
                raise FileNotFoundError(args.run/f)
        from omegaconf import OmegaConf
        trained = OmegaConf.load(args.run/'config.yaml')
        if trained.model.raymap_representation != 'libero_rothko_all_absolute':
            raise ValueError('Evaluation run is not the full-absolute representation')
    record = ROOT/'evaluate_results/libero/all4_absolute'/('launch_'+args.stage+'_'+stamp)
    record.mkdir(parents=True, exist_ok=False)
    (record/'inputs.json').write_text(json.dumps(result, indent=2)+'\n')
    (record/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    env = os.environ.copy()
    envbin = EVAL_ENV if args.stage == 'eval' else TRAIN_ENV
    # Short path is necessary for DataLoader AF_UNIX sockets.
    tmp = ROOT/'evaluate_results/libero'/('tmp_la_'+str(os.getpid()))
    tmp.mkdir(exist_ok=False)
    env.update(PATH=str(envbin)+os.pathsep+env['PATH'], PYTHONPATH=str(ROOT/'src'),
        CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', TMPDIR=str(tmp),
        DIFFSYNTH_MODEL_BASE_PATH=str(ROOT/'checkpoints'), PYTHONDONTWRITEBYTECODE='1')
    with socket.socket() as sock:
        sock.bind(('', 0)); env['MASTER_PORT'] = str(sock.getsockname()[1])
    source_paths = [Path(__file__), ROOT/'configs/task'/f'{TASK}.yaml',
        ROOT/'src/fastwam/trainer.py', ROOT/'src/fastwam/models/wan22/fastwam_visual_action.py',
        ROOT/'src/fastwam/representations/libero_rothko_all_absolute.py',
        ROOT/'scripts/precompute_visual_action_latents.py']
    (record/'source_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):digest(f) for f in source_paths},indent=2))
    (record/'status.json').write_text(json.dumps(dict(state='running',command=cmd)))
    print(f'Launch records and output.log: {record}', flush=True)
    with (record/'output.log').open('x') as log:
        with subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                log.write(line); log.flush()
                print(line, end='', flush=True)
            code = child.wait()
    (record/'status.json').write_text(json.dumps(dict(state='complete' if code == 0 else 'failed',returncode=code)))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
