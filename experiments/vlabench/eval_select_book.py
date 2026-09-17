"""Independent Track-1 select_book entry point using official control loop.

Runtime errors are recorded and re-raised, never counted as successful trials.
The first version refuses nonempty output directories (no silent overwrites).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from contextlib import nullcontext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vae-safetensors-path', type=Path)
    parser.add_argument('--allow-vae-mismatch', action='store_true')
    parser.add_argument('--decode-mode', choices=['legacy', 'robust_joint'], default='legacy')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--replan-steps', type=int, required=True)
    parser.add_argument('--gripper-threshold', type=float, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--diagnostics', action='store_true', help='Record EE/IK and frames; no control changes.')
    parser.add_argument('--episode-ids', type=int, nargs='+', default=None,
                        help='Global Track-1 episode IDs; used by the multi-GPU manager.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, str(root / 'third_party/VLABench'))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.chdir(root)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.vae_safetensors_path is not None:
        args.vae_safetensors_path = args.vae_safetensors_path.resolve()
        if not args.vae_safetensors_path.is_file():
            raise FileNotFoundError(args.vae_safetensors_path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Refusing nonempty output directory: {args.output_dir}')
    track_path = root / 'third_party/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json'
    track = json.loads(track_path.read_text())
    if not 1 <= args.episodes <= len(track['select_book']):
        raise ValueError('Invalid Track 1 episode count')
    episode_ids = list(range(args.episodes)) if args.episode_ids is None else args.episode_ids
    if len(set(episode_ids)) != len(episode_ids) or any(i < 0 or i >= args.episodes for i in episode_ids):
        raise ValueError('episode-ids must be unique and within [0, episodes)')
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    from experiments.vlabench.policy import FastWAMVLABenchPolicy
    import VLABench.robots  # Official registry initialization.
    import VLABench.tasks
    from VLABench.evaluation.evaluator.base import Evaluator
    from VLABench.configs import name2config
    from VLABench.utils.utils import find_key_by_value
    register_default_resolvers()
    config = OmegaConf.load(args.checkpoint.resolve().parents[2] / 'config.yaml')
    config.model.load_text_encoder = True
    config.model.rothko_decode_mode = args.decode_mode
    config.model.rothko_decode_anchor_alpha = 0.
    config.model.vae_safetensors_path = str(args.vae_safetensors_path) if args.vae_safetensors_path else None
    if args.allow_vae_mismatch:
        config.model.allow_vae_mismatch = True
    model = instantiate(config.model, device='cuda', model_dtype=torch.bfloat16)
    model.load_checkpoint(str(args.checkpoint))
    model.validate_dataset_stats(config.data.train.pretrained_norm_stats)
    model.eval().requires_grad_(False)
    agent = FastWAMVLABenchPolicy(model, replan_steps=args.replan_steps,
                                 gripper_threshold=args.gripper_threshold, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(tasks=['select_book'], n_episodes=args.episodes,
        episode_config=track, max_substeps=1, save_dir=str(args.output_dir),
        visulization=True, metrics=['success_rate', 'intention_score', 'progress_score'])
    task_config = evaluator.task_configs.get(find_key_by_value(name2config, 'select_book'), {})
    limit = task_config.get('evaluation', {}).get('max_episode_length', 200)
    identity = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    identity.update(track_sha256=hashlib.sha256(track_path.read_bytes()).hexdigest(),
                    max_episode_length=limit, max_substeps=1, decoder=args.decode_mode,
                    vae=str(args.vae_safetensors_path) if args.vae_safetensors_path else 'original')
    identity.update(raymap_codec=model.raymap_codec.metadata(),
                    rothko_stats_fingerprint=model.raymap_codec.norm_stats.fingerprint(),
                    camera_ids=([2,0,3] if getattr(model.raymap_codec.config, 'horizontal_copies', 2)==3 else [2,3]))
    (args.output_dir / 'evaluation_config.json').write_text(json.dumps(identity, indent=2))
    results = []
    for episode in episode_ids:
        np.random.seed(args.seed + episode); random.seed(args.seed + episode)
        torch.manual_seed(args.seed + episode)
        agent.reset()
        try:
            context = nullcontext()
            if args.diagnostics:
                from experiments.vlabench.diagnostics import record_episode
                context = record_episode(agent, args.output_dir / f'diagnostics_episode_{episode:03d}')
            with context:
                result = evaluator.evaluate_single_episode(agent, 'select_book', episode,
                    track['select_book'][episode], seed=args.seed+episode, max_episode_length=limit)
        except Exception as error:
            (args.output_dir / f'episode_{episode:03d}.error.json').write_text(
                json.dumps(dict(episode=episode,error=repr(error)),indent=2))
            raise
        result.update(episode=episode, language=agent.language_history)
        (args.output_dir / f'episode_{episode:03d}.json').write_text(json.dumps(result, indent=2))
        results.append(result)
        (args.output_dir / 'metrics.json').write_text(json.dumps(evaluator.compute_metric(results),indent=2))


if __name__ == '__main__':
    main()
