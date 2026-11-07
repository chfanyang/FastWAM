#!/usr/bin/env python3
"""Serve an original FastWAM realworld checkpoint to client_fastwam_ee.py."""
import argparse
from functools import partial
from pathlib import Path

from serve import handle_connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--save-rgb-dir')
    args = parser.parse_args()
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from websockets.sync.server import serve
    from fastwam.realworld.baseline_policy import OriginalFastWAMPolicy
    from fastwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    if cfg.model._target_ != 'fastwam.realworld.baseline_model.create_model':
        raise ValueError('Expected original FastWAM realworld training config')
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg['skip_dit_load_from_pretrain'] = True
    model_cfg['mot_checkpoint_mixed_attn'] = False
    for name in ('video_dit_config', 'action_dit_config'):
        model_cfg[name]['use_gradient_checkpointing'] = False
    model = instantiate(model_cfg, device=args.device, model_dtype=torch.bfloat16)
    payload = model.load_checkpoint(args.checkpoint)
    step = payload.get('step')
    del payload
    model.eval().requires_grad_(False)
    data = cfg.data.train
    policy = OriginalFastWAMPolicy(model, data.split_manifest, data.pretrained_norm_stats,
        data.text_embedding_cache_dir, task_name=data.get('task_name'),
        num_inference_steps=args.steps, save_rgb_dir=args.save_rgb_dir)
    policy.metadata['checkpoint_step'] = step
    print(f'Serving original FastWAM step={step} at ws://{args.host}:{args.port}', flush=True)
    with serve(partial(handle_connection, policy), args.host, args.port,
               compression=None, max_size=8 * 1024 * 1024) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
