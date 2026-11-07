#!/usr/bin/env python3
"""Serve a trained realworld checkpoint over OpenPI-style binary WebSockets."""
import argparse
from pathlib import Path
from functools import partial


def handle_connection(policy, connection):
    """Transport handler shared with CPU loopback tests; no model loading here."""
    from fastwam.realworld.protocol import pack, unpack
    connection.send(pack(policy.metadata))
    for message in connection:
        try:
            response = pack(policy.infer(unpack(message)))
        except Exception as exc:
            # OpenPI clients treat text frames as inference errors.
            response = f"{type(exc).__name__}: {exc}"
        connection.send(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training run config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder-vae", help="Optional realworld conv2+decoder VAE export; encoder must match")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save-rgb-dir", help="Save each request's 16 future RGB PNGs here")
    args = parser.parse_args()
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from websockets.sync.server import serve
    from fastwam.realworld.policy import RealWorldPolicy
    from fastwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    if cfg.model._target_ != "fastwam.realworld.runtime.create_model":
        raise ValueError("This server only loads isolated realworld configs")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["skip_dit_load_from_pretrain"] = True
    model = instantiate(model_cfg, device=args.device, model_dtype=torch.bfloat16)
    model.load_checkpoint(args.checkpoint)
    data = cfg.data.train
    if model.checkpoint_dataset_stats_sha256 is None:
        raise ValueError("Realworld checkpoint is missing its training data identity")
    model.validate_dataset_stats(data.pretrained_norm_stats)
    decoder_info = None
    if args.decoder_vae:
        from fastwam.realworld.decoder import load_decoder_override
        decoder_info = load_decoder_override(model, args.decoder_vae)
        print(f"Loaded decoder override: {decoder_info}", flush=True)
    model.eval().requires_grad_(False)
    policy = RealWorldPolicy(model, data.split_manifest, data.rothko_norm_stats,
        data.text_embedding_cache_dir, num_inference_steps=args.steps,
        save_rgb_dir=args.save_rgb_dir)
    policy.metadata["decoder_override"] = decoder_info

    print(f"Serving {args.checkpoint} at ws://{args.host}:{args.port}", flush=True)
    with serve(partial(handle_connection, policy), args.host, args.port, compression=None, max_size=8 * 1024 * 1024) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
