#!/usr/bin/env python3
"""Validate prepared inputs on CPU before a GPU job starts."""
import argparse
import json
from pathlib import Path

from fastwam.realworld.common import sha256_file, image_inventory_fingerprint


def check_sources(root, manifest):
    for record in manifest["episodes"]:
        ep = Path(root) / record["path"]
        if sha256_file(ep / "states.npz") != record["states_sha256"]:
            raise ValueError(f"Source states changed: {ep}")
        if image_inventory_fingerprint(ep, record["length"]) != record["image_inventory_sha256"]:
            raise ValueError(f"Source images changed: {ep}. Prepare a new artifact/cache version.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/realworld_data")
    parser.add_argument("--artifacts", default="data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_v2")
    parser.add_argument("--require-text", action="store_true")
    parser.add_argument("--require-latents", action="store_true")
    args = parser.parse_args()
    root = Path(args.artifacts)
    manifest = json.loads((root / "split.json").read_text())
    check_sources(args.data_root, manifest)
    import torch
    from fastwam.realworld.dataset import RealWorldVideoDataset, load_context
    torch.set_num_threads(1)
    ds = RealWorldVideoDataset(dataset_dir=args.data_root, split_manifest=root / "split.json",
        rothko_norm_stats=root / "rothko_stats.pt", pretrained_norm_stats=root / "rothko_stats.json",
        include_text_context=False,
        latent_cache_dir=root / "latents_wan21_bf16" if args.require_latents else None,
        latent_cache_only=args.require_latents)
    if args.require_text:
        for instruction in sorted({r["instruction"] for r in manifest["episodes"]}):
            load_context(root / "text", instruction)
    print(f"CPU preflight passed: {len(manifest['episodes'])} episodes, {len(ds)} train windows", flush=True)


if __name__ == "__main__":
    main()
