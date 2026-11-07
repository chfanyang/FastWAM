#!/usr/bin/env python3
"""CPU data/config checks before explicitly launching the original 5B baseline."""
import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from fastwam.realworld.common import sha256_file
from fastwam.trainer import load_fixed_validation_manifest
from fastwam.utils.config_resolvers import register_default_resolvers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="realworld_all4_original_fastwam_wan22_5b_ep8")
    args = parser.parse_args()
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}",
            "data.train.include_text_context=false"])
    train, val = instantiate(cfg.data.train), instantiate(cfg.data.val)
    if train.codec.fingerprint != val.codec.fingerprint:
        raise ValueError("Mismatched action statistics")
    source_stats = Path(cfg.data.train.split_manifest).with_name("rothko_stats.json")
    if train.codec.stats["source_statistics_sha256"] != sha256_file(source_stats):
        raise ValueError("Baseline normalization no longer matches shared train-only bounds")
    manifest = load_fixed_validation_manifest(cfg.eval_sample_manifest,
        expected_split_manifest_sha256=train.episode_split_metadata["sha256"], val_dataset_length=len(val))
    if len(manifest["samples"]) != int(cfg.eval_num_samples):
        raise ValueError("Fixed validation sample count differs from task config")
    for sample in manifest["samples"]:
        value = val[sample["val_dataset_index"]]
        if value["action"].shape != (16, 7) or value["proprio"].shape != (17, 8):
            raise ValueError("Invalid baseline data shapes")
    if not Path(cfg.model.action_dit_pretrained_path).is_file():
        raise FileNotFoundError(cfg.model.action_dit_pretrained_path)
    if len(list(Path("checkpoints/Wan-AI/Wan2.2-TI2V-5B").glob("diffusion_pytorch_model*.safetensors"))) != 3:
        raise FileNotFoundError("Expected three original Wan2.2 5B DiT weight shards")
    print(json.dumps(dict(model="original FastWAM Wan2.2 5B + ActionDiT", train_windows=len(train),
        val_windows=len(val), task=train.task_name, epochs=cfg.num_epochs, batch_per_gpu=cfg.batch_size,
        gradient_accumulation=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.model.mot_checkpoint_mixed_attn)))


if __name__ == "__main__":
    main()
