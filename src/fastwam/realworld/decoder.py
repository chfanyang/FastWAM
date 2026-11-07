"""Load a realworld decoder export after the original DiT/VAE identity check."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .common import sha256_file


def load_decoder_override(model, path):
    path = Path(path).resolve()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        info = handle.metadata() or {}
        meta = json.loads(info.get("training_metadata_json", "{}"))
        if (info.get("finetuned_modules") != "conv2,decoder"
                or meta.get("environment") != "realworld"
                or meta.get("norm_stats_fingerprint") != model.raymap_codec.norm_stats.fingerprint()
                or meta.get("codec_metadata") != model.raymap_codec.metadata()):
            raise ValueError("Decoder export does not match realworld training statistics/codec")
        current = model.vae.model.state_dict()
        if set(handle.keys()) != set(current):
            raise ValueError("Decoder export VAE architecture mismatch")
        changed = {}
        for name, tensor in current.items():
            value = handle.get_tensor(name)
            if value.shape != tensor.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid VAE tensor: {name}")
            if name.startswith(("decoder.", "conv2.")):
                changed[name] = value
            elif not torch.equal(value, tensor.detach().cpu().to(value.dtype)):
                raise ValueError(f"Frozen VAE weights differ: {name}")
        # Validate the complete export before mutating either trainable module.
        for module_name in ("decoder", "conv2"):
            prefix = module_name + "."
            getattr(model.vae.model, module_name).load_state_dict(
                {k[len(prefix):]: v for k, v in changed.items() if k.startswith(prefix)}, strict=True)
    return dict(path=str(path), sha256=sha256_file(path), step=info.get("checkpoint_step"),
                frozen_encoder_verified=True, norm_stats_fingerprint=meta["norm_stats_fingerprint"])
