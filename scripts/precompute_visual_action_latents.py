#!/usr/bin/env python3
"""Precompute frozen Wan VAE latents for RGB/Rothko training windows."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, default_collate

from fastwam.datasets.latent_cache import (
    LATENT_CACHE_DTYPE,
    LATENT_CACHE_METADATA,
    LATENT_CACHE_SUCCESS,
    LATENT_CACHE_VERSION,
)
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging


logger = get_logger(__name__)


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, offset: int):
        index = self.indices[offset]
        return index, self.dataset[index]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_object(value: Any, rank: int) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _shard_specs(
    num_samples: int, samples_per_shard: int
) -> list[dict[str, Any]]:
    count = math.ceil(num_samples / samples_per_shard)
    specs = []
    for shard_index in range(count):
        start = shard_index * samples_per_shard
        end = min(num_samples, start + samples_per_shard)
        specs.append(
            {
                "index": shard_index,
                "start": start,
                "end": end,
                "file": f"latents-{shard_index:05d}-of-{count:05d}.bin",
            }
        )
    return specs


def _expected_shard_bytes(
    shard: dict[str, Any], latent_shape: tuple[int, ...]
) -> int:
    count = int(shard["end"]) - int(shard["start"])
    return count * 2 * int(np.prod(latent_shape)) * np.dtype(np.uint16).itemsize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pin DataLoader CPU tensors (disabled by default for this workload).",
    )
    parser.add_argument("--samples-per-shard", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latent precomputation.")
    if args.batch_size <= 0 or args.samples_per_shard <= 0:
        raise ValueError("batch size and samples per shard must be positive.")

    rank, world_size, local_rank = _init_distributed()
    setup_logging(is_main_process=rank == 0)
    register_default_resolvers()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(output_dir)

    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    overrides = [
        f"task={args.task}",
        "data.train.latent_cache_dir=null",
        "data.train.latent_cache_only=false",
        "data.train.sample_error_mode=raise",
        *args.override,
    ]
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)

    dataset = instantiate(cfg.data.train)
    device = f"cuda:{local_rank}"
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    model.eval().requires_grad_(False)
    vae = model.vae
    model_variant = model.model_variant
    vae_identity = model._current_vae_identity() if rank == 0 else None
    vae_identity = _broadcast_object(vae_identity, rank)

    latent_shape = (
        int(model.vae_latent_channels),
        int(model.num_latent_frames_per_modality),
        int(cfg.data.train.video_size[0]) // int(vae.upsampling_factor),
        int(cfg.data.train.video_size[1]) // int(vae.upsampling_factor),
    )
    shards = _shard_specs(len(dataset), args.samples_per_shard)
    metadata = {
        "cache_version": LATENT_CACHE_VERSION,
        "complete": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_task": args.task,
        "source_overrides": overrides,
        "num_samples": len(dataset),
        "latent_shape": list(latent_shape),
        "modalities": ["rgb", "raymap"],
        "dtype": LATENT_CACHE_DTYPE,
        "encoding_torch_dtype": str(torch.bfloat16),
        "model_variant": model_variant,
        "vae_identity": vae_identity,
        "dataset_contract": dataset.latent_cache_dataset_contract,
        "samples_per_shard": args.samples_per_shard,
        "shards": shards,
    }
    metadata_path = output_dir / LATENT_CACHE_METADATA
    success_path = output_dir / LATENT_CACHE_SUCCESS
    if rank == 0:
        if success_path.exists():
            raise FileExistsError(
                f"Latent cache is already complete: {output_dir}"
            )
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            stable_keys = (
                "cache_version",
                "num_samples",
                "latent_shape",
                "modalities",
                "dtype",
                "encoding_torch_dtype",
                "model_variant",
                "vae_identity",
                "dataset_contract",
                "samples_per_shard",
                "shards",
            )
            mismatched = [
                key for key in stable_keys if existing.get(key) != metadata.get(key)
            ]
            if mismatched:
                raise ValueError(
                    "Cannot resume incompatible latent cache; mismatched metadata "
                    f"keys={mismatched}, path={metadata_path}"
                )
            metadata["created_at_utc"] = existing.get(
                "created_at_utc", metadata["created_at_utc"]
            )
        _write_json_atomic(metadata_path, metadata)
    _barrier()

    shard_begin = len(shards) * rank // world_size
    shard_end = len(shards) * (rank + 1) // world_size
    local_shards = shards[shard_begin:shard_end]
    pending_shards = []
    indices = []
    for shard in local_shards:
        final_path = output_dir / shard["file"]
        expected_bytes = _expected_shard_bytes(shard, latent_shape)
        if final_path.is_file() and final_path.stat().st_size == expected_bytes:
            logger.info("Rank %d skipping complete shard %s", rank, final_path.name)
            continue
        pending_shards.append(shard)
        indices.extend(range(int(shard["start"]), int(shard["end"])))

    indexed_dataset = _IndexedDataset(dataset, indices)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    shard_by_index = {
        index: shard
        for shard in pending_shards
        for index in range(int(shard["start"]), int(shard["end"]))
    }
    open_arrays: dict[int, np.memmap] = {}
    written_counts = {int(shard["index"]): 0 for shard in pending_shards}
    started = time.perf_counter()
    processed = 0

    def open_array(shard: dict[str, Any]) -> np.memmap:
        shard_index = int(shard["index"])
        array = open_arrays.get(shard_index)
        if array is not None:
            return array
        temporary_path = output_dir / (shard["file"] + f".partial-rank{rank:03d}")
        if temporary_path.exists():
            temporary_path.unlink()
        count = int(shard["end"]) - int(shard["start"])
        array = np.memmap(
            temporary_path,
            mode="w+",
            dtype=np.uint16,
            shape=(count, 2, *latent_shape),
        )
        open_arrays[shard_index] = array
        return array

    with torch.inference_mode():
        for batch_index, (global_indices, sample) in enumerate(loader):
            rgb = sample["video"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            raymap = sample["raymap"].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            rgb_latents = vae.encode(rgb, device=device, tiled=False)
            raymap_latents = vae.encode(raymap, device=device, tiled=False)
            pair = torch.stack((rgb_latents, raymap_latents), dim=1)
            if tuple(pair.shape[2:]) != latent_shape or pair.dtype != torch.bfloat16:
                raise ValueError(
                    f"Unexpected cached latent tensor: shape={tuple(pair.shape)}, "
                    f"dtype={pair.dtype}, expected=[B,2,{latent_shape}] BF16."
                )
            raw = pair.contiguous().cpu().view(torch.uint16).numpy()
            for row, global_index_tensor in enumerate(global_indices):
                global_index = int(global_index_tensor)
                shard = shard_by_index[global_index]
                shard_index = int(shard["index"])
                local_index = global_index - int(shard["start"])
                open_array(shard)[local_index] = raw[row]
                written_counts[shard_index] += 1
                if written_counts[shard_index] == int(shard["end"]) - int(shard["start"]):
                    array = open_arrays.pop(shard_index)
                    array.flush()
                    del array
                    temporary_path = output_dir / (
                        shard["file"] + f".partial-rank{rank:03d}"
                    )
                    temporary_path.replace(output_dir / shard["file"])
                    logger.info("Rank %d completed shard %s", rank, shard["file"])
            processed += len(global_indices)
            if batch_index % args.log_every == 0:
                elapsed = time.perf_counter() - started
                rate = processed / max(elapsed, 1e-6)
                logger.info(
                    "Rank %d cached %d/%d local samples (%.2f samples/s)",
                    rank,
                    processed,
                    len(indices),
                    rate,
                )

    if open_arrays:
        raise RuntimeError(
            f"Rank {rank} finished with incomplete open shards: {sorted(open_arrays)}"
        )
    _barrier()

    if rank == 0:
        invalid = []
        for shard in shards:
            path = output_dir / shard["file"]
            expected_bytes = _expected_shard_bytes(shard, latent_shape)
            if not path.is_file() or path.stat().st_size != expected_bytes:
                invalid.append(shard["file"])
        if invalid:
            raise RuntimeError(f"Missing or invalid completed shards: {invalid}")
        shard_starts = [int(shard["start"]) for shard in shards]

        def read_cached_pair(index: int) -> tuple[torch.Tensor, torch.Tensor]:
            shard_index = bisect.bisect_right(shard_starts, index) - 1
            shard = shards[shard_index]
            count = int(shard["end"]) - int(shard["start"])
            array = np.memmap(
                output_dir / shard["file"],
                mode="r",
                dtype=np.uint16,
                shape=(count, 2, *latent_shape),
            )
            raw = np.array(array[index - int(shard["start"])], copy=True)
            del array
            pair = torch.from_numpy(raw).view(torch.bfloat16)
            return pair[0], pair[1]

        verification_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        verification_samples = []
        with torch.inference_mode():
            for index in verification_indices:
                sample = dataset[index]
                online_rgb = vae.encode(
                    sample["video"].unsqueeze(0).to(
                        device=device, dtype=torch.bfloat16
                    ),
                    device=device,
                    tiled=False,
                )[0].cpu()
                online_raymap = vae.encode(
                    sample["raymap"].unsqueeze(0).to(
                        device=device, dtype=torch.bfloat16
                    ),
                    device=device,
                    tiled=False,
                )[0].cpu()
                cached_rgb, cached_raymap = read_cached_pair(index)
                if not torch.equal(online_rgb, cached_rgb) or not torch.equal(
                    online_raymap, cached_raymap
                ):
                    raise RuntimeError(
                        "Latent cache bit-exact verification failed for dataset "
                        f"index {index}."
                    )
                sample["rgb_latents"] = cached_rgb
                sample["raymap_latents"] = cached_raymap
                verification_samples.append(sample)

        # Verify the actual training_loss path as well. The first pass removes
        # cached keys and recomputes VAE latents; the second consumes cache.
        cached_batch = default_collate(verification_samples[:2])
        online_batch = {
            key: value
            for key, value in cached_batch.items()
            if key not in {"rgb_latents", "raymap_latents"}
        }
        model.eval()
        model.dit.train()
        torch.manual_seed(123456)
        torch.cuda.manual_seed_all(123456)
        with torch.no_grad():
            online_loss, online_metrics = model.training_loss(online_batch)
        torch.manual_seed(123456)
        torch.cuda.manual_seed_all(123456)
        with torch.no_grad():
            cached_loss, cached_metrics = model.training_loss(cached_batch)
        if not torch.equal(online_loss, cached_loss) or online_metrics != cached_metrics:
            raise RuntimeError(
                "Latent cache training-loss equivalence failed: "
                f"online_loss={float(online_loss)}, cached_loss={float(cached_loss)}, "
                f"online_metrics={online_metrics}, cached_metrics={cached_metrics}."
            )
        metadata["verification"] = {
            "bit_exact_indices": verification_indices,
            "training_loss_seed": 123456,
            "training_loss_bit_exact": True,
            "loss": float(cached_loss),
        }
        metadata["complete"] = True
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(metadata_path, metadata)
        success_path.write_text("complete\n", encoding="utf-8")
        total_gib = sum((output_dir / shard["file"]).stat().st_size for shard in shards) / (1024**3)
        logger.info(
            "Completed latent cache: %s samples=%d size=%.2f GiB",
            output_dir,
            len(dataset),
            total_gib,
        )
    _barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
