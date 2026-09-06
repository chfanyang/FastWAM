#!/usr/bin/env python3
"""Profile dataloading, frozen VAE encoding, and DiT forward/backward time."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


def _sync() -> None:
    torch.cuda.synchronize()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--vae-only",
        action="store_true",
        help="Only time dataloading/build_inputs and the two frozen VAE encodes.",
    )
    parser.add_argument(
        "--benchmark-batched-vae",
        action="store_true",
        help="Also call the underlying frozen VAE encoder on the complete batch.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler.")
    register_default_resolvers()
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    overrides = [f"task={args.task}", *args.override]
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    OmegaConf.resolve(cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(args.output.parent)
    dataset = instantiate(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    iterator = iter(loader)
    loader_times: list[float] = []
    batch = None
    for index in range(args.warmup + args.repeats):
        start = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter() - start
        if index >= args.warmup:
            loader_times.append(elapsed)
    assert batch is not None

    model = instantiate(
        cfg.model,
        model_dtype=torch.bfloat16,
        device="cuda",
    )
    model.eval()
    model.requires_grad_(False)
    if not args.vae_only:
        model.dit.train()
        model.dit.requires_grad_(True)

    original_encode = model._encode_video_latents
    active_encode_times: list[float] | None = None

    def timed_encode(video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        nonlocal active_encode_times
        _sync()
        start = time.perf_counter()
        output = original_encode(video, tiled=tiled)
        _sync()
        if active_encode_times is not None:
            active_encode_times.append(time.perf_counter() - start)
        return output

    model._encode_video_latents = timed_encode
    vae_rgb_times: list[float] = []
    vae_ray_times: list[float] = []
    build_input_times: list[float] = []
    cached_inputs = None
    for index in range(args.warmup + args.repeats):
        per_call: list[float] = []
        active_encode_times = per_call
        _sync()
        start = time.perf_counter()
        with torch.no_grad():
            cached_inputs = model.build_inputs(batch)
        _sync()
        elapsed = time.perf_counter() - start
        if len(per_call) != 2:
            raise RuntimeError(f"Expected RGB and Raymap VAE calls, got {len(per_call)}")
        if index >= args.warmup:
            vae_rgb_times.append(per_call[0])
            vae_ray_times.append(per_call[1])
            build_input_times.append(elapsed)
    active_encode_times = None
    assert cached_inputs is not None

    batched_vae: dict[str, float] | None = None
    if args.benchmark_batched_vae:
        rgb = batch["video"].to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        raymap = batch["raymap"].to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        batched_rgb_times: list[float] = []
        batched_ray_times: list[float] = []
        batched_rgb = None
        batched_ray = None
        torch.cuda.reset_peak_memory_stats()
        for index in range(args.warmup + args.repeats):
            _sync()
            start = time.perf_counter()
            with torch.no_grad():
                batched_rgb = model.vae.model.encode(rgb, model.vae.scale)
            _sync()
            rgb_elapsed = time.perf_counter() - start
            start = time.perf_counter()
            with torch.no_grad():
                batched_ray = model.vae.model.encode(raymap, model.vae.scale)
            _sync()
            ray_elapsed = time.perf_counter() - start
            if index >= args.warmup:
                batched_rgb_times.append(rgb_elapsed)
                batched_ray_times.append(ray_elapsed)
        assert batched_rgb is not None and batched_ray is not None
        rgb_diff = (batched_rgb - cached_inputs["rgb_latents"]).abs().float()
        ray_diff = (batched_ray - cached_inputs["raymap_latents"]).abs().float()
        batched_total = _mean(batched_rgb_times) + _mean(batched_ray_times)
        serial_total = _mean(vae_rgb_times) + _mean(vae_ray_times)
        batched_vae = {
            "rgb_seconds": _mean(batched_rgb_times),
            "raymap_seconds": _mean(batched_ray_times),
            "total_seconds": batched_total,
            "speedup_over_serial": serial_total / batched_total,
            "rgb_max_abs_diff": float(rgb_diff.max()),
            "rgb_mean_abs_diff": float(rgb_diff.mean()),
            "raymap_max_abs_diff": float(ray_diff.max()),
            "raymap_mean_abs_diff": float(ray_diff.mean()),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }

    dit_forward_times: list[float] = []
    dit_backward_times: list[float] = []
    peak_memory_bytes: list[int] = []
    if not args.vae_only:
        original_build_inputs = model.build_inputs
        model.build_inputs = lambda sample, tiled=False: cached_inputs
        for index in range(args.warmup + args.repeats):
            model.dit.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            _sync()
            start = time.perf_counter()
            loss, _ = model.training_loss(batch)
            _sync()
            forward_elapsed = time.perf_counter() - start
            start = time.perf_counter()
            loss.backward()
            _sync()
            backward_elapsed = time.perf_counter() - start
            if index >= args.warmup:
                dit_forward_times.append(forward_elapsed)
                dit_backward_times.append(backward_elapsed)
                peak_memory_bytes.append(torch.cuda.max_memory_allocated())
        model.build_inputs = original_build_inputs

    vae_total = _mean(vae_rgb_times) + _mean(vae_ray_times)
    dit_total = (
        _mean(dit_forward_times) + _mean(dit_backward_times)
        if dit_forward_times
        else None
    )
    measured_total = vae_total + dit_total if dit_total is not None else None
    result = {
        "task": args.task,
        "overrides": overrides,
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dataset_size": len(dataset),
        "mean_seconds": {
            "dataloader": _mean(loader_times),
            "build_inputs": _mean(build_input_times),
            "vae_rgb": _mean(vae_rgb_times),
            "vae_raymap": _mean(vae_ray_times),
            "vae_total": vae_total,
            "dit_forward_without_vae": _mean(dit_forward_times) if dit_forward_times else None,
            "dit_backward": _mean(dit_backward_times) if dit_backward_times else None,
            "dit_total": dit_total,
            "vae_plus_dit": measured_total,
        },
        "fractions": {
            "vae_of_vae_plus_dit": vae_total / measured_total if measured_total else None,
            "dit_of_vae_plus_dit": dit_total / measured_total if measured_total else None,
        },
        "batched_vae": batched_vae,
        "peak_memory_gib": (
            max(peak_memory_bytes) / (1024**3)
            if peak_memory_bytes
            else torch.cuda.max_memory_allocated() / (1024**3)
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
