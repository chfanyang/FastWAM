#!/usr/bin/env python
"""Compute Q99.95 region-aware Rothko normalization stats for FastWAM.

Each sampled training window is aligned exactly like the video-only policy:

* frame 0 is ``observation.state.endpose[t]``;
* frames 1..H are ``action.endpose[t:t+H]``;
* both arms are expressed relative to the frame-0 EE coordinate frame.

Only the center-region translation distribution needs to be estimated.
Peripheral ray directions have a known ``[-1, 1]`` range.  The output covers
the new 384 x 320 duplicated layout and records the codec parameters in its
metadata so incompatible checkpoints fail loudly at load time.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


ARM_HEIGHT = 192
ARM_WIDTH = 160
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 320


def _stack_column(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float32) for value in series.values])


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.astype(np.float64, copy=False)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _uniform_starts(
    num_frames: int, horizon: int, windows_per_episode: int
) -> np.ndarray:
    count = num_frames - horizon + 1
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= windows_per_episode:
        return np.arange(count, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, count - 1, windows_per_episode)).astype(np.int64)
    )


def _relative_future_positions(
    state_endpose: np.ndarray,
    action_endpose: np.ndarray,
    starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    future_indices = starts[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    output = []
    for offset in (0, 7):
        base_position = state_endpose[starts, offset : offset + 3].astype(
            np.float64, copy=False
        )
        base_rotation = _quat_wxyz_to_matrix(
            state_endpose[starts, offset + 3 : offset + 7]
        )
        future_position = action_endpose[
            future_indices, offset : offset + 3
        ].astype(np.float64, copy=False)
        delta = future_position - base_position[:, None]
        output.append(
            np.einsum("wij,wtj->wti", base_rotation.transpose(0, 2, 1), delta)
        )
    return np.concatenate(output, axis=0).reshape(-1, 3)


def _histogram(
    values: np.ndarray, bins: int, max_abs: float
) -> tuple[np.ndarray, np.ndarray, int]:
    absolute = np.abs(values)
    histogram = np.empty((3, bins), dtype=np.int64)
    for channel in range(3):
        histogram[channel] = np.histogram(
            absolute[:, channel], bins=bins, range=(0.0, max_abs)
        )[0]
    maxima = absolute.max(axis=0, initial=0.0)
    overflow = int(np.count_nonzero(absolute >= max_abs))
    return histogram, maxima, overflow


def _process_shard(
    arguments: tuple[Sequence[str], int, int, int, float]
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    paths, horizon, windows_per_episode, bins, max_abs = arguments
    histogram = np.zeros((3, bins), dtype=np.int64)
    maxima = np.zeros(3, dtype=np.float64)
    episodes_used = windows_used = overflow = 0
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["action.endpose", "observation.state.endpose"],
        )
        action_endpose = _stack_column(frame["action.endpose"])
        state_endpose = _stack_column(frame["observation.state.endpose"])
        starts = _uniform_starts(len(frame), horizon, windows_per_episode)
        if not len(starts):
            continue
        values = _relative_future_positions(
            state_endpose, action_endpose, starts, horizon
        )
        local_histogram, local_maxima, local_overflow = _histogram(
            values, bins, max_abs
        )
        histogram += local_histogram
        maxima = np.maximum(maxima, local_maxima)
        overflow += local_overflow
        episodes_used += 1
        windows_used += len(starts)
    return histogram, maxima, episodes_used, windows_used, overflow


def _episode_paths(
    dataset_root: Path, limit: int | None
) -> tuple[list[str], dict[str, Any]]:
    with (dataset_root / "meta" / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    total = int(info["total_episodes"])
    if limit is not None:
        total = min(total, limit)
    paths = []
    for episode_index in range(total):
        relative = info["data_path"].format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            episode_index=episode_index,
        )
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path))
    return paths, info


def _quantile_bounds(
    histogram: np.ndarray, quantile: float, max_abs: float
) -> np.ndarray:
    cumulative = np.cumsum(histogram, axis=1)
    totals = cumulative[:, -1]
    if np.any(totals == 0):
        raise RuntimeError("At least one Rothko translation channel has no samples.")
    targets = np.ceil(totals * quantile).astype(np.int64)
    indices = np.asarray(
        [
            np.searchsorted(cumulative[channel], targets[channel], side="left")
            for channel in range(3)
        ]
    )
    return (indices + 1) * (max_abs / histogram.shape[1])


def _center_mask(center_frac: float) -> torch.Tensor:
    center_height = max(1, int(round(ARM_HEIGHT * center_frac)))
    center_width = max(1, int(round(ARM_WIDTH * center_frac)))
    y0 = (ARM_HEIGHT - center_height) // 2
    x0 = (ARM_WIDTH - center_width) // 2
    mask = torch.zeros(ARM_HEIGHT, ARM_WIDTH, dtype=torch.bool)
    mask[y0 : y0 + center_height, x0 : x0 + center_width] = True
    return mask


def _build_stats(bounds: np.ndarray, center_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.full((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), -1.0)
    hi = torch.full_like(lo, 1.0)
    center = _center_mask(center_frac)
    for y_offset in (0, ARM_HEIGHT):
        for x_offset in (0, ARM_WIDTH):
            for channel, bound in enumerate(bounds):
                lo_tile = lo[
                    0,
                    channel,
                    y_offset : y_offset + ARM_HEIGHT,
                    x_offset : x_offset + ARM_WIDTH,
                ]
                hi_tile = hi[
                    0,
                    channel,
                    y_offset : y_offset + ARM_HEIGHT,
                    x_offset : x_offset + ARM_WIDTH,
                ]
                lo_tile[center] = -float(bound)
                hi_tile[center] = float(bound)
    return lo, hi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/robotwin2.0/robotwin2.0"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/robotwin2.0/"
            "rothko_region_symmetric_q99p95_h16_384x320.pt"
        ),
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--windows-per-episode", type=int, default=32)
    parser.add_argument("--quantile", type=float, default=0.9995)
    parser.add_argument("--bins", type=int, default=100_000)
    parser.add_argument("--histogram-max", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(16, os_cpu_count()))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--focal", type=float, default=0.2)
    parser.add_argument("--center-scale", type=float, default=1.0)
    parser.add_argument("--dir-scale", type=float, default=1.0)
    parser.add_argument("--center-frac", type=float, default=0.5)
    parser.add_argument("--boundary-margin", type=int, default=8)
    parser.add_argument("--outer-margin", type=int, default=8)
    return parser


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 8


def main() -> None:
    args = build_parser().parse_args()
    if min(
        args.horizon,
        args.windows_per_episode,
        args.bins,
        args.workers,
    ) < 1:
        raise ValueError("horizon, windows, bins, and workers must be positive.")
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("quantile must be in (0,1).")
    if args.histogram_max <= 0:
        raise ValueError("histogram-max must be positive.")

    dataset_root = args.dataset_root.resolve()
    paths, info = _episode_paths(dataset_root, args.limit)
    worker_count = min(args.workers, len(paths))
    shard_size = math.ceil(len(paths) / worker_count)
    shards = [paths[index : index + shard_size] for index in range(0, len(paths), shard_size)]
    jobs = [
        (
            shard,
            args.horizon,
            args.windows_per_episode,
            args.bins,
            args.histogram_max,
        )
        for shard in shards
    ]

    print(
        "Computing FastWAM Rothko normalization stats:\n"
        f"  dataset={dataset_root}\n"
        f"  episodes={len(paths)} workers={len(jobs)}\n"
        f"  horizon={args.horizon} windows_per_episode={args.windows_per_episode}\n"
        f"  quantile={args.quantile} histogram=[0,{args.histogram_max}]/{args.bins}",
        flush=True,
    )
    started = time.time()
    histogram = np.zeros((3, args.bins), dtype=np.int64)
    maxima = np.zeros(3, dtype=np.float64)
    episodes_used = windows_used = overflow = 0
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(_process_shard, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            local_histogram, local_maxima, local_episodes, local_windows, local_overflow = (
                future.result()
            )
            histogram += local_histogram
            maxima = np.maximum(maxima, local_maxima)
            episodes_used += local_episodes
            windows_used += local_windows
            overflow += local_overflow
            print(
                f"  shards {completed}/{len(jobs)}; episodes={episodes_used}",
                flush=True,
            )
    if overflow:
        raise RuntimeError(
            f"{overflow} values reached histogram-max={args.histogram_max}; "
            "rerun with a larger range."
        )

    bounds = _quantile_bounds(histogram, args.quantile, args.histogram_max)
    lo, hi = _build_stats(bounds, args.center_frac)
    elapsed = time.time() - started
    metadata: dict[str, Any] = {
        "representation": "rothko",
        "encoding": "state_frame0_plus_future_action_endpose",
        "quaternion_order": "wxyz",
        "dataset_root": str(dataset_root),
        "dataset_codebase_version": info.get("codebase_version"),
        "action_horizon": args.horizon,
        "pixel_frames": args.horizon + 1,
        "sampling": "equal_episode_uniform_valid_windows",
        "windows_per_episode": args.windows_per_episode,
        "translation_quantile": args.quantile,
        "translation_abs_bounds_xyz_m": bounds.tolist(),
        "observed_max_abs_xyz_m": maxima.tolist(),
        "focal": args.focal,
        "center_scale": args.center_scale,
        "dir_scale": args.dir_scale,
        "center_frac": args.center_frac,
        "boundary_margin": args.boundary_margin,
        "outer_margin": args.outer_margin,
        "duplicate_vertical": True,
        "gripper_encoding": "normalized_outer_border_2g_minus_1",
        "active_shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "stats_shape": list(lo.shape),
        "num_episodes_requested": len(paths),
        "num_episodes_used": episodes_used,
        "num_windows": windows_used,
        "num_relative_vectors": windows_used * args.horizon * 2,
        "histogram_bins": args.bins,
        "histogram_max_abs_m": args.histogram_max,
        "histogram_resolution_m": args.histogram_max / args.bins,
        "elapsed_sec": elapsed,
        "num_workers": len(jobs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lo": lo, "hi": hi, "metadata": metadata}, args.output)
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved {args.output}", flush=True)
    print(f"  Q{args.quantile * 100:g} bounds xyz={bounds.tolist()} m", flush=True)
    print(f"  observed max xyz={maxima.tolist()} m; elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
