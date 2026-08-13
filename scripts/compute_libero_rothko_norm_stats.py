#!/usr/bin/env python
"""Compute region-aware Q99.95 stats for the LIBERO single-arm Rothko codec."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch


DATASET_NAMES = (
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)
STATE_KEY = "observation.state.ee_pose_wxyz"
TARGET_KEY = "action.osc_target_pose_wxyz"
IMAGE_HEIGHT = TILE_HEIGHT = 224
IMAGE_WIDTH = 448
TILE_WIDTH = 224


def _stack(table, key: str) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
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


def _uniform_starts(length: int, horizon: int, count: int) -> np.ndarray:
    valid = length - horizon + 1
    if valid <= 0:
        return np.empty(0, dtype=np.int64)
    if valid <= count:
        return np.arange(valid, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, valid - 1, count)).astype(np.int64))


def _process_shard(
    args: tuple[Sequence[str], int, int, int, float]
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    paths, horizon, windows_per_episode, bins, max_abs = args
    histogram = np.zeros((3, bins), dtype=np.int64)
    maxima = np.zeros(3, dtype=np.float64)
    episodes = windows = overflow = 0
    for path in paths:
        table = pq.read_table(path, columns=[STATE_KEY, TARGET_KEY])
        state = _stack(table, STATE_KEY)
        target = _stack(table, TARGET_KEY)
        starts = _uniform_starts(len(state), horizon, windows_per_episode)
        if not len(starts):
            continue
        indices = starts[:, None] + np.arange(horizon)[None]
        base_position = state[starts, :3].astype(np.float64)
        base_rotation = _quat_to_matrix(state[starts, 3:7])
        delta = target[indices, :3].astype(np.float64) - base_position[:, None]
        relative = np.einsum(
            "wij,wtj->wti", base_rotation.transpose(0, 2, 1), delta
        ).reshape(-1, 3)
        absolute = np.abs(relative)
        for channel in range(3):
            histogram[channel] += np.histogram(
                absolute[:, channel], bins=bins, range=(0.0, max_abs)
            )[0]
        maxima = np.maximum(maxima, absolute.max(axis=0, initial=0.0))
        overflow += int(np.count_nonzero(absolute >= max_abs))
        episodes += 1
        windows += len(starts)
    return histogram, maxima, episodes, windows, overflow


def _paths(data_root: Path, limit: int | None) -> tuple[list[str], list[dict[str, Any]]]:
    paths: list[str] = []
    infos = []
    for name in DATASET_NAMES:
        root = data_root / name
        info = json.loads((root / "meta" / "info.json").read_text())
        infos.append(info)
        total = int(info["total_episodes"])
        if limit is not None:
            total = min(total, limit)
        for episode_index in range(total):
            relative = info["data_path"].format(
                episode_chunk=episode_index // int(info["chunks_size"]),
                episode_index=episode_index,
            )
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(str(path))
    return paths, infos


def _bounds(histogram: np.ndarray, quantile: float, max_abs: float) -> np.ndarray:
    cumulative = np.cumsum(histogram, axis=1)
    totals = cumulative[:, -1]
    targets = np.ceil(totals * quantile).astype(np.int64)
    indices = np.asarray(
        [
            np.searchsorted(cumulative[channel], targets[channel], side="left")
            for channel in range(3)
        ]
    )
    return (indices + 1) * max_abs / histogram.shape[1]


def _build_stats(
    bounds: np.ndarray,
    center_frac: float,
    center_scale: float,
    dir_scale: float,
    outer_margin: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.full((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), -float(dir_scale))
    hi = torch.full_like(lo, float(dir_scale))
    center_h = int(round(TILE_HEIGHT * center_frac))
    center_w = int(round(TILE_WIDTH * center_frac))
    y0 = (TILE_HEIGHT - center_h) // 2
    x0 = (TILE_WIDTH - center_w) // 2
    for x_offset in (0, TILE_WIDTH):
        for channel, bound in enumerate(bounds):
            lo[
                0,
                channel,
                y0 : y0 + center_h,
                x_offset + x0 : x_offset + x0 + center_w,
            ] = -float(bound) * float(center_scale)
            hi[
                0,
                channel,
                y0 : y0 + center_h,
                x_offset + x0 : x_offset + x0 + center_w,
            ] = float(bound) * float(center_scale)
        if outer_margin:
            tile_lo = lo[:, :, :, x_offset : x_offset + TILE_WIDTH]
            tile_hi = hi[:, :, :, x_offset : x_offset + TILE_WIDTH]
            tile_lo[:, :, :outer_margin, :] = -1.0
            tile_lo[:, :, -outer_margin:, :] = -1.0
            tile_lo[:, :, :, :outer_margin] = -1.0
            tile_lo[:, :, :, -outer_margin:] = -1.0
            tile_hi[:, :, :outer_margin, :] = 1.0
            tile_hi[:, :, -outer_margin:, :] = 1.0
            tile_hi[:, :, :, :outer_margin] = 1.0
            tile_hi[:, :, :, -outer_margin:] = 1.0
    return lo, hi


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/libero_mujoco3.3.2")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pt path. Defaults to an h{horizon}-specific filename.",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--windows-per-episode", type=int, default=32)
    parser.add_argument("--quantile", type=float, default=0.9995)
    parser.add_argument("--bins", type=int, default=100_000)
    parser.add_argument("--histogram-max", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 16))
    parser.add_argument("--limit", type=int, help="Per-dataset episode limit.")
    parser.add_argument("--focal", type=float, default=0.2)
    parser.add_argument("--center-scale", type=float, default=1.0)
    parser.add_argument("--dir-scale", type=float, default=1.0)
    parser.add_argument("--center-frac", type=float, default=0.5)
    parser.add_argument("--boundary-margin", type=int, default=8)
    parser.add_argument("--outer-margin", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("quantile must be in (0,1).")
    if min(args.center_scale, args.dir_scale, args.focal) <= 0:
        raise ValueError("focal, center-scale, and dir-scale must be positive.")
    quantile_label = (f"{args.quantile * 100:.8f}".rstrip("0").rstrip(".")).replace(
        ".", "p"
    )
    output = args.output or (
        args.data_root
        / f"libero_rothko_region_symmetric_q{quantile_label}_h{args.horizon}_224x448.pt"
    )
    paths, infos = _paths(args.data_root.resolve(), args.limit)
    worker_count = min(args.workers, len(paths))
    shard_size = math.ceil(len(paths) / worker_count)
    shards = [paths[i : i + shard_size] for i in range(0, len(paths), shard_size)]
    started = time.time()
    histogram = np.zeros((3, args.bins), dtype=np.int64)
    maxima = np.zeros(3, dtype=np.float64)
    episodes = windows = overflow = 0
    jobs = [
        (shard, args.horizon, args.windows_per_episode, args.bins, args.histogram_max)
        for shard in shards
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_process_shard, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            local_hist, local_max, local_episodes, local_windows, local_overflow = (
                future.result()
            )
            histogram += local_hist
            maxima = np.maximum(maxima, local_max)
            episodes += local_episodes
            windows += local_windows
            overflow += local_overflow
            print(f"shards {completed}/{len(futures)} episodes={episodes}", flush=True)
    if overflow:
        raise RuntimeError(
            f"{overflow} relative positions reached --histogram-max={args.histogram_max}."
        )
    bounds = _bounds(histogram, args.quantile, args.histogram_max)
    lo, hi = _build_stats(
        bounds,
        args.center_frac,
        args.center_scale,
        args.dir_scale,
        args.outer_margin,
    )
    metadata: dict[str, Any] = {
        "stats_format_version": 2,
        "environment": "libero",
        "representation": "rothko",
        "raymap_representation": "libero_rothko",
        "encoding": "current_ee_plus_future_absolute_osc_targets",
        "layout": "single_arm_duplicated_horizontal",
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "tile_size": [TILE_HEIGHT, TILE_WIDTH],
        "pose_dim": 7,
        "gripper_dim": 1,
        "quaternion_order": "wxyz",
        "action_horizon": args.horizon,
        "pixel_frames": args.horizon + 1,
        "controller_position_scale": 0.05,
        "controller_rotation_scale": 0.5,
        "translation_quantile": args.quantile,
        "translation_abs_bounds_xyz_m": bounds.tolist(),
        "observed_max_abs_xyz_m": maxima.tolist(),
        "focal": args.focal,
        "center_scale": args.center_scale,
        "dir_scale": args.dir_scale,
        "center_frac": args.center_frac,
        "boundary_margin": args.boundary_margin,
        "outer_margin": args.outer_margin,
        "duplicate_horizontal": True,
        "gripper_conversion": "(left_qpos-right_qpos)/(2*0.04m)",
        "dataset_roots": [
            str((args.data_root / name).resolve()) for name in DATASET_NAMES
        ],
        "dataset_codebase_versions": [info.get("codebase_version") for info in infos],
        "num_episodes": episodes,
        "num_windows": windows,
        "num_relative_vectors": windows * args.horizon,
        "windows_per_episode": args.windows_per_episode,
        "sampling": "equal_episode_uniform_valid_windows",
        "histogram_bins": args.bins,
        "histogram_max_abs_m": args.histogram_max,
        "elapsed_sec": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lo": lo, "hi": hi, "metadata": metadata}, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output}; bounds={bounds.tolist()} max={maxima.tolist()}")


if __name__ == "__main__":
    main()
