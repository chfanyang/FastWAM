#!/usr/bin/env python
"""Compute full-train-window LIBERO-Plus Rothko translation statistics.

Every frame in every train episode is treated as a window start. Future OSC
targets past the episode end replicate the final action row, exactly matching
BaseLerobotDataset's episode-tail padding behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from fastwam.representations.rothko import RothkoNormStats


STATE_KEY = "observation.state.ee_pose_wxyz"
TARGET_KEY = "action.osc_target_pose_wxyz"
IMAGE_HEIGHT = TILE_HEIGHT = 224
IMAGE_WIDTH = 448
TILE_WIDTH = 224


def _stack(table, key: str) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.astype(np.float64, copy=False)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12
    )
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


def relative_positions_with_tail_replication(
    state_pose_wxyz: np.ndarray,
    target_pose_wxyz: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return [episode_frames, horizon, 3] base-frame translations."""
    state = np.asarray(state_pose_wxyz)
    target = np.asarray(target_pose_wxyz)
    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(f"state pose must be [T,7], got {state.shape}.")
    if target.shape != state.shape:
        raise ValueError(
            f"target pose must match state shape {state.shape}, got {target.shape}."
        )
    if len(state) < 1:
        raise ValueError("Episode must contain at least one frame.")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    starts = np.arange(len(state), dtype=np.int64)
    target_indices = np.minimum(
        starts[:, None] + np.arange(horizon, dtype=np.int64)[None],
        len(state) - 1,
    )
    base_position = state[:, :3].astype(np.float64)
    base_rotation = _quat_to_matrix(state[:, 3:7])
    delta = target[target_indices, :3].astype(np.float64) - base_position[:, None]
    return np.einsum(
        "wij,wtj->wti", base_rotation.transpose(0, 2, 1), delta
    )


def _process_shard(
    args: tuple[
        Sequence[tuple[str, int, int]],
        int,
        int,
        float,
        Sequence[Sequence[float]] | Sequence[float],
    ]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    records, horizon, bins, max_abs, comparison_bounds = args
    histogram = np.zeros((3, bins), dtype=np.int64)
    maxima = np.zeros(3, dtype=np.float64)
    comparison_bounds_array = np.asarray(comparison_bounds, dtype=np.float64)
    if comparison_bounds_array.ndim == 1:
        comparison_bounds_array = comparison_bounds_array[None]
    if comparison_bounds_array.ndim != 2 or comparison_bounds_array.shape[1] != 3:
        raise ValueError(
            "comparison bounds must have shape [3] or [N,3], got "
            f"{comparison_bounds_array.shape}."
        )
    comparison_clipped = np.zeros_like(comparison_bounds_array, dtype=np.int64)
    episodes = windows = overflow = 0
    for path, episode_index, expected_length in records:
        table = pq.read_table(path, columns=[STATE_KEY, TARGET_KEY])
        state = _stack(table, STATE_KEY)
        target = _stack(table, TARGET_KEY)
        if len(state) != expected_length or len(target) != expected_length:
            raise ValueError(
                f"Episode {episode_index} length mismatch: manifest={expected_length}, "
                f"state={len(state)}, target={len(target)}."
            )
        relative = relative_positions_with_tail_replication(
            state, target, horizon
        ).reshape(-1, 3)
        if not np.isfinite(relative).all():
            raise ValueError(f"Episode {episode_index} contains non-finite positions.")
        absolute = np.abs(relative)
        for channel in range(3):
            histogram[channel] += np.histogram(
                absolute[:, channel], bins=bins, range=(0.0, max_abs)
            )[0]
        maxima = np.maximum(maxima, absolute.max(axis=0, initial=0.0))
        for index, bounds in enumerate(comparison_bounds_array):
            comparison_clipped[index] += np.count_nonzero(
                absolute > bounds[None], axis=0
            )
        overflow += int(np.count_nonzero(absolute >= max_abs))
        episodes += 1
        windows += len(state)
    return histogram, maxima, comparison_clipped, episodes, windows, overflow


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


def _tail_ratios(
    histogram: np.ndarray, bounds: np.ndarray, max_abs: float
) -> np.ndarray:
    bin_width = max_abs / histogram.shape[1]
    last_included_bin = np.minimum(
        np.ceil(bounds / bin_width).astype(np.int64) - 1,
        histogram.shape[1] - 1,
    )
    return np.asarray(
        [
            histogram[channel, last_included_bin[channel] + 1 :].sum()
            / histogram[channel].sum()
            for channel in range(3)
        ],
        dtype=np.float64,
    )


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


def _load_split_records(
    dataset_root: Path, split_manifest: Path, split: str
) -> tuple[list[tuple[str, int, int]], str]:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be train or val, got {split!r}.")
    split_bytes = split_manifest.read_bytes()
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    records = []
    for line in split_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["split"] != split:
            continue
        episode_index = int(record["episode_index"])
        path = (
            dataset_root
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append((str(path), episode_index, int(record["episode_length"])))
    if not records:
        raise ValueError(f"No {split} episodes in {split_manifest}.")
    records.sort(key=lambda item: item[1])
    return records, split_sha256


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/libero_plus/libero_plus_lerobot"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(
            "data/libero_plus/libero_plus_lerobot_source_manifest_val02_seed42.jsonl"
        ),
    )
    parser.add_argument(
        "--source-stats",
        type=Path,
        default=Path(
            "data/libero_mujoco3.3.2/"
            "libero_rothko_region_symmetric_q99p95_h16_224x448.pt"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--quantile", type=float, default=0.9995)
    parser.add_argument("--bins", type=int, default=100_000)
    parser.add_argument("--histogram-max", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 16))
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
    if args.horizon < 1 or args.bins < 2 or args.histogram_max <= 0:
        raise ValueError("horizon/bins/histogram-max must be positive.")
    if min(args.center_scale, args.dir_scale, args.focal) <= 0:
        raise ValueError("focal, center-scale, and dir-scale must be positive.")

    quantile_label = (
        f"{args.quantile * 100:.8f}".rstrip("0").rstrip(".")
    ).replace(".", "p")
    output = args.output or (
        args.dataset_root.parent
        / (
            f"libero_plus_rothko_region_symmetric_q{quantile_label}_"
            f"h{args.horizon}_centerfrac05_train_allwindows_224x448.pt"
        )
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    source_stats = RothkoNormStats.load(args.source_stats)
    source_bounds = source_stats.metadata.get("translation_abs_bounds_xyz_m")
    if not isinstance(source_bounds, list) or len(source_bounds) != 3:
        raise ValueError(
            f"Source stats lack translation_abs_bounds_xyz_m: {args.source_stats}"
        )
    source_bounds_array = np.asarray(source_bounds, dtype=np.float64)
    source_file_sha256 = hashlib.sha256(args.source_stats.read_bytes()).hexdigest()

    records, split_sha256 = _load_split_records(
        args.dataset_root.resolve(), args.split_manifest.resolve(), "train"
    )
    worker_count = min(args.workers, len(records))
    shard_size = math.ceil(len(records) / worker_count)
    shards = [records[i : i + shard_size] for i in range(0, len(records), shard_size)]
    jobs = [
        (
            shard,
            args.horizon,
            args.bins,
            args.histogram_max,
            [source_bounds_array.tolist()],
        )
        for shard in shards
    ]
    started = time.time()
    histogram = np.zeros((3, args.bins), dtype=np.int64)
    maxima = np.zeros(3, dtype=np.float64)
    source_clipped = np.zeros((1, 3), dtype=np.int64)
    episodes = windows = overflow = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_process_shard, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            (
                local_histogram,
                local_maxima,
                local_source_clipped,
                local_episodes,
                local_windows,
                local_overflow,
            ) = future.result()
            histogram += local_histogram
            maxima = np.maximum(maxima, local_maxima)
            source_clipped += local_source_clipped
            episodes += local_episodes
            windows += local_windows
            overflow += local_overflow
            print(
                f"shards {completed}/{len(futures)} "
                f"episodes={episodes}/{len(records)} windows={windows}",
                flush=True,
            )
    if overflow:
        raise RuntimeError(
            f"{overflow} values reached histogram-max={args.histogram_max}."
        )
    expected_windows = sum(record[2] for record in records)
    if episodes != len(records) or windows != expected_windows:
        raise RuntimeError(
            f"Coverage mismatch: episodes={episodes}/{len(records)}, "
            f"windows={windows}/{expected_windows}."
        )

    bounds = _bounds(histogram, args.quantile, args.histogram_max)
    candidate_tail_ratio = _tail_ratios(histogram, bounds, args.histogram_max)
    total_per_axis = windows * args.horizon
    source_clipped = source_clipped[0]
    source_clipping_ratio = source_clipped.astype(np.float64) / total_per_axis
    lo, hi = _build_stats(
        bounds,
        args.center_frac,
        args.center_scale,
        args.dir_scale,
        args.outer_margin,
    )
    val_records, val_split_sha256 = _load_split_records(
        args.dataset_root.resolve(), args.split_manifest.resolve(), "val"
    )
    if val_split_sha256 != split_sha256:
        raise RuntimeError("Split manifest changed while statistics were running.")
    (
        _,
        val_maxima,
        val_clipped,
        val_episodes,
        val_windows,
        val_overflow,
    ) = _process_shard(
        (
            val_records,
            args.horizon,
            args.bins,
            args.histogram_max,
            [source_bounds_array.tolist(), bounds.tolist()],
        )
    )
    if val_overflow:
        raise RuntimeError(
            f"{val_overflow} validation values reached "
            f"histogram-max={args.histogram_max}."
        )
    val_total_per_axis = val_windows * args.horizon
    val_source_clipping_ratio = val_clipped[0].astype(np.float64) / val_total_per_axis
    val_candidate_clipping_ratio = (
        val_clipped[1].astype(np.float64) / val_total_per_axis
    )
    elapsed = time.time() - started
    metadata: dict[str, Any] = {
        "stats_format_version": 2,
        # LIBERO-Plus uses the same robot/control representation as LIBERO;
        # keep the codec compatibility key stable and record the data domain
        # separately.
        "environment": "libero",
        "dataset_domain": "libero_plus",
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
        "gripper_conversion": "gripper_open=(1-action_env)/2",
        "dataset_root": str(args.dataset_root.resolve()),
        "split": "train",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": split_sha256,
        "num_episodes": episodes,
        "num_windows": windows,
        "num_relative_vectors": total_per_axis,
        "windows_per_episode": None,
        "sampling": "all_train_frame_starts_with_episode_tail_replication",
        "tail_padding": "future_action_index=min(start+offset,episode_length-1)",
        "histogram_bins": args.bins,
        "histogram_max_abs_m": args.histogram_max,
        "candidate_histogram_tail_ratio_xyz": candidate_tail_ratio.tolist(),
        "source_stats_audit": {
            "path": str(args.source_stats.resolve()),
            "file_sha256": source_file_sha256,
            "codec_fingerprint": source_stats.fingerprint(),
            "translation_abs_bounds_xyz_m": source_bounds_array.tolist(),
            "clipped_count_xyz": source_clipped.tolist(),
            "total_values_per_axis": total_per_axis,
            "clipping_ratio_xyz": source_clipping_ratio.tolist(),
        },
        "held_out_validation_audit": {
            "num_episodes": val_episodes,
            "num_windows": val_windows,
            "num_relative_vectors": val_total_per_axis,
            "observed_max_abs_xyz_m": val_maxima.tolist(),
            "source_stats": {
                "clipped_count_xyz": val_clipped[0].tolist(),
                "clipping_ratio_xyz": val_source_clipping_ratio.tolist(),
            },
            "plus_candidate_stats": {
                "clipped_count_xyz": val_clipped[1].tolist(),
                "clipping_ratio_xyz": val_candidate_clipping_ratio.tolist(),
            },
        },
        "elapsed_sec": elapsed,
    }
    candidate = RothkoNormStats(lo=lo, hi=hi, metadata=metadata)
    metadata["codec_fingerprint"] = candidate.fingerprint()
    payload = {"lo": lo, "hi": hi, "metadata": metadata}
    _atomic_torch_save(output, payload)
    _atomic_write_json(output.with_suffix(".json"), metadata)
    print(
        f"Saved {output}\n"
        f"bounds={bounds.tolist()} max={maxima.tolist()}\n"
        f"source_clipping_ratio={source_clipping_ratio.tolist()}\n"
        f"candidate_fingerprint={metadata['codec_fingerprint']} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
