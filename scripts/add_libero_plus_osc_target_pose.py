#!/usr/bin/env python
"""Add FastWAM OSC/Rothko side channels to packed LIBERO-Plus parquet data.

LIBERO-Plus stores the absolute end-effector state and gripper joints in one
``observation.state`` column with width 8.  Its action gripper uses the
robosuite environment convention ``-1=open, +1=close``.  FastWAM's existing
LIBERO Rothko data path instead uses ``0=closed, 1=open``.  This script adds
explicit side-channel columns without modifying any original column:

* ``observation.state.ee_pose_wxyz``: absolute EE xyz + quaternion(wxyz);
* ``action.osc_target_pose_wxyz``: absolute OSC controller target;
* ``observation.state.gripper_open``: current opening, 0=closed and 1=open;
* ``action.gripper_open``: commanded opening, 0=closed and 1=open.

Always run ``--dry-run --limit 1`` before writing a dataset.  Episode writes
are atomic and metadata is updated only after all requested episodes succeed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastwam.representations.libero_osc import (
    LIBERO_POSITION_SCALE_M,
    LIBERO_ROTATION_SCALE_RAD,
    PANDA_FINGER_TRAVEL_M,
    absolute_target_to_normalized_action,
    ee_state_axis_angle_to_pose_wxyz,
    normalized_action_to_absolute_target,
    panda_gripper_qpos_to_open,
)


STATE_SOURCE_KEY = "observation.state"
ACTION_SOURCE_KEY = "action"
EE_POSE_KEY = "observation.state.ee_pose_wxyz"
TARGET_POSE_KEY = "action.osc_target_pose_wxyz"
STATE_GRIPPER_OPEN_KEY = "observation.state.gripper_open"
ACTION_GRIPPER_OPEN_KEY = "action.gripper_open"
OUTPUT_KEYS = (
    EE_POSE_KEY,
    TARGET_POSE_KEY,
    STATE_GRIPPER_OPEN_KEY,
    ACTION_GRIPPER_OPEN_KEY,
)


@dataclass
class EpisodeResult:
    path: str
    episode_index: int
    rows: int
    status: str
    output_stats: dict[str, dict[str, Any]]
    position_action_error_max: float
    rotation_action_error_max: float


def _stack_list_column(table: pa.Table, key: str, width: int) -> np.ndarray:
    values = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{key} must have shape [N,{width}], got {values.shape}")
    return values


def libero_plus_env_gripper_to_open(action_gripper: torch.Tensor) -> torch.Tensor:
    """Convert robosuite ``-1=open,+1=close`` into ``0=closed,1=open``."""
    return ((1.0 - action_gripper) / 2.0).clamp(0.0, 1.0)


def compute_libero_plus_outputs(
    table: pa.Table,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Compute all Plus side channels while preserving float64 rotations."""
    state = torch.from_numpy(_stack_list_column(table, STATE_SOURCE_KEY, 8)).to(
        torch.float64
    )
    action = torch.from_numpy(_stack_list_column(table, ACTION_SOURCE_KEY, 7)).to(
        torch.float64
    )

    pose = ee_state_axis_angle_to_pose_wxyz(state[..., :6])
    target = normalized_action_to_absolute_target(pose, action)
    state_gripper_open = panda_gripper_qpos_to_open(state[..., 6:8])
    action_gripper_open = libero_plus_env_gripper_to_open(action[..., 6:7])

    recovered = absolute_target_to_normalized_action(pose, target)
    position_error = float((recovered[..., :3] - action[..., :3]).abs().max())
    rotation_error = float((recovered[..., 3:6] - action[..., 3:6]).abs().max())
    tensors = (pose, target, state_gripper_open, action_gripper_open)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("Generated LIBERO-Plus OSC side-channel contains non-finite values.")
    quaternion_norm_error = float((target[..., 3:].norm(dim=-1) - 1.0).abs().max())
    if quaternion_norm_error > 1e-5:
        raise ValueError(f"Target quaternion norm error is {quaternion_norm_error:.3e}")
    if max(position_error, rotation_error) > 2e-5:
        raise ValueError(
            "OSC conversion roundtrip exceeded tolerance: "
            f"position={position_error:.3e}, rotation={rotation_error:.3e}"
        )
    outputs = {
        EE_POSE_KEY: pose.to(torch.float32).numpy(),
        TARGET_POSE_KEY: target.to(torch.float32).numpy(),
        STATE_GRIPPER_OPEN_KEY: state_gripper_open.to(torch.float32).numpy(),
        ACTION_GRIPPER_OPEN_KEY: action_gripper_open.to(torch.float32).numpy(),
    }
    return outputs, position_error, rotation_error


def _episode_stats(values: np.ndarray) -> dict[str, Any]:
    values64 = values.astype(np.float64, copy=False)
    return {
        "min": values64.min(axis=0).tolist(),
        "max": values64.max(axis=0).tolist(),
        "mean": values64.mean(axis=0).tolist(),
        "std": values64.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _list_array(values: np.ndarray) -> pa.Array:
    width = int(values.shape[1])
    flat = pa.array(values.astype(np.float32, copy=False).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _verify_written(
    path: Path, original: pa.Table, expected: dict[str, np.ndarray]
) -> None:
    written = pq.read_table(path)
    if written.num_rows != original.num_rows:
        raise ValueError(f"Row count changed for {path}")
    for key in original.column_names:
        if key not in written.column_names:
            raise ValueError(f"Original column {key!r} disappeared from {path}")
        if not original[key].combine_chunks().equals(written[key].combine_chunks()):
            raise ValueError(f"Original column {key!r} changed in {path}")
    for key, values in expected.items():
        actual = _stack_list_column(written, key, values.shape[1])
        if not np.array_equal(actual, values):
            raise ValueError(f"Written values for {key!r} differ in {path}")


def _process_episode(
    path_string: str,
    episode_index: int,
    *,
    dry_run: bool,
    skip_existing: bool,
    force: bool,
    verify_only: bool,
) -> EpisodeResult:
    path = Path(path_string)
    table = pq.read_table(path)
    present = [key in table.column_names for key in OUTPUT_KEYS]
    if any(present) and not all(present):
        raise ValueError(f"Only part of {OUTPUT_KEYS} exists in {path}")

    expected, position_error, rotation_error = compute_libero_plus_outputs(table)
    status = "dry-run" if dry_run else "written"
    if all(present):
        matches = all(
            np.allclose(
                _stack_list_column(table, key, values.shape[1]),
                values,
                atol=1e-6,
                rtol=1e-6,
            )
            for key, values in expected.items()
        )
        if not matches and not force:
            raise ValueError(f"Existing LIBERO-Plus OSC fields fail verification in {path}")
        if matches and (verify_only or skip_existing or not force):
            status = "verified" if verify_only else "skipped"
            return EpisodeResult(
                str(path),
                episode_index,
                table.num_rows,
                status,
                {key: _episode_stats(value) for key, value in expected.items()},
                position_error,
                rotation_error,
            )
        table = table.select([key for key in table.column_names if key not in OUTPUT_KEYS])
    elif verify_only:
        raise ValueError(f"LIBERO-Plus OSC fields are missing in {path}")

    result = EpisodeResult(
        str(path),
        episode_index,
        table.num_rows,
        status,
        {key: _episode_stats(value) for key, value in expected.items()},
        position_error,
        rotation_error,
    )
    if dry_run:
        return result

    output = table
    for key, values in expected.items():
        output = output.append_column(key, _list_array(values))
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.libero-plus-osc-", suffix=".parquet", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(output, tmp_path, compression="snappy")
        _verify_written(tmp_path, table, expected)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return result


def _feature_specs() -> dict[str, Any]:
    pose_names = {"pose": ["x", "y", "z", "qw", "qx", "qy", "qz"]}
    return {
        EE_POSE_KEY: {
            "dtype": "float32",
            "shape": [7],
            "names": pose_names,
            "info": {
                "quaternion_order": "wxyz",
                "source": f"{STATE_SOURCE_KEY}[:6]",
            },
        },
        TARGET_POSE_KEY: {
            "dtype": "float32",
            "shape": [7],
            "names": pose_names,
            "info": {
                "quaternion_order": "wxyz",
                "controller": "OSC_POSE",
                "control_delta": True,
                "source": f"{ACTION_SOURCE_KEY}[:6]",
                "position_scale_m": LIBERO_POSITION_SCALE_M,
                "rotation_scale_rad": LIBERO_ROTATION_SCALE_RAD,
                "rotation_composition": "R_target = R_delta @ R_current",
            },
        },
        STATE_GRIPPER_OPEN_KEY: {
            "dtype": "float32",
            "shape": [1],
            "names": ["open"],
            "info": {
                "range": [0.0, 1.0],
                "meaning": "0=closed,1=open",
                "source": f"{STATE_SOURCE_KEY}[6:8]",
                "formula": "(left_qpos - right_qpos) / (2 * finger_travel)",
                "finger_travel_m": PANDA_FINGER_TRAVEL_M,
            },
        },
        ACTION_GRIPPER_OPEN_KEY: {
            "dtype": "float32",
            "shape": [1],
            "names": ["open"],
            "info": {
                "range": [0.0, 1.0],
                "meaning": "0=closed,1=open",
                "source": f"{ACTION_SOURCE_KEY}[6]",
                "source_convention": "-1=open,+1=closed",
                "formula": "(1 - action_gripper) / 2",
            },
        },
    }


def _atomic_json(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            if jsonl:
                for item in payload:
                    handle.write(json.dumps(item, separators=(",", ":")) + "\n")
            else:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _episode_paths(dataset_root: Path, limit: int | None) -> list[tuple[int, Path]]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
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
        paths.append((episode_index, path))
    return paths


def _update_metadata(dataset_root: Path, results: list[EpisodeResult]) -> None:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    for path in (info_path, stats_path):
        backup = path.with_name(path.name + ".before_libero_plus_osc_fields")
        if not backup.exists():
            shutil.copy2(path, backup)

    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    for key, spec in _feature_specs().items():
        if key in features and features[key] != spec:
            raise ValueError(f"Metadata for {key!r} already exists but differs")
        features[key] = spec

    result_by_episode = {item.episode_index: item for item in results}
    entries = [
        json.loads(line)
        for line in stats_path.read_text().splitlines()
        if line.strip()
    ]
    seen = set()
    for entry in entries:
        episode_index = int(entry["episode_index"])
        result = result_by_episode.get(episode_index)
        if result is None:
            continue
        entry.setdefault("stats", {}).update(result.output_stats)
        seen.add(episode_index)
    missing = set(result_by_episode) - seen
    if missing:
        raise ValueError(f"episodes_stats.jsonl misses episodes {sorted(missing)[:10]}")
    _atomic_json(info_path, info)
    _atomic_json(stats_path, entries, jsonl=True)


def _verify_metadata(dataset_root: Path, results: list[EpisodeResult]) -> None:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    for key, spec in _feature_specs().items():
        if (info.get("features") or {}).get(key) != spec:
            raise ValueError(f"Metadata verification failed for {key!r}")
    expected = {item.episode_index for item in results}
    seen = set()
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    for line in stats_path.read_text().splitlines():
        entry = json.loads(line)
        episode_index = int(entry["episode_index"])
        if episode_index in expected:
            missing = [key for key in OUTPUT_KEYS if key not in entry.get("stats", {})]
            if missing:
                raise ValueError(f"Episode {episode_index} metadata misses {missing}")
            seen.add(episode_index)
    if seen != expected:
        raise ValueError(f"Metadata misses episodes {sorted(expected - seen)[:10]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/libero_plus/libero_plus_lerobot"),
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 8))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.verify and (args.force or args.dry_run):
        raise ValueError("--verify cannot be combined with --force or --dry-run")
    dataset_root = args.dataset_root.resolve()
    paths = _episode_paths(dataset_root, args.limit)
    print(
        f"dataset={dataset_root} episodes={len(paths)} "
        f"workers={min(args.workers, len(paths))} dry_run={args.dry_run} "
        f"verify={args.verify}",
        flush=True,
    )
    results: list[EpisodeResult] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(paths))) as executor:
        futures = {
            executor.submit(
                _process_episode,
                str(path),
                episode_index,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                force=args.force,
                verify_only=args.verify,
            ): episode_index
            for episode_index, path in paths
        }
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if completed == 1 or completed % 50 == 0 or completed == len(futures):
                print(f"  completed {completed}/{len(futures)}", flush=True)
    results.sort(key=lambda item: item.episode_index)
    if not args.dry_run and not args.verify:
        _update_metadata(dataset_root, results)
    elif args.verify:
        _verify_metadata(dataset_root, results)

    statuses: dict[str, int] = {}
    for item in results:
        statuses[item.status] = statuses.get(item.status, 0) + 1
    print(
        f"Done: episodes={len(results)} rows={sum(item.rows for item in results)} "
        f"statuses={statuses} "
        f"max_position_roundtrip={max(item.position_action_error_max for item in results):.3e} "
        f"max_rotation_roundtrip={max(item.rotation_action_error_max for item in results):.3e}",
        flush=True,
    )
    first = results[0]
    for key in OUTPUT_KEYS:
        stats = first.output_stats[key]
        print(f"episode0 {key}: min={stats['min']} max={stats['max']}", flush=True)


if __name__ == "__main__":
    main()
