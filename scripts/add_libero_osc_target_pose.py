#!/usr/bin/env python
"""Add absolute OSC target poses and gripper openness to LIBERO parquet data.

The script is intentionally conservative:

* original columns are never rewritten in memory;
* each episode is written to a sibling temporary file and verified;
* ``os.replace`` makes the per-episode update atomic;
* partially-present output fields are treated as an error;
* metadata is updated only after every requested episode succeeds.

Use ``--dry-run --limit 1`` before modifying a dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
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
    ee_state_axis_angle_to_pose_wxyz,
    normalized_action_to_absolute_target,
    panda_gripper_qpos_to_open,
)


EE_SOURCE_KEY = "observation.states.ee_state"
GRIPPER_SOURCE_KEY = "observation.states.gripper_state"
ACTION_SOURCE_KEY = "action"
EE_POSE_KEY = "observation.state.ee_pose_wxyz"
TARGET_POSE_KEY = "action.osc_target_pose_wxyz"
GRIPPER_OPEN_KEY = "observation.state.gripper_open"
OUTPUT_KEYS = (EE_POSE_KEY, TARGET_POSE_KEY, GRIPPER_OPEN_KEY)
DEFAULT_DATASET_NAMES = (
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)


@dataclass
class EpisodeResult:
    path: str
    episode_index: int
    rows: int
    status: str
    ee_pose_stats: dict[str, Any] | None = None
    target_pose_stats: dict[str, Any] | None = None
    gripper_open_stats: dict[str, Any] | None = None
    position_action_error_max: float = 0.0
    rotation_action_error_max: float = 0.0


def _stack_list_column(table: pa.Table, key: str, width: int) -> np.ndarray:
    values = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{key} must have shape [N,{width}], got {values.shape}")
    return values


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
    return pa.array(values.astype(np.float32, copy=False).tolist(), type=pa.list_(pa.float32()))


def _compute_outputs(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    # Compute rotations in float64, especially because LIBERO absolute EE
    # rotations frequently sit close to pi.  Persist the resulting fields as
    # float32 to match the rest of the LeRobot dataset.
    ee_state = torch.from_numpy(
        _stack_list_column(table, EE_SOURCE_KEY, 6)
    ).to(torch.float64)
    action = torch.from_numpy(
        _stack_list_column(table, ACTION_SOURCE_KEY, 7)
    ).to(torch.float64)
    gripper_qpos = torch.from_numpy(
        _stack_list_column(table, GRIPPER_SOURCE_KEY, 2)
    ).to(torch.float64)

    pose = ee_state_axis_angle_to_pose_wxyz(ee_state)
    target = normalized_action_to_absolute_target(pose, action)
    gripper_open = panda_gripper_qpos_to_open(gripper_qpos)

    # Verify position directly and rotation by reconstructing the exact
    # controller delta.  Keep this local so conversion cannot silently drift
    # from robosuite semantics.
    from fastwam.representations.libero_osc import absolute_target_to_normalized_action

    recovered = absolute_target_to_normalized_action(pose, target)
    position_error = float((recovered[..., :3] - action[..., :3]).abs().max())
    rotation_error = float((recovered[..., 3:6] - action[..., 3:6]).abs().max())
    if not all(torch.isfinite(x).all() for x in (pose, target, gripper_open)):
        raise ValueError("Generated LIBERO OSC side-channel contains non-finite values.")
    quaternion_norm_error = float((target[..., 3:].norm(dim=-1) - 1.0).abs().max())
    if quaternion_norm_error > 1e-5:
        raise ValueError(f"Target quaternion norm error is {quaternion_norm_error:.3e}")
    if max(position_error, rotation_error) > 2e-5:
        raise ValueError(
            "OSC conversion roundtrip exceeded tolerance: "
            f"position={position_error:.3e}, rotation={rotation_error:.3e}"
        )
    return (
        pose.to(torch.float32).numpy(),
        target.to(torch.float32).numpy(),
        gripper_open.to(torch.float32).numpy(),
        position_error,
        rotation_error,
    )


def _verify_written(path: Path, original: pa.Table, expected: dict[str, np.ndarray]) -> None:
    written = pq.read_table(path)
    if written.num_rows != original.num_rows:
        raise ValueError(
            f"Row count changed for {path}: {written.num_rows} vs {original.num_rows}"
        )
    original_keys = set(original.column_names)
    if not original_keys.issubset(written.column_names):
        raise ValueError(f"Original columns disappeared while writing {path}")
    for key in original.column_names:
        if not original[key].combine_chunks().equals(written[key].combine_chunks()):
            raise ValueError(f"Original column {key!r} changed while writing {path}")
    for key, values in expected.items():
        actual = _stack_list_column(written, key, values.shape[1])
        if not np.array_equal(actual, values.astype(np.float32, copy=False)):
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

    if all(present):
        pose = _stack_list_column(table, EE_POSE_KEY, 7)
        target = _stack_list_column(table, TARGET_POSE_KEY, 7)
        gripper = _stack_list_column(table, GRIPPER_OPEN_KEY, 1)
        expected_pose, expected_target, expected_gripper, pos_err, rot_err = _compute_outputs(table)
        if not (
            np.allclose(pose, expected_pose, atol=1e-6, rtol=1e-6)
            and np.allclose(target, expected_target, atol=1e-6, rtol=1e-6)
            and np.allclose(gripper, expected_gripper, atol=1e-6, rtol=1e-6)
        ):
            if not force:
                raise ValueError(f"Existing LIBERO OSC fields fail verification in {path}")
        elif verify_only or skip_existing or not force:
            return EpisodeResult(
                str(path),
                episode_index,
                table.num_rows,
                "verified" if verify_only else "skipped",
                _episode_stats(pose),
                _episode_stats(target),
                _episode_stats(gripper),
                pos_err,
                rot_err,
            )
        if force:
            keep = [key for key in table.column_names if key not in OUTPUT_KEYS]
            table = table.select(keep)
    elif verify_only:
        raise ValueError(f"LIBERO OSC fields are missing in verify-only mode: {path}")

    pose, target, gripper, pos_err, rot_err = _compute_outputs(table)
    result = EpisodeResult(
        str(path),
        episode_index,
        table.num_rows,
        "dry-run" if dry_run else "written",
        _episode_stats(pose),
        _episode_stats(target),
        _episode_stats(gripper),
        pos_err,
        rot_err,
    )
    if dry_run:
        return result

    output = table
    for key, values in (
        (EE_POSE_KEY, pose),
        (TARGET_POSE_KEY, target),
        (GRIPPER_OPEN_KEY, gripper),
    ):
        output = output.append_column(key, _list_array(values))

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.libero-osc-",
        suffix=".parquet",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(output, tmp_path, compression="snappy")
        _verify_written(
            tmp_path,
            table,
            {EE_POSE_KEY: pose, TARGET_POSE_KEY: target, GRIPPER_OPEN_KEY: gripper},
        )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return result


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


def _feature_specs() -> dict[str, Any]:
    return {
        EE_POSE_KEY: {
            "dtype": "float32",
            "shape": [7],
            "names": {"pose": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
            "info": {
                "quaternion_order": "wxyz",
                "source": EE_SOURCE_KEY,
            },
        },
        TARGET_POSE_KEY: {
            "dtype": "float32",
            "shape": [7],
            "names": {"pose": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
            "info": {
                "quaternion_order": "wxyz",
                "controller": "OSC_POSE",
                "control_delta": True,
                "position_scale_m": LIBERO_POSITION_SCALE_M,
                "rotation_scale_rad": LIBERO_ROTATION_SCALE_RAD,
                "rotation_composition": "R_target = R_delta @ R_current",
            },
        },
        GRIPPER_OPEN_KEY: {
            "dtype": "float32",
            "shape": [1],
            "names": ["open"],
            "info": {
                "range": [0.0, 1.0],
                "meaning": "0=closed,1=open",
                "source": GRIPPER_SOURCE_KEY,
                "formula": "(left_qpos - right_qpos) / (2 * finger_travel)",
                "finger_travel_m": PANDA_FINGER_TRAVEL_M,
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


def _update_metadata(dataset_root: Path, results: list[EpisodeResult]) -> None:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    backup_suffix = ".before_libero_osc_fields"
    for path in (info_path, stats_path):
        backup = path.with_name(path.name + backup_suffix)
        if not backup.exists():
            shutil.copy2(path, backup)

    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    specs = _feature_specs()
    for key, spec in specs.items():
        if key in features and features[key] != spec:
            raise ValueError(f"Metadata for {key!r} already exists but does not match.")
        features[key] = spec

    result_by_episode = {result.episode_index: result for result in results}
    lines = [
        json.loads(line)
        for line in stats_path.read_text().splitlines()
        if line.strip()
    ]
    seen = set()
    for entry in lines:
        episode_index = int(entry["episode_index"])
        result = result_by_episode.get(episode_index)
        if result is None:
            continue
        entry_stats = entry.setdefault("stats", {})
        entry_stats[EE_POSE_KEY] = result.ee_pose_stats
        entry_stats[TARGET_POSE_KEY] = result.target_pose_stats
        entry_stats[GRIPPER_OPEN_KEY] = result.gripper_open_stats
        seen.add(episode_index)
    missing = set(result_by_episode) - seen
    if missing:
        raise ValueError(
            f"episodes_stats.jsonl is missing converted episodes: {sorted(missing)[:10]}"
        )
    _atomic_json(info_path, info)
    _atomic_json(stats_path, lines, jsonl=True)


def _verify_metadata(dataset_root: Path, results: list[EpisodeResult]) -> None:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    info = json.loads(info_path.read_text())
    features = info.get("features") or {}
    for key, spec in _feature_specs().items():
        if features.get(key) != spec:
            raise ValueError(
                f"Metadata verification failed for {key!r} in {info_path}"
            )
    expected_episodes = {result.episode_index for result in results}
    seen = set()
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        episode_index = int(entry["episode_index"])
        if episode_index not in expected_episodes:
            continue
        stats = entry.get("stats") or {}
        missing = [key for key in OUTPUT_KEYS if key not in stats]
        if missing:
            raise ValueError(
                f"Episode {episode_index} metadata is missing fields {missing}"
            )
        seen.add(episode_index)
    if seen != expected_episodes:
        raise ValueError(
            "episodes_stats verification missed episodes: "
            f"{sorted(expected_episodes - seen)[:10]}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/libero_mujoco3.3.2"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset directory name under --data-root; repeat for multiple datasets.",
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 8))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify existing fields only.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.verify and (args.force or args.dry_run):
        raise ValueError("--verify cannot be combined with --force or --dry-run.")
    datasets = args.datasets or list(DEFAULT_DATASET_NAMES)
    all_results: list[EpisodeResult] = []
    for dataset_name in datasets:
        dataset_root = (args.data_root / dataset_name).resolve()
        paths = _episode_paths(dataset_root, args.limit)
        print(
            f"{dataset_name}: episodes={len(paths)} workers={min(args.workers, len(paths))} "
            f"dry_run={args.dry_run} verify={args.verify}",
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
                result = future.result()
                results.append(result)
                if completed == 1 or completed % 50 == 0 or completed == len(futures):
                    print(f"  completed {completed}/{len(futures)}", flush=True)
        results.sort(key=lambda item: item.episode_index)
        if not args.dry_run and not args.verify:
            _update_metadata(dataset_root, results)
        elif args.verify:
            _verify_metadata(dataset_root, results)
        all_results.extend(results)

    statuses: dict[str, int] = {}
    for result in all_results:
        statuses[result.status] = statuses.get(result.status, 0) + 1
    print(
        "Done: "
        f"episodes={len(all_results)} rows={sum(x.rows for x in all_results)} "
        f"statuses={statuses} "
        f"max_position_roundtrip={max(x.position_action_error_max for x in all_results):.3e} "
        f"max_rotation_roundtrip={max(x.rotation_action_error_max for x in all_results):.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
