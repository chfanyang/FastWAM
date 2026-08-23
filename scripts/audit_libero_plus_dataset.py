#!/usr/bin/env python
"""Read-only structural and control-semantics audit for LIBERO-Plus LeRobot v2.1.

The script never rewrites parquet, metadata, or videos. It validates the full
tabular dataset, quantifies normalized OSC response and gripper semantics, and
optionally probes one front/wrist video pair per task with ffprobe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import torch

from fastwam.representations.libero_osc import (
    LIBERO_POSITION_SCALE_M,
    absolute_target_to_normalized_action,
    ee_state_axis_angle_to_pose_wxyz,
    normalized_action_to_absolute_target,
    panda_gripper_qpos_to_open,
)


CORE_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fixed_list_numpy(table, key: str, width: int) -> np.ndarray:
    column = table[key].combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False)
    result = np.asarray(values).reshape(-1, width)
    return result


def _scalar_numpy(table, key: str) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_numpy(zero_copy_only=False))


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    probabilities = np.array(
        [0.0, 0.0005, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9995, 1.0],
        dtype=np.float64,
    )
    result = np.quantile(values.astype(np.float64, copy=False), probabilities, axis=0)
    return {
        f"q{probability:.4f}": row.tolist()
        for probability, row in zip(probabilities, result, strict=True)
    }


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> list[float | None]:
    correlations: list[float | None] = []
    for dim in range(x.shape[1]):
        x_dim = x[:, dim].astype(np.float64, copy=False)
        y_dim = y[:, dim].astype(np.float64, copy=False)
        if float(x_dim.std()) == 0.0 or float(y_dim.std()) == 0.0:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(x_dim, y_dim)[0, 1]))
    return correlations


def _count_trailing(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    false_positions = np.flatnonzero(~mask)
    return int(mask.size if false_positions.size == 0 else mask.size - false_positions[-1] - 1)


def _source_trajectory_audit(
    *,
    state: np.ndarray,
    action: np.ndarray,
    episodes: list[dict[str, Any]],
    offsets: np.ndarray,
    task_text_to_index: dict[str, int],
    suite_by_task: dict[str, str],
    validation_group_proportion: float,
    split_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recover replay groups from exact state/action trajectory identity.

    LIBERO-Plus does not publish a source trajectory identifier. Exact hashes
    are nevertheless a reliable local identifier when both state and action
    arrays are byte-identical across visual replays.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    state_hash_by_action_hash: dict[str, set[str]] = defaultdict(set)
    task_by_action_hash: dict[str, set[str]] = defaultdict(set)
    episode_hash: dict[int, str] = {}
    for episode, offset in zip(episodes, offsets, strict=True):
        length = int(episode["length"])
        episode_state = np.ascontiguousarray(state[offset : offset + length], dtype="<f4")
        episode_action = np.ascontiguousarray(action[offset : offset + length], dtype="<f4")
        action_hash = hashlib.sha256(episode_action.tobytes()).hexdigest()
        state_hash = hashlib.sha256(episode_state.tobytes()).hexdigest()
        groups[action_hash].append(int(episode["episode_index"]))
        episode_hash[int(episode["episode_index"])] = action_hash
        state_hash_by_action_hash[action_hash].add(state_hash)
        task_by_action_hash[action_hash].add(str(episode["tasks"][0]))

    inconsistent_state_groups = int(
        sum(len(values) != 1 for values in state_hash_by_action_hash.values())
    )
    cross_task_groups = int(sum(len(values) != 1 for values in task_by_action_hash.values()))
    if inconsistent_state_groups or cross_task_groups:
        raise ValueError(
            "Exact action replay groups are not valid source groups: "
            f"nonidentical_state={inconsistent_state_groups}, cross_task={cross_task_groups}"
        )

    hashes_by_task: dict[str, list[str]] = defaultdict(list)
    for action_hash, task_values in task_by_action_hash.items():
        hashes_by_task[next(iter(task_values))].append(action_hash)
    validation_hashes: set[str] = set()
    validation_groups_by_task: dict[str, int] = {}
    for task, source_hashes in hashes_by_task.items():
        ordered = sorted(
            source_hashes,
            key=lambda value: hashlib.sha256(
                f"{split_seed}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        validation_count = max(1, int(round(len(ordered) * validation_group_proportion)))
        validation_count = min(validation_count, len(ordered) - 1)
        validation_hashes.update(ordered[:validation_count])
        validation_groups_by_task[task] = validation_count

    manifest = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        task = str(episode["tasks"][0])
        source_hash = episode_hash[episode_index]
        manifest.append(
            {
                "episode_index": episode_index,
                "source_trajectory_id": source_hash,
                "split": "val" if source_hash in validation_hashes else "train",
                "task_index": int(task_text_to_index[task]),
                "task": task,
                "suite": suite_by_task.get(task),
                "episode_length": int(episode["length"]),
                "replay_group_size": len(groups[source_hash]),
            }
        )

    group_size_counts = Counter(len(indices) for indices in groups.values())
    split_episode_counts = Counter(item["split"] for item in manifest)
    summary = {
        "source_id_definition": "sha256 of exact little-endian float32 action trajectory",
        "unique_source_trajectories": len(groups),
        "episodes": len(episodes),
        "mean_visual_replays_per_source": float(len(episodes) / len(groups)),
        "group_size_distribution": {
            str(size): int(count) for size, count in sorted(group_size_counts.items())
        },
        "groups_with_nonidentical_state": inconsistent_state_groups,
        "groups_crossing_task_text": cross_task_groups,
        "episodes_in_replayed_groups": int(
            sum(len(indices) for indices in groups.values() if len(indices) > 1)
        ),
        "fixed_group_split": {
            "algorithm": "per-task keyed SHA256 ordering of source trajectory IDs",
            "seed": split_seed,
            "requested_validation_group_proportion": validation_group_proportion,
            "train_source_groups": len(groups) - len(validation_hashes),
            "val_source_groups": len(validation_hashes),
            "train_episodes": int(split_episode_counts["train"]),
            "val_episodes": int(split_episode_counts["val"]),
            "validation_groups_per_task_min": min(validation_groups_by_task.values()),
            "validation_groups_per_task_max": max(validation_groups_by_task.values()),
        },
        "example_groups": [
            {
                "source_trajectory_id": action_hash,
                "episode_indices": indices,
                "task": next(iter(task_by_action_hash[action_hash])),
            }
            for action_hash, indices in sorted(
                groups.items(), key=lambda item: (-len(item[1]), item[0])
            )[:10]
        ],
    }
    return summary, manifest


def _position_alignment_audit(
    *,
    state: np.ndarray,
    action: np.ndarray,
    episode_index: np.ndarray,
) -> dict[str, Any]:
    """Compare nearby action rows to the observed t->t+1 EE response.

    Smooth demonstrations and controller dynamics make this diagnostic rather
    than a proof of pre-step/post-step row semantics.
    """
    response = (state[1:, :3] - state[:-1, :3]) / float(LIBERO_POSITION_SCALE_M)
    transition_start = np.arange(response.shape[0], dtype=np.int64)
    same_episode = episode_index[:-1] == episode_index[1:]
    records = []
    for offset in range(-4, 5):
        action_index = transition_start + offset
        clipped = np.clip(action_index, 0, action.shape[0] - 1)
        valid = (
            same_episode
            & (action_index >= 0)
            & (action_index < action.shape[0])
            & (episode_index[clipped] == episode_index[:-1])
        )
        command = action[action_index[valid], :3]
        observed = response[valid]
        records.append(
            {
                "action_row_offset_from_transition_start": offset,
                "samples": int(command.shape[0]),
                "correlation_xyz": _safe_corrcoef(command, observed),
                "mae_xyz": np.mean(np.abs(command - observed), axis=0).tolist(),
                "mse_xyz": np.mean(np.square(command - observed), axis=0).tolist(),
                "mean_mse": float(np.mean(np.square(command - observed))),
            }
        )
    return {
        "definition": "response=(state[t+1]-state[t])/0.05; compare action[t+offset]",
        "warning": (
            "Cross-correlation cannot by itself prove row semantics because expert actions "
            "are smooth and the OSC controller has dynamics. Confirm with converter/replay source."
        ),
        "offsets": records,
    }


def _suite_task_map(original_data_root: Path | None) -> dict[str, str]:
    if original_data_root is None:
        return {}
    result: dict[str, str] = {}
    for path in sorted(original_data_root.glob("libero_*_no_noops_lerobot/meta/tasks.jsonl")):
        suite = path.parent.parent.name.removesuffix("_no_noops_lerobot")
        for entry in _load_jsonl(path):
            task = str(entry["task"])
            if task in result and result[task] != suite:
                raise ValueError(f"Task {task!r} appears in multiple suites.")
            result[task] = suite
    return result


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    streams = json.loads(completed.stdout).get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}, got {len(streams)}")
    return streams[0]


def _sample_video_audit(
    *,
    root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    task_text_to_index: dict[str, int],
) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        return {"skipped": "ffprobe is not available"}
    chosen: dict[int, dict[str, Any]] = {}
    for episode in episodes:
        task_index = task_text_to_index[str(episode["tasks"][0])]
        chosen.setdefault(task_index, episode)
    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    chunk_size = int(info["chunks_size"])
    for task_index, episode in sorted(chosen.items()):
        episode_index = int(episode["episode_index"])
        expected_frames = int(episode["length"])
        chunk = episode_index // chunk_size
        for camera in ("observation.images.front", "observation.images.wrist"):
            relative = info["video_path"].format(
                episode_chunk=chunk,
                video_key=camera,
                episode_index=episode_index,
            )
            path = root / relative
            try:
                stream = _probe_video(path)
                actual_frames = int(stream["nb_frames"])
                record = {
                    "task_index": task_index,
                    "episode_index": episode_index,
                    "camera": camera,
                    "path": str(path),
                    "expected_frames": expected_frames,
                    "actual_frames": actual_frames,
                    "codec": stream.get("codec_name"),
                    "width": int(stream["width"]),
                    "height": int(stream["height"]),
                    "avg_frame_rate": stream.get("avg_frame_rate"),
                }
                records.append(record)
                if actual_frames != expected_frames:
                    failures.append(record)
            except Exception as exc:  # report every broken sample in one run
                failures.append(
                    {
                        "task_index": task_index,
                        "episode_index": episode_index,
                        "camera": camera,
                        "path": str(path),
                        "error": repr(exc),
                    }
                )
    return {
        "sampled_tasks": len(chosen),
        "sampled_videos": len(records),
        "failures": failures,
        "records": records,
    }


def _rotation_response_audit(
    state: np.ndarray,
    action: np.ndarray,
    transition_indices: np.ndarray,
    max_samples: int,
) -> dict[str, Any]:
    if transition_indices.size > max_samples:
        selected = np.linspace(
            0, transition_indices.size - 1, num=max_samples, dtype=np.int64
        )
        transition_indices = transition_indices[selected]
    command_chunks: list[np.ndarray] = []
    response_chunks: list[np.ndarray] = []
    target_position_error: list[np.ndarray] = []
    target_rotation_error: list[np.ndarray] = []
    chunk_size = 50_000
    for start in range(0, transition_indices.size, chunk_size):
        indices = transition_indices[start : start + chunk_size]
        current_state = torch.from_numpy(state[indices, :6]).to(torch.float64)
        next_state = torch.from_numpy(state[indices + 1, :6]).to(torch.float64)
        command = torch.from_numpy(action[indices]).to(torch.float64)
        current_pose = ee_state_axis_angle_to_pose_wxyz(current_state)
        next_pose = ee_state_axis_angle_to_pose_wxyz(next_state)
        target_pose = normalized_action_to_absolute_target(current_pose, command)
        response = absolute_target_to_normalized_action(current_pose, next_pose)
        target_to_next = absolute_target_to_normalized_action(next_pose, target_pose)
        command_chunks.append(command[:, 3:6].numpy())
        response_chunks.append(response[:, 3:6].numpy())
        target_position_error.append(
            torch.linalg.vector_norm(target_pose[:, :3] - next_pose[:, :3], dim=-1).numpy()
        )
        target_rotation_error.append(
            torch.linalg.vector_norm(target_to_next[:, 3:6], dim=-1).mul(0.5).numpy()
        )
    command_rotation = np.concatenate(command_chunks)
    response_rotation = np.concatenate(response_chunks)
    position_error = np.concatenate(target_position_error)
    rotation_error = np.concatenate(target_rotation_error)
    return {
        "samples": int(transition_indices.size),
        "command_vs_next_rotation_response_corr": _safe_corrcoef(
            command_rotation, response_rotation
        ),
        "target_to_next_position_error_m": {
            "mean": float(position_error.mean()),
            "median": float(np.median(position_error)),
            "q99": float(np.quantile(position_error, 0.99)),
        },
        "target_to_next_rotation_geodesic_rad": {
            "mean": float(rotation_error.mean()),
            "median": float(np.median(rotation_error)),
            "q99": float(np.quantile(rotation_error, 0.99)),
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.resolve()
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    tasks = _load_jsonl(meta / "tasks.jsonl")
    episodes = _load_jsonl(meta / "episodes.jsonl")
    task_text_to_index = {str(item["task"]): int(item["task_index"]) for item in tasks}
    suite_by_task = _suite_task_map(args.original_data_root)

    if len(tasks) != int(info["total_tasks"]):
        raise ValueError("tasks.jsonl count does not match info.json")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("episodes.jsonl count does not match info.json")

    expected_lengths = np.asarray([int(item["length"]) for item in episodes], dtype=np.int64)
    expected_tasks = np.asarray(
        [task_text_to_index[str(item["tasks"][0])] for item in episodes], dtype=np.int64
    )
    expected_offsets = np.concatenate(([0], np.cumsum(expected_lengths[:-1])))

    dataset = pads.dataset(root / "data", format="parquet")
    table = dataset.to_table(columns=list(CORE_COLUMNS), use_threads=True)
    state = _fixed_list_numpy(table, "observation.state", 8).astype(np.float32, copy=False)
    action = _fixed_list_numpy(table, "action", 7).astype(np.float32, copy=False)
    timestamp = _scalar_numpy(table, "timestamp").astype(np.float64, copy=False)
    frame_index = _scalar_numpy(table, "frame_index").astype(np.int64, copy=False)
    episode_index = _scalar_numpy(table, "episode_index").astype(np.int64, copy=False)
    global_index = _scalar_numpy(table, "index").astype(np.int64, copy=False)
    task_index = _scalar_numpy(table, "task_index").astype(np.int64, copy=False)

    if table.num_rows != int(info["total_frames"]):
        raise ValueError(f"Expected {info['total_frames']} rows, got {table.num_rows}")
    order = np.argsort(global_index, kind="stable")
    already_sorted = bool(np.array_equal(order, np.arange(order.size)))
    if not already_sorted:
        state = state[order]
        action = action[order]
        timestamp = timestamp[order]
        frame_index = frame_index[order]
        episode_index = episode_index[order]
        global_index = global_index[order]
        task_index = task_index[order]

    expected_global_index = np.arange(global_index.size, dtype=np.int64)
    global_index_mismatches = int(np.count_nonzero(global_index != expected_global_index))
    expected_episode_rows = np.repeat(np.arange(len(episodes), dtype=np.int64), expected_lengths)
    expected_frame_rows = np.concatenate(
        [np.arange(length, dtype=np.int64) for length in expected_lengths]
    )
    expected_task_rows = np.repeat(expected_tasks, expected_lengths)
    timestamp_error = np.abs(timestamp - expected_frame_rows / float(info["fps"]))

    same_episode_next = episode_index[:-1] == episode_index[1:]
    transition_indices = np.flatnonzero(same_episode_next)
    command_position = action[transition_indices, :3]
    response_position = (
        state[transition_indices + 1, :3] - state[transition_indices, :3]
    ) / float(LIBERO_POSITION_SCALE_M)

    pose_action_norm = np.linalg.norm(action[:, :6], axis=-1)
    noop_thresholds = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)
    noop = {
        f"le_{threshold:g}": {
            "count": int(np.count_nonzero(pose_action_norm <= threshold)),
            "ratio": float(np.mean(pose_action_norm <= threshold)),
        }
        for threshold in noop_thresholds
    }
    trailing_noop: dict[str, list[int]] = {f"le_{value:g}": [] for value in noop_thresholds}
    for offset, length in zip(expected_offsets, expected_lengths, strict=True):
        values = pose_action_norm[offset : offset + length]
        for threshold in noop_thresholds:
            trailing_noop[f"le_{threshold:g}"].append(
                _count_trailing(values <= threshold)
            )

    gripper_qpos = torch.from_numpy(np.array(state[:, 6:8], copy=True)).to(torch.float64)
    gripper_open = panda_gripper_qpos_to_open(gripper_qpos).squeeze(-1).numpy()
    gripper_values, gripper_counts = np.unique(action[:, 6], return_counts=True)
    gripper_mapping: dict[str, Any] = {}
    for lag in (1, 2, 4, 8):
        valid = (
            np.arange(action.shape[0] - lag, dtype=np.int64)
        )
        valid = valid[episode_index[valid] == episode_index[valid + lag]]
        command = action[valid, 6]
        future_open = gripper_open[valid + lag]
        expected_open = np.clip((1.0 - command) / 2.0, 0.0, 1.0)
        reversed_open = np.clip((1.0 + command) / 2.0, 0.0, 1.0)
        gripper_mapping[f"lag_{lag}"] = {
            "samples": int(valid.size),
            "mae_minus1_open_plus1_close": float(np.mean(np.abs(future_open - expected_open))),
            "mae_plus1_open_minus1_close": float(np.mean(np.abs(future_open - reversed_open))),
            "mean_future_open_when_action_minus1": float(future_open[command < 0].mean()),
            "mean_future_open_when_action_plus1": float(future_open[command > 0].mean()),
        }

    task_episode_counts = Counter(expected_tasks.tolist())
    task_frame_counts = Counter(task_index.tolist())
    task_valid_window_counts: Counter[int] = Counter()
    for task, length in zip(expected_tasks, expected_lengths, strict=True):
        task_valid_window_counts[int(task)] += max(int(length) - 16, 0)
    task_distribution = []
    for item in sorted(tasks, key=lambda value: int(value["task_index"])):
        index = int(item["task_index"])
        text = str(item["task"])
        task_distribution.append(
            {
                "task_index": index,
                "task": text,
                "suite": suite_by_task.get(text),
                "episodes": int(task_episode_counts[index]),
                "frames": int(task_frame_counts[index]),
                "complete_17_frame_windows": int(task_valid_window_counts[index]),
                "training_sample_starts_with_current_padding_policy": int(
                    task_frame_counts[index]
                ),
                "episode_tail_padded_sample_starts": int(
                    task_frame_counts[index] - task_valid_window_counts[index]
                ),
            }
        )

    published_norm_stats_path = root / "norm_stats.json"
    published_norm_stats: dict[str, Any]
    if published_norm_stats_path.is_file():
        payload = json.loads(published_norm_stats_path.read_text(encoding="utf-8"))
        stats = payload.get("norm_stats") or {}
        published_state_mean = np.asarray(stats.get("state", {}).get("mean", []))
        published_action_mean = np.asarray(stats.get("actions", {}).get("mean", []))
        published_norm_stats = {
            "path": str(published_norm_stats_path),
            "state_mean_matches_parquet": bool(
                published_state_mean.shape == (8,)
                and np.allclose(
                    published_state_mean,
                    state.mean(axis=0, dtype=np.float64),
                    atol=2e-5,
                    rtol=0,
                )
            ),
            "action_mean_matches_parquet": bool(
                published_action_mean.shape == (7,)
                and np.allclose(published_action_mean, action.mean(axis=0), atol=2e-5, rtol=0)
            ),
            "published_action_mean": published_action_mean.tolist(),
            "parquet_action_mean": action.mean(axis=0, dtype=np.float64).tolist(),
            "warning": (
                "The published 'actions' stats describe a different representation when "
                "action_mean_matches_parquet is false; do not normalize the parquet action "
                "column with them."
            ),
        }
    else:
        published_norm_stats = {"skipped": "norm_stats.json is absent"}

    source_trajectory_summary, source_manifest = _source_trajectory_audit(
        state=state,
        action=action,
        episodes=episodes,
        offsets=expected_offsets,
        task_text_to_index=task_text_to_index,
        suite_by_task=suite_by_task,
        validation_group_proportion=args.validation_group_proportion,
        split_seed=args.split_seed,
    )

    report: dict[str, Any] = {
        "dataset_root": str(root),
        "codebase_version": info.get("codebase_version"),
        "metadata": {
            "episodes": len(episodes),
            "frames": int(table.num_rows),
            "tasks": len(tasks),
            "fps": int(info["fps"]),
            "episode_length": {
                "min": int(expected_lengths.min()),
                "median": float(np.median(expected_lengths)),
                "mean": float(expected_lengths.mean()),
                "q95": float(np.quantile(expected_lengths, 0.95)),
                "max": int(expected_lengths.max()),
            },
            "task_distribution": task_distribution,
        },
        "schema_and_alignment": {
            "dataset_was_global_index_sorted": already_sorted,
            "global_index_mismatches": global_index_mismatches,
            "episode_index_mismatches": int(
                np.count_nonzero(episode_index != expected_episode_rows)
            ),
            "frame_index_mismatches": int(np.count_nonzero(frame_index != expected_frame_rows)),
            "task_index_mismatches": int(np.count_nonzero(task_index != expected_task_rows)),
            "timestamp_max_abs_error_seconds": float(timestamp_error.max()),
            "nonfinite_state_values": int(np.size(state) - np.count_nonzero(np.isfinite(state))),
            "nonfinite_action_values": int(np.size(action) - np.count_nonzero(np.isfinite(action))),
        },
        "state": {
            "min": state.min(axis=0).astype(float).tolist(),
            "max": state.max(axis=0).astype(float).tolist(),
            "mean": state.mean(axis=0, dtype=np.float64).tolist(),
            "quantiles": _quantiles(state),
        },
        "action": {
            "min": action.min(axis=0).astype(float).tolist(),
            "max": action.max(axis=0).astype(float).tolist(),
            "mean": action.mean(axis=0, dtype=np.float64).tolist(),
            "quantiles": _quantiles(action),
            "pose_noop": noop,
            "trailing_pose_noop": {
                key: {
                    "episodes_with_any": int(np.count_nonzero(values)),
                    "mean_frames": float(np.mean(values)),
                    "median_frames": float(np.median(values)),
                    "max_frames": int(np.max(values)),
                }
                for key, values in trailing_noop.items()
            },
            "gripper_unique": [
                {"value": float(value), "count": int(count)}
                for value, count in zip(gripper_values, gripper_counts, strict=True)
            ],
        },
        "osc_response": {
            "transitions": int(transition_indices.size),
            "command_vs_next_position_response_corr": _safe_corrcoef(
                command_position, response_position
            ),
            "next_position_response_normalized": {
                "mean": response_position.mean(axis=0, dtype=np.float64).tolist(),
                "mean_abs": np.abs(response_position).mean(axis=0, dtype=np.float64).tolist(),
            },
            "rotation_sample": _rotation_response_audit(
                state, action, transition_indices, args.max_rotation_samples
            ),
            "position_row_alignment": _position_alignment_audit(
                state=state,
                action=action,
                episode_index=episode_index,
            ),
        },
        "gripper": {
            "open_min": float(gripper_open.min()),
            "open_max": float(gripper_open.max()),
            "open_mean": float(gripper_open.mean()),
            "action_mapping": gripper_mapping,
        },
        "video_sample": (
            _sample_video_audit(
                root=root,
                info=info,
                episodes=episodes,
                task_text_to_index=task_text_to_index,
            )
            if not args.skip_video_probe
            else {"skipped": "--skip-video-probe"}
        ),
        "release_metadata_limitations": {
            "episode_fields": sorted(episodes[0]),
            "has_suite_field": any("suite" in item for item in episodes),
            "has_perturbation_field": any(
                any("perturb" in key.lower() or "category" in key.lower() for key in item)
                for item in episodes
            ),
            "unique_instruction_count": len(task_text_to_index),
        },
        "source_trajectory_replays": source_trajectory_summary,
        "published_norm_stats": published_norm_stats,
        "_source_manifest": source_manifest,
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/libero_plus/libero_plus_lerobot"),
    )
    parser.add_argument(
        "--original-data-root",
        type=Path,
        default=Path("data/libero_mujoco3.3.2"),
        help="Used only to recover the canonical 40-task suite mapping.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/libero_plus/libero_plus_lerobot_audit.json"),
    )
    parser.add_argument(
        "--source-manifest-output",
        type=Path,
        default=Path("data/libero_plus/libero_plus_lerobot_source_manifest.jsonl"),
    )
    parser.add_argument("--validation-group-proportion", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-rotation-samples", type=int, default=200_000)
    parser.add_argument("--skip-video-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_rotation_samples < 1:
        raise ValueError("--max-rotation-samples must be positive")
    if not 0.0 < args.validation_group_proportion < 1.0:
        raise ValueError("--validation-group-proportion must be between 0 and 1")
    report = audit(args)
    source_manifest = report.pop("_source_manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.source_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with args.source_manifest_output.open("w", encoding="utf-8") as handle:
        for item in source_manifest:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(f"Audit saved to {args.output}")
    print(f"Source manifest saved to {args.source_manifest_output}")
    print(json.dumps({
        "metadata": report["metadata"] | {"task_distribution": "omitted"},
        "schema_and_alignment": report["schema_and_alignment"],
        "action_pose_noop": report["action"]["pose_noop"],
        "gripper_action_mapping": report["gripper"]["action_mapping"],
        "video_failures": len(report["video_sample"].get("failures", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
