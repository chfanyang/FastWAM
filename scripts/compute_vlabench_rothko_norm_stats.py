"""Exact train-only Q99.95 stats over ALL VLABench frame starts, including tails.

No image/VAE/GPU work. Every window contributes 16 future targets; padding
repeats the last action. The relative RAY0 is identically zero and is not added
to the empirical future-displacement distribution, matching the LIBERO method.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch

from build_vlabench_image_split import array_hash
from compute_libero_rothko_norm_stats import _build_stats
from fastwam.representations.vlabench_rothko import (
    VLABenchRothkoCodec, VLABenchRothkoCodecConfig,
)


def relative_positions(state, action, horizon=16):
    """All starts: R(state[t])^T * (action[min(t+j,N-1)] - state[t])."""
    n = len(state)
    indices = np.minimum(np.arange(n)[:, None] + np.arange(horizon)[None], n - 1)
    rotation = Rotation.from_euler("xyz", state[:, 3:6]).as_matrix()
    delta = action[indices, :3].astype(np.float64) - state[:, None, :3].astype(np.float64)
    return np.einsum("wij,wtj->wti", rotation.transpose(0, 2, 1), delta)


def clipping_report(values, bounds):
    clipped = np.abs(values) > bounds
    return {
        "windows": len(values), "future_vectors": int(values.shape[0] * values.shape[1]),
        "axis_clipped_count": clipped.sum(axis=(0, 1)).tolist(),
        "axis_clipped_fraction": clipped.mean(axis=(0, 1)).tolist(),
        "any_axis_vector_clipped_fraction": float(clipped.any(axis=-1).mean()),
        "any_future_window_clipped_fraction": float(clipped.any(axis=(1, 2)).mean()),
        "max_abs_xyz_m": np.abs(values).max(axis=(0, 1)).tolist(),
        "max_excess_xyz_m": np.maximum(np.abs(values) - bounds, 0).max(axis=(0, 1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new output; existing stats will not be overwritten")
    manifest_bytes = args.split_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    for name, digest in manifest["metadata_sha256"].items():
        if hashlib.sha256((args.dataset_root / "meta" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Metadata differs from frozen split: {name}")
    records = manifest["records"]
    train_ids, val_ids = set(manifest["train_episode_indices"]), set(manifest["val_episode_indices"])
    if (train_ids & val_ids or train_ids | val_ids != {r["episode_index"] for r in records}):
        raise ValueError("Invalid train/val partition")
    horizon = 16
    arrays = {split: np.empty((sum(r["length"] for r in records if r["split"] == split), horizon, 3),
                              dtype=np.float64) for split in ("train", "val")}
    offsets = dict(train=0, val=0)
    intervals = {s: {} for s in arrays}
    pad_counts = {s: 0 for s in arrays}
    for count, record in enumerate(records, 1):
        eid, split, n = record["episode_index"], record["split"], record["length"]
        if (eid in train_ids) != (split == "train"):
            raise ValueError("Record partition mismatch")
        table = pq.read_table(args.dataset_root / record["data_path"], columns=["state", "actions"]).to_pydict()
        state, action = (np.asarray(table[k], dtype=np.float32) for k in ("state", "actions"))
        if (array_hash(action) != record["action_sha256"] or
                array_hash(np.concatenate([state, action], axis=-1)) != record["state_action_sha256"]):
            raise ValueError(f"Episode {eid} changed since split was frozen")
        rel = relative_positions(state, action, horizon)
        lo, hi = offsets[split], offsets[split] + n
        arrays[split][lo:hi] = rel
        intervals[split].setdefault(record["base_task"], []).append((lo, hi))
        pad_counts[split] += int((np.arange(n)[:, None] + np.arange(horizon)[None] >= n).sum())
        offsets[split] = hi
        if count % 500 == 0:
            print(f"Read {count}/{len(records)} episodes", flush=True)
    for split in arrays:
        assert offsets[split] == len(arrays[split])
    # Exact empirical quantile, no histogram cap or per-episode subsampling.
    bounds = np.quantile(np.abs(arrays["train"]).reshape(-1, 3), 0.9995, axis=0, method="linear")
    if not np.isfinite(bounds).all() or (bounds <= 0).any():
        raise ValueError("Degenerate translation bounds")
    cfg = VLABenchRothkoCodecConfig()
    lo, hi = _build_stats(bounds, cfg.center_frac, cfg.center_scale, cfg.dir_scale, cfg.outer_margin)
    # Measure clipping against the actual float32 tensor bounds, not rounded text.
    applied_bounds = hi[0, :, 112, 112].numpy().astype(np.float64) / cfg.center_scale
    reports = {}
    for split, values in arrays.items():
        reports[split] = clipping_report(values, applied_bounds)
        reports[split]["padded_future_vectors"] = pad_counts[split]
        reports[split]["per_task"] = {}
        for task, spans in intervals[split].items():
            subset = np.concatenate([values[a:b] for a, b in spans], axis=0)
            reports[split]["per_task"][task] = clipping_report(subset, applied_bounds)
    metadata = {
        **VLABenchRothkoCodec(config=cfg).metadata(),
        "stats_format_version": 2, "image_size": [224, 448], "tile_size": [224, 224],
        "action_horizon": horizon, "pixel_frames": 17,
        "encoding": "current_observed_ee_plus_future_absolute_target_ee",
        "input_pose_convention": "robot-translation-relative xyz + scipy xyz Euler radians; codec takes wxyz",
        "frame0_pose_mode": "relative",
        "gripper_conversion": "state_open=1-state[...,6]; action_open=actions[...,6]",
        "translation_quantile": 0.9995, "quantile_method": "numpy exact linear quantile of absolute components",
        "translation_abs_bounds_xyz_m": applied_bounds.tolist(),
        "exact_quantile_before_float32_cast_xyz_m": bounds.tolist(),
        "sampling": "all_frame_starts_edge_repeat_padding",
        "tail_padding": "target_index=min(start+offset,N-1), offset=0..15; padded targets INCLUDED",
        "ray0_in_quantile": False,
        "dataset_root": str(args.dataset_root.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "num_episodes": len(train_ids), "num_windows": len(arrays["train"]),
        "num_relative_vectors": len(arrays["train"]) * horizon,
        "fit_split": "train", "validation_used_for_fitting": False,
        "clipping_reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lo": lo, "hi": hi, "metadata": metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Require metadata/geometry/horizon validation before declaring success.
    VLABenchRothkoCodec(norm_stats=args.output, expected_action_horizon=horizon)
    print("Saved", args.output, "bounds (m)", applied_bounds.tolist(), flush=True)
    for split in reports:
        print(split, json.dumps({k: v for k, v in reports[split].items() if k != "per_task"}), flush=True)


if __name__ == "__main__":
    main()
