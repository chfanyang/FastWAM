#!/usr/bin/env python3
"""CPU-only audit, episode split and train-only normalization; source data is read-only."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastwam.realworld.common import (TASKS, CAMERAS, PREPROCESSING, POSE_CONVENTION, GRIPPER_OPEN_THRESHOLD_M,
    load_episode, sha256_file, write_json_checked)
from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig, LiberoRothkoCodec
from fastwam.representations.rothko import (RothkoNormStats, _center_and_read_masks,
    _border_mask, quaternion_wxyz_to_matrix)


def prepare(root, output, val_episodes_per_task=2, seed=42, threshold=GRIPPER_OPEN_THRESHOLD_M,
            preserve_validation_from=None):
    root, output = Path(root), Path(output)
    previous_path = Path(preserve_validation_from) if preserve_validation_from else output / "split.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    if preserve_validation_from and previous is None:
        raise FileNotFoundError(previous_path)
    rng = np.random.default_rng(seed)
    records, cached, summary, no_grasp = [], {}, {}, []
    for task in TASKS:
        episodes = sorted((root / task).glob("episode_*"))
        if not 1 <= val_episodes_per_task < len(episodes):
            raise ValueError(f"Invalid validation episode count for {task}")
        language = json.loads((root / task / "language.json").read_text())
        nval = val_episodes_per_task
        validation = set(rng.choice(len(episodes), nval, replace=False).tolist())
        if previous is not None:
            paths = {r["path"] for r in previous["episodes"]
                     if r["task"] == task and r["split"] == "val"}
            available = {str(ep.relative_to(root)) for ep in episodes}
            if len(paths) != nval or not paths <= available:
                raise ValueError(f"Cannot preserve validation episodes for {task}")
            validation = {i for i, ep in enumerate(episodes) if str(ep.relative_to(root)) in paths}
        frames, windows, rejected = 0, Counter(), 0
        widths, intervals = [], []
        for i, ep in enumerate(episodes):
            poses, width, timestamps, _ = load_episode(ep)
            if np.all(width >= threshold):
                no_grasp.append(str(ep.relative_to(root)))
            length = len(poses)
            delta = np.diff(timestamps)
            # Reject timing gaps/jitter outside 50..150 ms; do not bridge dropped frames.
            bad = (delta < 0.05) | (delta > 0.15)
            prefix = np.r_[0, np.cumsum(bad)]
            starts = np.arange(max(0, length - 16))
            valid = starts[(prefix[starts + 16] - prefix[starts]) == 0]
            if not len(valid):
                raise ValueError(f"No full valid windows in {ep}")
            inventory = hashlib.sha256()
            for camera in ("right_camera", "left_camera", "wrist_camera"):
                paths = sorted((ep / "image" / camera).glob("*.png"))
                if [p.name for p in paths] != [f"{n:06d}.png" for n in range(length)]:
                    raise ValueError(f"Missing/misaligned images: {ep}/{camera}")
                with Image.open(paths[0]) as image:
                    if image.size != (224, 224):
                        raise ValueError(f"Images must already be resized: {paths[0]}")
                for p in paths:
                    stat = p.stat()
                    inventory.update(f"{camera}/{p.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
            instruction = language[ep.name]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Missing task instruction: {ep}")
            split = "val" if i in validation else "train"
            record = dict(path=str(ep.relative_to(root)), task=task, split=split,
                length=length, valid_starts=valid.tolist(), instruction=instruction,
                states_sha256=sha256_file(ep / "states.npz"),
                image_inventory_sha256=inventory.hexdigest())
            records.append(record)
            cached[record["path"]] = poses
            frames += length
            windows[split] += len(valid)
            rejected += len(starts) - len(valid)
            widths.append(width)
            intervals.extend(delta.tolist())
        summary[task] = dict(episodes=len(episodes), val_episodes=nval, frames=frames,
            windows=dict(windows), rejected_timing_windows=rejected,
            median_hz=1 / float(np.median(intervals)),
            width_quantiles_m=np.quantile(np.concatenate(widths), [0, .1, .5, .9, 1]).tolist())
        print(task, summary[task], flush=True)
    manifest = dict(version=1, seed=seed, val_episodes_per_task=val_episodes_per_task, horizon=16,
        action_dt=0.1, camera_keys=list(CAMERAS), preprocessing=PREPROCESSING,
        pose_convention=POSE_CONVENTION, gripper_open_threshold_m=threshold,
        gripper_labels="future_measured_width_threshold_not_recorded_commands",
        timestamp_interval_bounds_s=[0.05, 0.15], episodes=records)
    manifest_path = output / "split.json"
    write_json_checked(manifest_path, manifest)
    train = [r for r in records if r["split"] == "train"]
    positions = np.concatenate([cached[r["path"]][:, :3] for r in train])
    absolute_lo, absolute_hi = positions.min(0) - 0.01, positions.max(0) + 0.01
    relative_max = np.zeros(3)
    for r in train:
        pose = cached[r["path"]]
        starts = np.asarray(r["valid_starts"])
        rotations = quaternion_wxyz_to_matrix(torch.from_numpy(pose[starts, 3:])).numpy()
        target = pose[starts[:, None] + np.arange(1, 17)[None, :], :3]
        local = np.einsum("nji,ntj->nti", rotations, target - pose[starts, None, :3])
        relative_max = np.maximum(relative_max, np.abs(local).max((0, 1)))
    # Exact train-window maxima plus 5% margin; no training translation clipping.
    bounds = np.maximum(relative_max * 1.05, 0.005)
    cfg = LiberoRothkoCodecConfig(frame0_pose_mode="absolute",
        absolute_position_min=tuple(absolute_lo.tolist()),
        absolute_position_max=tuple(absolute_hi.tolist()))
    cfg_dict = json.loads(json.dumps(asdict(cfg)))
    lo = torch.full((1, 3, 224, 224), -1.)
    hi = torch.ones_like(lo)
    center, _, _ = _center_and_read_masks(224, 224, center_frac=.5,
        boundary_margin=0, outer_margin=0, device=lo.device)
    hi[..., center] = torch.tensor(bounds, dtype=torch.float32)[None, :, None]
    lo[..., center] = -hi[..., center]
    border = _border_mask(224, 224, 8, lo.device)
    lo[..., border], hi[..., border] = -1, 1
    metadata = dict(LiberoRothkoCodec(config=cfg).metadata(), stats_format_version=2,
        image_size=[224, 448], tile_size=[224, 224], action_horizon=16, pixel_frames=17,
        realworld_codec_config=cfg_dict, split_manifest_sha256=sha256_file(manifest_path),
        source_environment="realworld_yarp_fr3", statistics_split="train",
        normalization_method="all_valid_train_windows_symmetric_max_margin05",
        translation_abs_bounds_xyz_m=bounds.tolist(),
        gripper_open_threshold_m=threshold, pose_convention=POSE_CONVENTION)
    stats = RothkoNormStats(torch.cat((lo, lo), -1), torch.cat((hi, hi), -1), metadata)
    stats_path = output / "rothko_stats.pt"
    if stats_path.exists():
        old = RothkoNormStats.load(stats_path)
        if old.metadata != metadata or old.fingerprint() != stats.fingerprint():
            raise ValueError("Existing normalization differs; choose new output directory")
    else:
        tmp = stats_path.with_suffix(".partial")
        torch.save(dict(lo=stats.lo, hi=stats.hi, metadata=metadata), tmp)
        tmp.replace(stats_path)
    write_json_checked(output / "rothko_stats.json", metadata)
    val_positions = np.concatenate([cached[r["path"]][:, :3] for r in records if r["split"] == "val"])
    report = dict(tasks=summary, train_episodes=len(train), val_episodes=len(records)-len(train),
        gripper_open_threshold_m=threshold, episodes_without_closed_gripper_labels=no_grasp,
        train_windows=sum(len(r["valid_starts"]) for r in train),
        val_windows=sum(len(r["valid_starts"]) for r in records if r["split"] == "val"),
        absolute_position_min=absolute_lo.tolist(), absolute_position_max=absolute_hi.tolist(),
        relative_position_bounds=bounds.tolist(),
        val_states_outside_absolute_bounds=int(np.any((val_positions < absolute_lo) | (val_positions > absolute_hi), axis=1).sum()),
        notes=["Camera exposure timestamps were not saved; same recorder tick is only approximate alignment.",
               "Training target is future measured pose, not the teleoperation target."])
    write_json_checked(output / "audit.json", report)
    probes, seen, offset = [], set(), 0
    for episode_index, r in enumerate(records):
        if r["split"] != "val":
            continue
        if r["task"] not in seen:
            local = len(r["valid_starts"]) // 2
            probes.append(dict(sample_id=r["path"].replace("/", "_") + f"_{local}",
                val_dataset_index=offset + local, episode_index=episode_index,
                frame_index=r["valid_starts"][local], task_index=TASKS.index(r["task"]),
                diffusion_seed=seed + len(probes), run_visual=True))
            seen.add(r["task"])
        offset += len(r["valid_starts"])
    write_json_checked(output / "validation_windows.json", dict(version=1,
        split_manifest_sha256=sha256_file(manifest_path), samples=probes,
        val_dataset_length=offset, environment="realworld"))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/realworld_data")
    parser.add_argument("--output-dir", default="data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_v2")
    parser.add_argument("--preserve-validation-from", help="Keep validation episode identities from this split manifest")
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.data_root, args.output_dir, preserve_validation_from=args.preserve_validation_from)
