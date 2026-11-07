#!/usr/bin/env python3
"""Prepare numerical action statistics for the original 5B baseline, CPU only."""
import argparse
import json
from pathlib import Path

from fastwam.realworld.baseline_data import REPRESENTATION, ActionCodec
from fastwam.realworld.common import POSE_CONVENTION, PREPROCESSING, sha256_file, write_json_checked


def prepare(source, output):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "split.json").read_text())
    stats = json.loads((source / "rothko_stats.json").read_text())
    if stats["split_manifest_sha256"] != sha256_file(source / "split.json"):
        raise ValueError("Source normalization / split mismatch")
    cfg = stats["realworld_codec_config"]
    result = dict(version=1, representation=REPRESENTATION, statistics_split="train",
        split_manifest_sha256=sha256_file(source / "split.json"),
        source_statistics_sha256=sha256_file(source / "rothko_stats.json"),
        absolute_position_min=cfg["absolute_position_min"], absolute_position_max=cfg["absolute_position_max"],
        relative_position_bounds=stats["translation_abs_bounds_xyz_m"],
        rotation_scale_rad=3.141592653589793, quaternion_order="wxyz", action_dim=7, proprio_dim=8,
        action_horizon=16, video_frames=17, action_dt=.1, camera_keys=manifest["camera_keys"],
        pose_convention=POSE_CONVENTION, preprocessing=PREPROCESSING,
        gripper_open_threshold_m=manifest["gripper_open_threshold_m"],
        action_fields=["local_x", "local_y", "local_z", "rotvec_x", "rotvec_y", "rotvec_z", "gripper_open"],
        proprio_fields=["base_x", "base_y", "base_z", "qw", "qx", "qy", "qz", "gripper_open"],
        gripper_normalized_range=[-1, 1],
        target="future measured pose relative to current EE; not recorded teleoperation command")
    write_json_checked(output / "action_stats.json", result)
    ActionCodec(output / "action_stats.json", source / "split.json")
    print(f"Prepared original FastWAM action statistics: {output / 'action_stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_q9995_v1")
    parser.add_argument("--output", default="data/realworld_artifacts/original_fastwam_wan22_5b_left_wrist_val2_g08_q9995_v1")
    args = parser.parse_args()
    prepare(args.source, args.output)
