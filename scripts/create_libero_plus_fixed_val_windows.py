#!/usr/bin/env python
"""Create deterministic LIBERO-Plus validation windows from a grouped split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def _keyed_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def build_fixed_validation_manifest(
    split_manifest_path: Path,
    *,
    selection_seed: int,
    num_frames: int,
    visual_samples_per_suite: int,
) -> dict[str, Any]:
    split_bytes = split_manifest_path.read_bytes()
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    records = [
        json.loads(line)
        for line in split_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    val_records = [record for record in records if record["split"] == "val"]
    if not val_records:
        raise ValueError("Split manifest contains no validation episodes.")

    val_episode_start: dict[int, int] = {}
    offset = 0
    for record in sorted(val_records, key=lambda item: int(item["episode_index"])):
        episode_index = int(record["episode_index"])
        val_episode_start[episode_index] = offset
        offset += int(record["episode_length"])

    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in val_records:
        by_task[int(record["task_index"])].append(record)
    if len(by_task) != 40:
        raise ValueError(f"Expected 40 validation tasks, got {len(by_task)}.")

    samples = []
    for sample_offset, (task_index, candidates) in enumerate(sorted(by_task.items())):
        ordered = sorted(
            candidates,
            key=lambda item: _keyed_digest(
                selection_seed,
                f"episode:{int(item['episode_index'])}",
            ),
        )
        episode = ordered[0]
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["episode_length"])
        full_window_starts = episode_length - num_frames + 1
        if full_window_starts < 1:
            raise ValueError(
                f"Episode {episode_index} is shorter than num_frames={num_frames}."
            )
        frame_digest = int(
            _keyed_digest(selection_seed, f"frame:{episode_index}"), 16
        )
        frame_index = frame_digest % full_window_starts
        samples.append(
            {
                "sample_id": f"task{task_index:02d}_episode{episode_index:06d}_frame{frame_index:04d}",
                "val_dataset_index": val_episode_start[episode_index] + frame_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task_index": task_index,
                "task": str(episode["task"]),
                "suite": str(episode["suite"]),
                "source_trajectory_id": str(episode["source_trajectory_id"]),
                "diffusion_seed": selection_seed + sample_offset,
                "run_visual": False,
            }
        )

    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_suite[sample["suite"]].append(sample)
    if len(by_suite) != 4:
        raise ValueError(f"Expected four LIBERO suites, got {sorted(by_suite)}.")
    visual_ids = set()
    for suite, candidates in sorted(by_suite.items()):
        ordered = sorted(
            candidates,
            key=lambda item: _keyed_digest(
                selection_seed, f"visual:{suite}:{item['sample_id']}"
            ),
        )
        if len(ordered) < visual_samples_per_suite:
            raise ValueError(
                f"Suite {suite} has only {len(ordered)} candidate task samples."
            )
        visual_ids.update(
            item["sample_id"] for item in ordered[:visual_samples_per_suite]
        )
    for sample in samples:
        sample["run_visual"] = sample["sample_id"] in visual_ids

    return {
        "version": 1,
        "dataset_root": "data/libero_plus/libero_plus_lerobot",
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": split_sha256,
        "selection_algorithm": (
            "one full 17-frame window per task via seed-keyed episode/frame SHA256; "
            "one visual sample per suite"
        ),
        "selection_seed": selection_seed,
        "num_frames": num_frames,
        "num_loss_samples": len(samples),
        "num_visual_samples": len(visual_ids),
        "val_dataset_length": offset,
        "samples": samples,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(
            "data/libero_plus/libero_plus_lerobot_source_manifest_val02_seed42.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/libero_plus/libero_plus_fixed_val_windows_val02_seed42_n40.json"
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--visual-samples-per-suite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2.")
    if args.visual_samples_per_suite < 1:
        raise ValueError("--visual-samples-per-suite must be positive.")
    payload = build_fixed_validation_manifest(
        args.split_manifest,
        selection_seed=args.selection_seed,
        num_frames=args.num_frames,
        visual_samples_per_suite=args.visual_samples_per_suite,
    )
    _atomic_json(args.output, payload)
    print(f"Fixed validation manifest saved to {args.output}")
    print(
        f"loss_samples={payload['num_loss_samples']} "
        f"visual_samples={payload['num_visual_samples']} "
        f"val_dataset_length={payload['val_dataset_length']} "
        f"split_sha256={payload['split_manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
