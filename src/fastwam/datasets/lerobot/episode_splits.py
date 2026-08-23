from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_grouped_episode_split(
    manifest_path: str | Path,
    split: str,
) -> tuple[list[int], dict[str, Any]]:
    """Load and validate a source-trajectory-grouped episode split manifest."""
    if split not in {"train", "val"}:
        raise ValueError(f"`split` must be 'train' or 'val', got {split!r}.")
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Episode split manifest does not exist: {path}")

    records = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            missing = {
                "episode_index",
                "source_trajectory_id",
                "split",
                "task_index",
            } - set(record)
            if missing:
                raise ValueError(
                    f"Missing keys {sorted(missing)} at {path}:{line_number}"
                )
            if record["split"] not in {"train", "val"}:
                raise ValueError(
                    f"Invalid split {record['split']!r} at {path}:{line_number}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"Episode split manifest is empty: {path}")

    episode_indices = [int(record["episode_index"]) for record in records]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError(f"Duplicate episode_index entries in {path}")

    source_splits: dict[str, set[str]] = defaultdict(set)
    source_tasks: dict[str, set[int]] = defaultdict(set)
    for record in records:
        source_id = str(record["source_trajectory_id"])
        source_splits[source_id].add(str(record["split"]))
        source_tasks[source_id].add(int(record["task_index"]))
    leaked = [source_id for source_id, values in source_splits.items() if len(values) != 1]
    if leaked:
        raise ValueError(
            "Source trajectories cross train/val in manifest: "
            f"{leaked[:10]}"
        )
    cross_task = [source_id for source_id, values in source_tasks.items() if len(values) != 1]
    if cross_task:
        raise ValueError(
            f"Source trajectories cross tasks in manifest: {cross_task[:10]}"
        )

    selected = sorted(
        int(record["episode_index"])
        for record in records
        if record["split"] == split
    )
    if not selected:
        raise ValueError(f"Manifest {path} contains no episodes for split {split!r}")
    selected_sources = {
        str(record["source_trajectory_id"])
        for record in records
        if record["split"] == split
    }
    selected_tasks = {
        int(record["task_index"])
        for record in records
        if record["split"] == split
    }
    return selected, {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "split": split,
        "episodes": len(selected),
        "source_trajectories": len(selected_sources),
        "tasks": len(selected_tasks),
    }
