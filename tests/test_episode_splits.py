import json

import pytest

from fastwam.datasets.lerobot.episode_splits import load_grouped_episode_split


def _write_manifest(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_grouped_episode_split_keeps_source_groups_disjoint(tmp_path):
    path = tmp_path / "split.jsonl"
    _write_manifest(
        path,
        [
            {
                "episode_index": 3,
                "source_trajectory_id": "source-a",
                "split": "train",
                "task_index": 0,
            },
            {
                "episode_index": 1,
                "source_trajectory_id": "source-a",
                "split": "train",
                "task_index": 0,
            },
            {
                "episode_index": 2,
                "source_trajectory_id": "source-b",
                "split": "val",
                "task_index": 1,
            },
        ],
    )

    train, train_meta = load_grouped_episode_split(path, "train")
    val, val_meta = load_grouped_episode_split(path, "val")

    assert train == [1, 3]
    assert val == [2]
    assert train_meta["source_trajectories"] == 1
    assert val_meta["source_trajectories"] == 1
    assert train_meta["sha256"] == val_meta["sha256"]


def test_load_grouped_episode_split_rejects_source_leakage(tmp_path):
    path = tmp_path / "split.jsonl"
    _write_manifest(
        path,
        [
            {
                "episode_index": 0,
                "source_trajectory_id": "same-source",
                "split": "train",
                "task_index": 0,
            },
            {
                "episode_index": 1,
                "source_trajectory_id": "same-source",
                "split": "val",
                "task_index": 0,
            },
        ],
    )

    with pytest.raises(ValueError, match="cross train/val"):
        load_grouped_episode_split(path, "train")


def test_load_grouped_episode_split_rejects_duplicate_episode(tmp_path):
    path = tmp_path / "split.jsonl"
    record = {
        "episode_index": 0,
        "source_trajectory_id": "source-a",
        "split": "train",
        "task_index": 0,
    }
    _write_manifest(path, [record, record])

    with pytest.raises(ValueError, match="Duplicate episode_index"):
        load_grouped_episode_split(path, "train")
