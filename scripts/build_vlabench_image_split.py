"""Freeze a task-stratified 1% episode split; keep identical actions together.

The export has no original source-trajectory / scene ID. Exact action hashing
is a duplicate guard, NOT a proof of source-group or scene-level independence.
Only scalar Parquet columns are scanned; images and original files are unchanged.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import pyarrow.parquet as pq


PATTERNS = {
    "add_condiment": r"Add .+ to the dish",
    "insert_flower": r"Insert .+ into the vase_seen\.",
    "select_book": r"Please take the book .+",
    "select_chemistry_tube": r"Take out the .+ solution",
    "select_drink": r"Take out the .+ from the fridge_open",
    "select_fruit": r"Put the .+ into the plate_seen",
    "select_mahjong": r"Pick up the mahjong of .+",
    "select_painting": r"Please select the painting of style .+\.",
    "select_poker": r"Please pick the poker .+",
    "select_toy": r"Put the .+ into the giftbox_seen",
}


def classify_language(language):
    matches = [task for task, pattern in PATTERNS.items() if re.fullmatch(pattern, language)]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous or unrecognized primitive instruction: {language!r}")
    return matches[0]


def array_hash(array):
    a = np.ascontiguousarray(array, dtype="<f4")
    return hashlib.sha256(str(a.shape).encode() + b"\0" + a.tobytes()).hexdigest()


def select_validation_groups(groups, target, seed):
    """Select whole groups totaling target episodes; never split duplicates."""
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    choices = {0: []}
    for key in keys:
        size = len(groups[key])
        for total in sorted(list(choices), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in choices:
                choices[new_total] = choices[total] + [key]
        if target in choices:
            return choices[target]
    raise ValueError(f"Cannot select exactly {target} episodes without splitting duplicate groups")


def build(root, seed=42):
    meta_paths = {name: root / "meta" / name
                  for name in ("info.json", "episodes.jsonl", "tasks.jsonl")}
    meta_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in meta_paths.items()}
    info = json.loads(meta_paths["info.json"].read_text())
    episodes = [json.loads(line) for line in meta_paths["episodes.jsonl"].read_text().splitlines()]
    languages = {r["task_index"]: r["task"] for r in
                 map(json.loads, meta_paths["tasks.jsonl"].read_text().splitlines())}
    language_tasks = {i: classify_language(text) for i, text in languages.items()}
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata count mismatch")
    records = []
    action_groups = defaultdict(list)
    exact_groups = defaultdict(list)
    offset = 0
    for expected_id, episode in enumerate(episodes):
        eid = episode["episode_index"]
        if eid != expected_id:
            raise ValueError("Episode metadata must be ordered and contiguous")
        relative = info["data_path"].format(episode_chunk=eid // info["chunks_size"], episode_index=eid)
        data = pq.read_table(root / relative, columns=[
            "state", "actions", "episode_index", "frame_index", "index", "task_index",
        ]).to_pydict()
        length = episode["length"]
        if (data["episode_index"] != [eid] * length or
                data["frame_index"] != list(range(length)) or
                data["index"] != list(range(offset, offset + length))):
            raise ValueError(f"Episode {eid}: index/length mismatch")
        tasks = {language_tasks[i] for i in data["task_index"]}
        if len(tasks) != 1 or any(languages[i] not in episode["tasks"] for i in data["task_index"]):
            raise ValueError(f"Episode {eid}: task metadata mismatch")
        state, action = (np.asarray(data[k], dtype="<f4") for k in ("state", "actions"))
        if (state.shape != (length, 7) or action.shape != (length, 7) or
                not np.isfinite(state).all() or not np.isfinite(action).all()):
            raise ValueError(f"Episode {eid}: invalid numeric data")
        action_digest = array_hash(action)
        full_digest = array_hash(np.concatenate([state, action], axis=-1))
        record = {"episode_index": eid, "base_task": tasks.pop(), "length": length,
                  "task_indices": sorted(set(data["task_index"])),
                  "data_path": relative, "action_sha256": action_digest,
                  "state_action_sha256": full_digest}
        records.append(record)
        action_groups[action_digest].append(record)
        exact_groups[full_digest].append(eid)
        offset += length
        if (eid + 1) % 500 == 0:
            print(f"Scanned {eid + 1}/{len(episodes)} episodes", flush=True)
    if offset != info["total_frames"]:
        raise ValueError("Total frame count mismatch")
    by_task = defaultdict(dict)
    for digest, members in action_groups.items():
        tasks = {r["base_task"] for r in members}
        if len(tasks) != 1:
            raise ValueError("Identical action trajectory spans tasks; review before splitting")
        by_task[tasks.pop()][digest] = members
    if set(by_task) != set(PATTERNS):
        raise ValueError("Expected coverage of all 10 primitive tasks")
    val_ids = set()
    summary = {}
    for task, groups in sorted(by_task.items()):
        size = sum(map(len, groups.values()))
        target = max(1, round(size * 0.01))
        task_seed = int.from_bytes(hashlib.sha256(f"{seed}:{task}".encode()).digest()[:8], "big")
        chosen = select_validation_groups(groups, target, task_seed)
        val_ids.update(r["episode_index"] for digest in chosen for r in groups[digest])
        summary[task] = {"total_episodes": size, "train_episodes": size - target,
                         "val_episodes": target}
    for r in records:
        r["split"] = "val" if r["episode_index"] in val_ids else "train"
    for task, counts in summary.items():
        for split in ("train", "val"):
            counts[f"{split}_frame_starts_including_padding"] = sum(
                r["length"] for r in records if r["base_task"] == task and r["split"] == split)
    duplicates = [[r["episode_index"] for r in members]
                  for members in action_groups.values() if len(members) > 1]
    assert all(len({records[i]["split"] for i in ids}) == 1 for ids in duplicates)
    return {
        "schema_version": 1, "seed": seed, "requested_val_episode_fraction": 0.01,
        "dataset_format": info["codebase_version"], "metadata_sha256": meta_hashes,
        "grouping": "exact float32 action-trajectory SHA256; groups remain in one split",
        "source_group_leakage_guaranteed": False,
        "limitation": "No source trajectory / scene IDs in export; near duplicates and modified replays not excluded",
        "image_bytes_scanned": False,
        "task_assignment": "Full-match official primitive instruction templates; all task IDs checked",
        "task_patterns": PATTERNS, "summary": summary,
        "total_episodes": len(records), "total_frames": offset,
        "train_episode_indices": [r["episode_index"] for r in records if r["split"] == "train"],
        "val_episode_indices": sorted(val_ids),
        "identical_action_groups": sorted(duplicates),
        "identical_state_action_groups": sorted(ids for ids in exact_groups.values() if len(ids) > 1),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = build(args.dataset_root, args.seed)
    if args.output.exists():
        if json.loads(args.output.read_text()) != payload:
            raise RuntimeError("Frozen manifest differs; refusing to overwrite it")
        print("Existing manifest matches exactly")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    print(json.dumps({"manifest": str(args.output), "summary": payload["summary"],
                      "train": len(payload["train_episode_indices"]),
                      "val": len(payload["val_episode_indices"]),
                      "identical_action_groups": payload["identical_action_groups"]}, indent=2))


if __name__ == "__main__":
    main()
