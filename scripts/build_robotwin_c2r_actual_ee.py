"""Build a standalone clean50 LeRobot dataset with official measured EE poses.

Original parquet and videos are read-only. Episodes/global indices/language
indices are compacted, with provenance recorded. Videos are symlinked.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
from pathlib import Path
import runpy

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASK_MAPPING = {}


def initialize(mapping):
    global TASK_MAPPING
    TASK_MAPPING = mapping
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(), "std": values.std(0).tolist(), "count": [len(values)]}


def build_episode(job):
    source, raw, output, old_id, new_id, offset, old_stats, task = job
    table = pq.read_table(source, use_threads=False)
    state = np.asarray(table["observation.state"].to_pylist())
    action = np.asarray(table["action"].to_pylist())
    with h5py.File(raw) as f:
        q = f["joint_action/vector"][:]
        ee = np.concatenate([f["endpose/left_endpose"][:], f["endpose/right_endpose"][:]], axis=1)
        gripper = np.stack([f["endpose/left_gripper"][:], f["endpose/right_gripper"][:]], axis=1)
    n = len(state)
    if len(q) != n+1 or ee.shape != (n+1, 14) or gripper.shape != (n+1, 2):
        raise ValueError(f"Length/shape mismatch: {raw}")
    errors = [np.abs(state-q[:-1]).max(), np.abs(action-q[1:]).max(),
              np.abs(state[:, [6, 13]]-gripper[:-1]).max(),
              np.abs(action[:, [6, 13]]-gripper[1:]).max()]
    if max(errors) > 1e-6 or not np.isfinite(ee).all():
        raise ValueError(f"Source alignment/pose validation failed: {raw}")
    for k in (3, 10):
        if not np.allclose(np.linalg.norm(ee[:, k:k+4], axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError(f"Invalid quaternion: {raw}")
    if table["frame_index"].to_pylist() != list(range(n)) or set(table["episode_index"].to_pylist()) != {old_id}:
        raise ValueError(f"Invalid source indices: {source}")
    # No quaternion conversion, FK, or sign canonicalization: copy measured wxyz.
    ee = ee.astype(np.float32)
    changes = {
        "episode_index": np.full(n, new_id, dtype=np.int64),
        "index": np.arange(offset, offset+n, dtype=np.int64),
        "task_index": np.array([TASK_MAPPING[x] for x in table["task_index"].to_pylist()], dtype=np.int64),
        "observation.state.ee_pose_wxyz": ee[:-1],
        "action.ee_pose_wxyz": ee[1:],
    }
    source_table = table
    for key, values in changes.items():
        array = pa.array(values.tolist(), type=pa.list_(pa.float32()) if values.ndim == 2 else pa.int64())
        if key in table.column_names:
            table = table.set_column(table.column_names.index(key), key, array)
        else:
            table = table.append_column(key, array)
    table = table.replace_schema_metadata(None)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="snappy")
    checked = pq.read_table(output, use_threads=False)
    if not table.equals(checked):
        raise ValueError(f"Output readback differs: {output}")
    unchanged = [k for k in source_table.column_names if k not in changes]
    if not source_table.select(unchanged).equals(checked.select(unchanged)):
        raise ValueError(f"Source values changed: {output}")
    new_stats = copy.deepcopy(old_stats)
    for key, values in changes.items():
        new_stats[key] = stats(values)
    return {"episode_index": new_id, "stats": new_stats}, {
        "task_name": task, "episode_index": new_id, "source_episode_index": old_id,
        "official_hdf5": str(raw), "source_parquet": str(source), "rows": n,
        "alignment_max_abs_errors": [float(x) for x in errors],
        "current_ee": "official endpose[:-1]", "future_ee": "official endpose[1:]"}


def jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/robotwin2.0/robotwin2.0"))
    p.add_argument("--official", type=Path, default=Path("data/robotwin2.0_official_aloha_clean50"))
    p.add_argument("--output", type=Path, default=Path("data/robotwin2.0_c2r_clean50"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    source, official, output = args.source.resolve(), args.official.resolve(), args.output.resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError("Output or .building already exists; inspect it before retrying")
    names = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                "src/fastwam/datasets/lerobot/robotwin_tasks.py"))["ROBOTWIN_TASK_NAMES"]
    info = json.loads((source / "meta/info.json").read_text())
    if info["total_episodes"] != 27500:
        raise ValueError("Expected the released 27,500-episode source")
    with (source / "meta/episodes.jsonl").open() as f:
        episodes = {r["episode_index"]: r for r in map(json.loads, f)}
    selected = [i*550+j for i in range(50) for j in range(50)]
    wanted_language = {t for ep in selected for t in episodes[ep]["tasks"]}
    task_mapping, tasks = {}, []
    with (source / "meta/tasks.jsonl").open() as f:
        for r in map(json.loads, f):
            if r["task"] in wanted_language:
                task_mapping[r["task_index"]] = len(tasks)
                tasks.append({"task_index": len(tasks), "task": r["task"]})
    if {r["task"] for r in tasks} != wanted_language:
        raise ValueError("Missing source language entries")
    with (source / "meta/episodes_stats.jsonl").open() as f:
        selected_set = set(selected)
        episode_stats = {r["episode_index"]: r["stats"] for r in map(json.loads, f)
                         if r["episode_index"] in selected_set}
    (staging / "meta").mkdir(parents=True)
    camera_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    jobs, new_episodes = [], []
    offset = 0
    for i, task in enumerate(names):
        files = sorted((official / "extracted" / task / "aloha-agilex_clean_50/data").glob("episode*.hdf5"))
        if len(files) != 50:
            raise ValueError(f"Expected 50 HDF5s for {task}")
        for j, raw in enumerate(files):
            old_id, new_id = i*550+j, i*50+j
            old_kwargs = {"episode_chunk": old_id//info["chunks_size"], "episode_index": old_id}
            new_kwargs = {"episode_chunk": new_id//info["chunks_size"], "episode_index": new_id}
            jobs.append((source / info["data_path"].format(**old_kwargs), raw,
                         staging / info["data_path"].format(**new_kwargs), old_id, new_id,
                         offset, episode_stats[old_id], task))
            e = copy.deepcopy(episodes[old_id]);e["episode_index"] = new_id
            new_episodes.append(e);offset += e["length"]
            for key in camera_keys:
                target = source / info["video_path"].format(**old_kwargs, video_key=key)
                if not target.is_file():
                    raise FileNotFoundError(target)
                link = staging / info["video_path"].format(**new_kwargs, video_key=key)
                link.parent.mkdir(parents=True, exist_ok=True);link.symlink_to(target)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(task_mapping,)) as pool:
        for result in pool.map(build_episode, jobs):
            results.append(result)
            if len(results) % 100 == 0:
                print(f"Built and verified {len(results)}/2500", flush=True)
    if sum(x[1]["rows"] for x in results) != offset:
        raise ValueError("Episode metadata lengths do not match parquet")
    info.update(total_episodes=2500, total_frames=offset, total_tasks=len(tasks),
                total_videos=2500*len(camera_keys), total_chunks=math.ceil(2500/info["chunks_size"]),
                splits={"train": "0:2500"})
    pose_names = [f"{arm}_{v}" for arm in ("left", "right") for v in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    for key in ("observation.state.ee_pose_wxyz", "action.ee_pose_wxyz"):
        info["features"][key] = {"dtype": "float32", "shape": [14], "names": [pose_names]}
    (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    jsonl(staging / "meta/tasks.jsonl", tasks)
    jsonl(staging / "meta/episodes.jsonl", new_episodes)
    jsonl(staging / "meta/episodes_stats.jsonl", [x[0] for x in results])
    manifest = {"source_root": str(source), "official_root": str(official),
                "official_download_manifest": json.loads((official / "manifests/clean50_download_manifest.json").read_text()),
                "ee_source": "official_measured_endpose_wxyz", "variant": "clean", "episodes": [x[1] for x in results],
                "language_index_mapping": task_mapping, "total_frames": offset,
                "video_storage": "absolute_symlinks_to_source", "quaternion_transform": "none; cast float32 only"}
    (staging / "meta/c2r_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(output)
    print(f"Complete: {output}; 2500 episodes, {offset} rows, {len(tasks)} language entries", flush=True)


if __name__ == "__main__":
    main()
