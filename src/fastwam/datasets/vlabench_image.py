"""Opt-in reader for VLABench's image LeRobot export.

This is a window adapter, NOT a Rothko/model-ready training dataset. Original
pose/action fields remain unchanged; explicit gripper_open side channels use
the audited VLABench convention. No temporal shift or pose conversion occurs.
"""

from pathlib import Path
import bisect
import io
import json
from collections import OrderedDict

import torch
from datasets import load_dataset

from .lerobot.lerobot.datasets.utils import hf_transform_to_torch
from .lerobot.lerobot.lerobot_dataset import LeRobotDataset


FIELD_MAP = {
    "image": "observation.images.image",
    "second_image": "observation.images.second_image",
    "wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
    "actions": "action",
}


class _LazyEpisodeReader(torch.utils.data.Dataset):
    """Local Parquet windows without materializing a second Arrow dataset."""

    def __init__(self, root, info, episodes, cameras, frames, cache_size=2):
        self.root, self.info = root, info
        records = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
        lengths = {r["episode_index"]: r["length"] for r in records}
        self.episodes = list(range(info["total_episodes"])) if episodes is None else list(episodes)
        self.ends = []
        total = 0
        for episode in self.episodes:
            total += lengths[episode]
            self.ends.append(total)
        self.lengths = lengths
        self.tasks = {r["task_index"]: r["task"] for r in
                      map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
        self.cameras, self.frames = cameras, frames
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("episode_cache_size must be >= 1")
        self.cache = OrderedDict()
        self.episode_data_index = {
            "from": torch.tensor([0] + self.ends[:-1]), "to": torch.tensor(self.ends)
        }

    def __len__(self):
        return self.ends[-1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["cache"] = OrderedDict()
        return state

    def __getitem__(self, index):
        import numpy as np
        import pyarrow.parquet as pq
        from PIL import Image
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect.bisect_right(self.ends, index)
        episode = self.episodes[position]
        start = index - (self.ends[position - 1] if position else 0)
        if episode not in self.cache:
            path = self.root / self.info["data_path"].format(
                episode_chunk=episode // self.info["chunks_size"], episode_index=episode)
            columns = [*self.cameras, "state", "actions", "task_index", "episode_index",
                       "frame_index", "index", "timestamp"]
            table = pq.read_table(path, columns=columns)
            if len(table) != self.lengths[episode]:
                raise ValueError(f"Episode length mismatch: {path}")
            self.cache[episode] = table
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        table = self.cache[episode]
        item = {k: torch.tensor(table[k][start].as_py()) for k in
                ("episode_index", "frame_index", "index", "timestamp")}
        item["task"] = self.tasks[table["task_index"][start].as_py()]
        for key in (*self.cameras, "state", "actions"):
            count = self.frames - (key == "actions")
            offsets = list(range(start, start + count))
            indices = [min(i, len(table) - 1) for i in offsets]
            values = {}
            for i in set(indices):
                value = table[key][i].as_py()
                if key in self.cameras:
                    if value.get("bytes") is None:
                        raise ValueError("Expected embedded image bytes in VLABench export")
                    with Image.open(io.BytesIO(value["bytes"])) as image:
                        value = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255
                else:
                    value = torch.tensor(value, dtype=torch.float32)
                values[i] = value
            item[key] = torch.stack([values[i] for i in indices])
            item[key + "_is_pad"] = torch.tensor([i >= len(table) for i in offsets])
        return item


def map_vlabench_fields(sample):
    """Rename keys (including padding masks), without touching tensor values."""
    mapped = {}
    for key, value in sample.items():
        suffix = "_is_pad" if key.endswith("_is_pad") else ""
        source = key[:-len(suffix)] if suffix else key
        target = FIELD_MAP.get(source, source) + suffix
        if target in mapped:
            raise ValueError(f"Duplicate mapped VLABench field: {target}")
        mapped[target] = value
    return mapped


def vlabench_gripper_open_fields(state, action):
    """Return [..., 1] opening indicators without modifying raw 7D fields.

    VLABench state[..., 6] is the observed below-threshold finger indicator
    (despite get_ee_open_state's name). Action[..., 6] is already an opening
    command. Neither field is a continuous measurement of finger aperture.
    State and action may have different horizons (17 and 16 respectively).
    """
    for name, value in (("state", state), ("action", action)):
        if value.ndim < 1 or value.shape[-1] != 7:
            raise ValueError(f"VLABench {name} must have last dimension 7")
        gripper = value[..., 6:7]
        if not torch.isfinite(gripper).all() or ((gripper < 0) | (gripper > 1)).any():
            raise ValueError(f"VLABench raw {name} gripper must be in [0, 1]")
    return 1 - state[..., 6:7], action[..., 6:7].clone()


class _LocalImageDataset(LeRobotDataset):
    """Keep optional Arrow cache placement local to this reader instance."""

    def __init__(self, *args, arrow_cache_dir=None, **kwargs):
        self.arrow_cache_dir = arrow_cache_dir
        super().__init__(*args, **kwargs)

    def load_hf_dataset(self):
        if self.episodes is None:
            files = {"data_dir": str(self.root / "data")}
        else:
            files = {"data_files": [
                str(self.root / self.meta.get_data_file_path(ep))
                for ep in self.episodes
            ]}
        dataset = load_dataset(
            "parquet", split="train", cache_dir=self.arrow_cache_dir, **files
        )
        dataset.set_transform(hf_transform_to_torch)
        return dataset

    def download_episodes(self, *args, **kwargs):
        raise FileNotFoundError("VLABench image adapter requires local episode files")


class VLABenchImageWindowDataset(torch.utils.data.Dataset):
    """Return FastWAM-style raw dictionaries from explicitly named cameras.

    Windows are RGB/state[t:t+17] and actions[t:t+16] by default, with the
    existing LeRobot end-of-episode replication and padding masks. These index
    ranges are a loading convention, not a claim about action/observation
    physical alignment. Camera names intentionally retain the export's names.

    No split, normalization, pose conversion, language embedding or Rothko
    encoding is performed. In particular, missing dataset stats stay None.
    raw_state/raw_action['gripper_open'] expose the audited opening convention;
    their ['default'] entries retain the original export values.
    """

    def __init__(
        self, dataset_dir, episode_indices=None, num_frames=17,
        camera_keys=("image", "wrist_image"), arrow_cache_dir=None,
        lazy=False, episode_cache_size=2, return_images=True,
    ):
        root = Path(dataset_dir)
        for filename in ("info.json", "episodes.jsonl", "tasks.jsonl"):
            if not (root / "meta" / filename).is_file():
                raise FileNotFoundError(root / "meta" / filename)
        self.num_frames = int(num_frames)
        if self.num_frames < 2:
            raise ValueError("num_frames must be >= 2")
        self.camera_keys = tuple(camera_keys)
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be nonempty and unique")
        if not set(self.camera_keys) <= {"image", "second_image", "wrist_image"}:
            raise ValueError("Unknown VLABench camera field")
        # Read FPS without instantiating or mutating shared metadata classes.
        import json
        info = json.loads((root / "meta/info.json").read_text())
        fps = info["fps"]
        if episode_indices is not None:
            episode_indices = list(episode_indices)
            if (not episode_indices or len(set(episode_indices)) != len(episode_indices)
                    or any(i < 0 or i >= info["total_episodes"] for i in episode_indices)):
                raise ValueError("episode_indices must be unique, nonempty and in range")
        if lazy:
            if not return_images:
                self.camera_keys = ()
            self.reader = _LazyEpisodeReader(root, info, episode_indices, self.camera_keys,
                                             self.num_frames, episode_cache_size)
            self.episode_data_index = self.reader.episode_data_index
            return
        self.reader = _LocalImageDataset(
            repo_id="VLABench/vlabench_primitive_ft_lerobot", root=root,
            episodes=episode_indices, download_videos=False, video_backend="pyav",
            arrow_cache_dir=arrow_cache_dir,
            delta_timestamps={
                **{k: [i / fps for i in range(self.num_frames)]
                   for k in (*self.camera_keys, "state")},
                "actions": [i / fps for i in range(self.num_frames - 1)],
            },
        )
        self.episode_data_index = self.reader.episode_data_index

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        item = map_vlabench_fields(self.reader[idx])
        state, action = item["observation.state"], item["action"]
        state_open, action_open = vlabench_gripper_open_fields(state, action)
        return {
            "idx": idx,
            "task": item["task"],
            "episode_index": item["episode_index"],
            "frame_index": item["frame_index"],
            "index": item["index"],
            "timestamp": item["timestamp"],
            # Match BaseLerobotDataset's uint8 CHW image contract exactly.
            "images": {k: (item[f"observation.images.{k}"] * 255).to(torch.uint8)
                       for k in self.camera_keys},
            "state": {"default": state},
            "action": {"default": action},
            "raw_state": {"default": state, "gripper_open": state_open},
            "raw_action": {"default": action, "gripper_open": action_open},
            "state_is_pad": item["observation.state_is_pad"],
            "action_is_pad": item["action_is_pad"],
            "image_is_pad": (item[f"observation.images.{self.camera_keys[0]}_is_pad"]
                             if self.camera_keys else item["observation.state_is_pad"]),
        }
