"""Opt-in numerical targets alongside the existing, validated RGB cache."""
import bisect
import json
from collections import OrderedDict
from pathlib import Path
import torch
import pyarrow.parquet as pq
from .robot_video_dataset import RobotVideoDataset


class LiberoLatentRepeatDataset(RobotVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.raymap_representation != 'libero_rothko' or self.num_frames != 17:
            raise ValueError('Only baseline LIBERO h16 chunk-relative targets are supported')
        if kwargs.get('val_set_proportion', 0) != 0:
            raise ValueError('This adapter requires the baseline all-episode ordering')
        self._repeat_episodes = []
        self._repeat_ends = []
        self._repeat_arrays = OrderedDict()
        offset = 0
        for root in self.dataset_dirs:
            root = Path(root)
            for ep in sorted((json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines()), key=lambda e:e['episode_index']):
                self._repeat_episodes.append((root, ep['episode_index'], offset, ep['length']))
                offset += ep['length']; self._repeat_ends.append(offset)
        if offset != len(self):
            raise ValueError('Episode ordering/count mismatch')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        index = int(sample.get('latent_cache_index', index))
        root, ep, offset, length = self._repeat_episodes[bisect.bisect_right(self._repeat_ends, index)]
        key = (str(root), ep)
        if key not in self._repeat_arrays:
            path = next(root.glob(f'data/chunk-*/episode_{ep:06d}.parquet'))
            cols = ['observation.state.ee_pose_wxyz','observation.state.gripper_open','action.osc_target_pose_wxyz','action']
            table = pq.read_table(path, columns=cols)
            self._repeat_arrays[key] = {k:torch.tensor(table[k].to_pylist(), dtype=torch.float32) for k in cols}
            if len(self._repeat_arrays) > 4:
                self._repeat_arrays.popitem(last=False)
        self._repeat_arrays.move_to_end(key)
        a = self._repeat_arrays[key]; start = index-offset
        future = torch.arange(start, start+16).clamp_max(length-1)
        pose = torch.cat((a['observation.state.ee_pose_wxyz'][start:start+1], a['action.osc_target_pose_wxyz'][future]))
        grip = torch.cat((a['observation.state.gripper_open'][start:start+1].reshape(1,1),a['action'][future,-1:])).clamp(0,1)
        sample.update(repeat_pose=pose,repeat_gripper=grip,current_endpose=pose[0],future_endpose=pose[1:],future_gripper=grip[1:])
        # These VAE action latents are intentionally never consumed.
        sample.pop('raymap_latents', None)
        return sample
