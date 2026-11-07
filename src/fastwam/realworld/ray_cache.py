"""Ray-only encoding inputs with bit-preserving RGB reuse from a validated cache."""
import bisect
import copy

import torch
from torch.utils.data import Dataset
from .common import gripper_open


def validate_rgb_reuse(source, target):
    old = copy.deepcopy(source.latent_cache_dataset_contract)
    new = copy.deepcopy(target.latent_cache_dataset_contract)
    old.pop('norm_stats_sha256')
    new.pop('norm_stats_sha256')
    if old != new or source.records != target.records:
        raise ValueError('RGB reuse requires identical sources, window order, RGB preprocessing and codec geometry')
    if source.latent_cache is None:
        raise ValueError('RGB source requires a complete validated latent cache')
    if not source.latent_cache_metadata.get('verification', {}).get('training_loss_bit_exact'):
        raise ValueError('Source cache has not passed online equivalence verification')


class RayOnlyDataset(Dataset):
    def __init__(self, target, source):
        validate_rgb_reuse(source, target)
        self.target, self.source = target, source

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect.bisect_right(self.target.offsets, index) - 1
        record = self.target.records[ep]
        start = record['valid_starts'][index - self.target.offsets[ep]]
        poses, width, _, _ = self.target._episode(ep)
        pose = torch.from_numpy(poses[start:start + 17].copy())
        grip = torch.from_numpy(gripper_open(width[start:start + 17], self.target.threshold)[:, None])
        rgb, _ = self.source.latent_cache[index]
        return dict(rgb_latents=rgb, raymap=self.target.codec.encode(pose, grip))
