"""Task-only sampling while preserving full-cache indices and normalization."""
import json
from pathlib import Path

from torch.utils.data import Dataset
from .dataset import RealWorldVideoDataset


class TaskSubsetVideoDataset(Dataset):
    def __init__(self, task_name, subset_manifest, **kwargs):
        self.base = RealWorldVideoDataset(**kwargs)
        self.task_name = task_name
        self.split = self.base.split
        payload = json.loads(Path(subset_manifest).read_text())
        if (payload['task'] != task_name
                or payload['source_split_sha256'] != self.base.episode_split_metadata['sha256']
                or payload['norm_stats_fingerprint'] != self.base.codec.norm_stats.fingerprint()):
            raise ValueError('Task subset split / normalization identity mismatch')
        self.records, expected = [], []
        for ep, record in enumerate(self.base.records):
            if record['task'] == task_name:
                self.records.append(record)
                expected.extend(range(self.base.offsets[ep], self.base.offsets[ep + 1]))
        if not expected or payload['indices'][self.split] != expected:
            raise ValueError('Task subset cache index mapping mismatch')
        self.indices = expected
        for name in ('codec', 'raymap_codec', 'num_frames', 'camera_keys', 'latent_cache_metadata'):
            setattr(self, name, getattr(self.base, name))
        self.episode_split_metadata = dict(self.base.episode_split_metadata, task=task_name)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Base __getitem__ uses the original full-dataset latent cache index.
        return self.base[self.indices[index]]
