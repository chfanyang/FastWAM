"""Opt-in concatenation of independently validated VLABench task caches."""
import json
from pathlib import Path

from torch.utils.data import ConcatDataset

from .vlabench_video import VLABenchVideoDataset


class VLABenchMultitaskCachedDataset(ConcatDataset):
    def __init__(self, cache_index_path, **dataset_kwargs):
        if dataset_kwargs.get('split', 'train') != 'train':
            raise ValueError('Task-cache concatenation is for training only')
        if not dataset_kwargs.get('latent_cache_only', False):
            raise ValueError('Multitask cached training requires latent_cache_only=true')
        if dataset_kwargs.get('task_names') is not None:
            raise ValueError('Multitask cache index selects tasks; task_names must be null')
        index = json.loads(Path(cache_index_path).read_text())
        entries = index['entries']
        split = json.loads(Path(dataset_kwargs['split_manifest']).read_text())
        records = {r['episode_index']: r for r in split['records']}
        expected = {records[e]['base_task'] for e in split['train_episode_indices']}
        tasks = [e['task'] for e in entries]
        if len(tasks) != len(set(tasks)) or set(tasks) != expected:
            raise ValueError('Cache index must contain every training task exactly once')
        datasets = []
        common = None
        for entry in entries:
            kwargs = dict(dataset_kwargs, task_names=[entry['task']], cache_scope='task',
                          latent_cache_dir=entry['path'])
            dataset = VLABenchVideoDataset(**kwargs)
            if len(dataset) != entry['samples']:
                raise ValueError(f"Cache index sample count mismatch: {entry['task']}")
            meta = dataset.latent_cache_metadata
            # Each child checks its entire dataset contract; compare all shared
            # cache fields before exposing their common model-facing identity.
            identity = {k: v for k, v in meta.items() if k not in
                        {'dataset_contract', 'num_samples', 'shards'}}
            if common is not None and identity != common:
                raise ValueError(f"Incompatible task cache identity: {entry['task']}")
            common = identity
            datasets.append(dataset)
        super().__init__(datasets)
        if len(self) != index['total_samples']:
            raise ValueError('Total cache sample count mismatch')
        seen = [e for d in datasets for e in d.latent_cache_dataset_contract['episode_indices']]
        if len(seen) != len(set(seen)) or set(seen) != set(split['train_episode_indices']):
            raise ValueError('Training episodes duplicated or missing from task caches')
        self.raymap_codec = self.codec = datasets[0].codec
        self.num_frames = datasets[0].num_frames
        self.episode_split_metadata = datasets[0].episode_split_metadata
        self.latent_cache_only = True
        self.latent_cache_metadata = common
        self.task_names = tasks
