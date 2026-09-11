"""A task subset view retaining the validated full C2R latent-cache indices."""
import json
from pathlib import Path

from .robot_video_dataset import RobotVideoDataset


def select_c2r_windows(episodes, task_names):
    names = set(task_names)
    if not names or names - {e['task_name'] for e in episodes}:
        raise ValueError('Unknown or empty C2R task selection')
    indices, counts, offset = [], {}, 0
    for episode_index, episode in enumerate(episodes):
        if episode['episode_index'] != episode_index:
            raise ValueError('Expected compact ordered C2R episode IDs')
        length = int(episode['rows'])
        if length <= 0:
            raise ValueError('C2R episode length must be positive')
        if episode['task_name'] in names:
            indices.extend(range(offset, offset + length))
            counts[episode['task_name']] = counts.get(episode['task_name'], 0) + length
        offset += length
    return indices, counts, offset


class RobotWinC2RTaskSubset(RobotVideoDataset):
    def __init__(self, c2r_task_names, **kwargs):
        if (kwargs.get('robotwin_task_names') is not None
                or kwargs.get('robotwin_data_variant', 'all') != 'all'
                or kwargs.get('episode_split_manifest') is not None
                or kwargs.get('val_set_proportion', 0) != 0
                or kwargs.get('global_sample_stride', 1) != 1
                or kwargs.get('skip_padding_as_possible', False)):
            raise ValueError('C2R subset view requires an unsplit full dataset, stride1, and unchanged padding')
        if len(kwargs['dataset_dirs']) != 1:
            raise ValueError('C2R subset view requires exactly one dataset root')
        super().__init__(**kwargs)
        manifest = json.loads((Path(self.dataset_dirs[0]) / 'meta/c2r_source_manifest.json').read_text())
        indices, self.c2r_task_window_counts, total = select_c2r_windows(manifest['episodes'], c2r_task_names)
        if total != len(self.lerobot_dataset):
            raise ValueError('C2R manifest and underlying full dataset length disagree')
        offset = 0
        for i, episode in enumerate(manifest['episodes']):
            end = offset + int(episode['rows'])
            if (int(self.lerobot_dataset.episode_data_index['from'][i]) != offset
                    or int(self.lerobot_dataset.episode_data_index['to'][i]) != end):
                raise ValueError(f'C2R episode/cache indexing mismatch at episode {i}')
            offset = end
        self.c2r_window_indices = indices
        # Keep the parent retry behavior, but retry at the subset level only.
        # A lower-level full-dataset retry could silently select another task.
        self.lerobot_dataset.sample_error_mode = 'raise'

    def __len__(self):
        return len(self.c2r_window_indices)

    def _get(self, idx):
        return super()._get(self.c2r_window_indices[idx])
