"""Read all observation EE positions; never modify source parquet or old stats."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, default=Path('data/libero_mujoco3.3.2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    suites = {}
    key = 'observation.state.ee_pose_wxyz'
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        files = sorted((args.dataset_root / f'{suite}_no_noops_lerobot' / 'data').rglob('*.parquet'))
        if not files:
            raise FileNotFoundError(suite)
        count = 0
        for path in files:
            poses = np.asarray(pq.read_table(path, columns=[key])[key].to_pylist(), dtype=np.float64)
            if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
                raise ValueError(f'Invalid EE poses: {path}')
            lo = np.minimum(lo, poses[:, :3].min(0))
            hi = np.maximum(hi, poses[:, :3].max(0))
            count += len(poses)
        suites[suite] = {'episodes': len(files), 'frames': count}
        print(suite, suites[suite], flush=True)
    payload = {
        'method': 'all_observation_frames_axis_min_max',
        'coordinate_frame': 'world', 'pose_column': key,
        'dataset_root': str(args.dataset_root.resolve()), 'suites': suites,
        'absolute_position_min': lo.tolist(), 'absolute_position_max': hi.tolist(),
        'rotation_bounds': [-1.0, 1.0],
        'notes': 'RAY0 only. All rows, no window subsampling, no quantile trimming. Future relative stats unchanged.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
