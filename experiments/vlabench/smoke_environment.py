"""Render a Track-1 scene and execute one hold-pose command, without a model."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import VLABench
    import VLABench.robots  # Register robot classes before load_env.
    import VLABench.tasks  # Register task classes before load_env.
    from VLABench.envs import load_env

    root = Path(VLABench.__file__).parent
    track = json.loads((root / 'configs/evaluation/tracks/track_1_in_distribution.json').read_text())
    env = load_env('select_book', episode_config=track['select_book'][0],
                   random_init=False, eval=False, run_mode='eval')
    try:
        obs = env.get_observation(require_pcd=False)
        state = np.asarray(obs['ee_state'])
        assert state.shape == (8,) and np.isfinite(state).all()
        for camera in (2, 3):
            Image.fromarray(np.asarray(obs['rgb'][camera], dtype=np.uint8)).save(
                args.output_dir / f'camera{camera}.png')
        _, qpos = env.robot.get_qpos_from_ee_pos(
            physics=env.physics, pos=state[:3], quat=state[3:7])
        fingers = np.full(2, .04 if 1 - state[7] >= .5 else 0.)
        action = np.concatenate([qpos, fingers])
        assert action.shape == (9,) and np.isfinite(action).all()
        env.step(action)
        after = np.asarray(env.robot.get_ee_state(env.physics))
        assert np.isfinite(after).all()
        result = dict(task='select_book', track_episode=0,
                      language=env.task.get_instruction(),
                      before=state.tolist(), after=after.tolist(),
                      hold_action=action.tolist(),
                      robot_origin=np.asarray(env.get_robot_frame_position()).tolist())
        (args.output_dir / 'environment_smoke.json').write_text(json.dumps(result, indent=2))
        print('ENVIRONMENT_SMOKE_PASS', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
