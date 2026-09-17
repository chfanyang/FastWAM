"""Opt-in logging around the official loop; no action modification or IK retry."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def rotation_error_degrees(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.degrees((Rotation.from_quat(a[[1, 2, 3, 0]]).inv()
                             * Rotation.from_quat(b[[1, 2, 3, 0]])).magnitude()))


@contextmanager
def record_episode(agent, output_dir):
    # Patch only during this opt-in episode, restoring all references afterwards.
    import VLABench.evaluation.evaluator.base as official
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    frames = output / 'frames'
    frames.mkdir()
    load_env, predict, infer = official.load_env, agent.predict, agent.model.infer
    state = {'step': 0, 'replan': -1, 'pending': None}
    restorations = []
    log = (output / 'steps.jsonl').open('w', buffering=1)

    def traced_infer(*args, **kwargs):
        result = infer(*args, **kwargs)
        state['replan'] += 1
        state['chunk_start'] = state['step']
        data = dict(replan=state['replan'], env_step=state['step'],
                    current_endpose=kwargs['current_endpose'].detach().cpu().float().tolist(),
                    pose=result['pose'].float().tolist(), gripper=result['gripper'].float().tolist())
        (output / f"replan_{state['replan']:03d}.json").write_text(json.dumps(data, indent=2))
        return result

    def traced_predict(obs, **kwargs):
        before = np.asarray(obs['ee_state']).copy()
        result = predict(obs, **kwargs)
        pos, euler, fingers = result
        xyzw = Rotation.from_euler('xyz', euler).as_quat()
        quat = xyzw[[3, 0, 1, 2]]
        step = state['step']
        image_name = f'frames/step_{step:04d}.png'
        Image.fromarray(np.concatenate([obs['rgb'][2], obs['rgb'][3]], axis=1)).save(output / image_name)
        state['pending'] = dict(step=step, replan=state['replan'],
            action_index=step-state['chunk_start'], frame=image_name,
            language=str(obs['instruction']), before_ee=before.tolist(),
            robot_origin=np.asarray(obs['robot_frame']).tolist(),
            target_xyz=np.asarray(pos).tolist(), target_quat_wxyz=quat.tolist(),
            target_euler_xyz=np.asarray(euler).tolist(), finger_command=np.asarray(fingers).tolist(),
            requested_translation_m=float(np.linalg.norm(np.asarray(pos)-before[:3])),
            requested_rotation_deg=rotation_error_degrees(before[3:7], quat))
        return result

    def traced_load(*args, **kwargs):
        env = load_env(*args, **kwargs)
        ik, step_fn = env.robot.get_qpos_from_ee_pos, env.step
        restorations.extend([(env.robot, 'get_qpos_from_ee_pos', ik), (env, 'step', step_fn)])

        def traced_ik(*args, **kwargs):
            result = ik(*args, **kwargs)
            row = state['pending']
            if row is not None:
                row['ik_success'] = bool(result[0])
                row['ik_target_qpos'] = np.asarray(result[1]).tolist()
                row['before_qpos'] = np.asarray(env.robot.get_qpos(env.physics)).tolist()
                # FK on a separate physics copy; the rollout state is never touched.
                probe = env.physics.copy(share_model=True)
                try:
                    probe.data.qpos[:len(result[1])] = result[1]
                    probe.forward()
                    fk = np.asarray(env.robot.get_ee_state(probe))
                    row['ik_solution_ee'] = fk.tolist()
                    row['ik_position_error_m'] = float(np.linalg.norm(fk[:3]-row['target_xyz']))
                    row['ik_rotation_error_deg'] = rotation_error_degrees(fk[3:7], row['target_quat_wxyz'])
                finally:
                    probe.free()
            return result

        def traced_step(action=None):
            result = step_fn(action)
            row = state['pending']
            if row is not None:
                after = np.asarray(env.robot.get_ee_state(env.physics))
                row['after_ee'] = after.tolist()
                row['after_qpos'] = np.asarray(env.robot.get_qpos(env.physics)).tolist()
                row['executed_action'] = np.asarray(action).tolist()
                row['position_error_m'] = float(np.linalg.norm(after[:3]-row['target_xyz']))
                row['rotation_error_deg'] = rotation_error_degrees(after[3:7], row['target_quat_wxyz'])
                row['actual_translation_m'] = float(np.linalg.norm(after[:3]-np.asarray(row['before_ee'])[:3]))
                row['actual_rotation_deg'] = rotation_error_degrees(row['before_ee'][3:7], after[3:7])
                log.write(json.dumps(row)+'\n')
                state['step'] += 1
                state['pending'] = None
            return result

        env.robot.get_qpos_from_ee_pos = traced_ik
        env.step = traced_step
        return env

    agent.predict, agent.model.infer = traced_predict, traced_infer
    official.load_env = traced_load
    try:
        yield
    finally:
        official.load_env = load_env
        agent.predict, agent.model.infer = predict, infer
        for obj, name, original in restorations:
            setattr(obj, name, original)
        log.close()
