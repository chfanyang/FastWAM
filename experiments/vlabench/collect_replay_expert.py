"""Local expert collection and same-state replay; no model or benchmark edits."""
import argparse
import json
import random
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--track-episode', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import VLABench
    import VLABench.robots
    import VLABench.tasks
    from VLABench.envs import load_env
    from VLABench.utils.utils import euler_to_quaternion
    track = json.loads((Path(VLABench.__file__).parent / 'configs/evaluation/tracks/track_1_in_distribution.json').read_text())
    random.seed(42); np.random.seed(42)
    scene = track['select_book'][args.track_episode]
    target = scene['task'].get('target_entity')
    if isinstance(target, list):
        if len(target) != 1:
            raise ValueError('Expert diagnostic requires one target entity')
        scene['task']['target_entity'] = target[0]
    env = load_env('select_book', episode_config=scene, random_init=False, eval=False, run_mode='eval')
    env.reset()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    def get_state():
        value = np.empty(mujoco.mj_stateSize(env.physics.model.ptr, state_spec))
        mujoco.mj_getState(env.physics.model.ptr, env.physics.data.ptr, value, state_spec)
        return value
    initial = get_state()
    initial_ee = np.asarray(env.robot.get_ee_state(env.physics)).copy()
    origin = np.asarray(env.get_robot_frame_position()).copy()
    model_arrays = {name: np.array(getattr(env.physics.model, name), copy=True)
                    for name in ('body_pos','body_quat','geom_pos','geom_quat')}
    np.savez(args.output/'initial_state.npz', integration_state=initial, state_spec=int(state_spec), ee=initial_ee, **model_arrays)
    (args.output/'scene.json').write_text(json.dumps(env.save(), indent=2))
    original_step = env.step
    original_ik = env.robot.get_qpos_from_ee_pos
    rows, controls, before_ee, after_ee = [], [], [], []
    stream = None
    pending_ik = None
    mode = 'collection'

    def camera_frame():
        return np.concatenate([env.physics.render(height=480,width=480,camera_id=i) for i in (2,0,3)],axis=1).astype('uint8')

    def start_video(name):
        return subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1440x480','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/f'{name}.mp4')],stdin=subprocess.PIPE)

    def finish_video():
        stream.stdin.close()
        if stream.wait(): raise RuntimeError('video encoding failed')

    def traced_ik(*a, **kw):
        nonlocal pending_ik
        result = original_ik(*a, **kw)
        pending_ik = dict(success=bool(result[0]),xyz=np.asarray(kw['pos']).tolist(),quat=np.asarray(kw['quat']).tolist())
        return result

    def traced_step(action=None):
        nonlocal pending_ik
        if action is None:
            return original_step(action)
        before = np.asarray(env.robot.get_ee_state(env.physics)).copy()
        stream.stdin.write(camera_frame().tobytes())
        result = original_step(action)
        after = np.asarray(env.robot.get_ee_state(env.physics)).copy()
        row = dict(step=len(rows),before=before.tolist(),after=after.tolist(),control=np.asarray(action).tolist(),ik=pending_ik,success=bool(result.last()))
        if pending_ik:
            row['target_error_mm']=float(np.linalg.norm(after[:3]-pending_ik['xyz'])*1000)
        rows.append(row); pending_ik=None
        if mode=='collection':
            controls.append(np.asarray(action).copy());before_ee.append(before);after_ee.append(after)
        if len(rows)%10==0: print(mode,'step',len(rows),'success',row['success'],flush=True)
        return result

    def save_rows(name):
        (args.output/f'{name}_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def restore():
        random.seed(42);np.random.seed(42)
        env.step=original_step
        env.reset()
        for name,value in model_arrays.items():
            if not np.array_equal(getattr(env.physics.model,name),value):
                raise RuntimeError(f'Model geometry changed across reset: {name}')
        mujoco.mj_setState(env.physics.model.ptr,env.physics.data.ptr,initial,state_spec)
        env.physics.forward()
        env.physics.data.qacc_warmstart[:] = initial_warmstart
        error=float(np.max(np.abs(get_state()-initial)))
        if error>1e-10: raise RuntimeError(f'Incomplete state restoration: {error}')
        env.timestep=0
        env.step=traced_step
        return error

    initial_warmstart=env.physics.data.qacc_warmstart.copy()
    report=dict(task='select_book',track_episode=args.track_episode,seed=42,language=env.task.get_instruction(),
                note='Current official skills; recorded complete MuJoCo integration state restored in the same environment; no DiT/VAE.',runs={})
    try:
        Image.fromarray(camera_frame()).save(args.output/'initial_rgb.png')
        stream=start_video(mode)
        env.step=traced_step;env.robot.get_qpos_from_ee_pos=traced_ik
        waypoints=[];states=[];stages=[]
        for skill in env.get_expert_skill_sequence():
            print('EXPERT_SKILL',skill,flush=True)
            obs,waypoint,stage_success,task_success=skill(env)
            waypoints.extend(waypoint)
            states.extend(np.asarray(o['ee_state']).copy() for o in obs)
            stages.append(dict(skill=str(skill),stage_success=bool(stage_success),task_success=bool(task_success),returned_frames=len(waypoint)))
            del obs
            if task_success: break
        finish_video();stream=None;save_rows(mode)
        world_waypoints=np.asarray(waypoints)
        local_waypoints=world_waypoints-np.r_[origin,np.zeros(5)]
        np.savez(args.output/'expert_trajectory.npz',world_waypoints=world_waypoints,local_waypoints=local_waypoints,recorded_ee=np.asarray(states),executed_controls=np.asarray(controls),before_ee=np.asarray(before_ee),after_ee=np.asarray(after_ee))
        report['runs']['collection']=dict(success=any(r['success'] for r in rows),executed_steps=len(rows),saved_waypoints=len(waypoints),stages=stages)
        collection_rows=rows.copy()
        (args.output/'summary.json').write_text(json.dumps(report,indent=2))
        if not report['runs']['collection']['success']:
            print('Expert collection failed; not treating it as a successful reference.',flush=True)
            return
        for mode in ('joint_exact','raw_saved_ee','exported_binary_ee'):
            restore_error=restore()
            rows=[];pending_ik=None;stream=start_video(mode)
            commands=np.asarray(controls) if mode=='joint_exact' else local_waypoints
            for i,cmd in enumerate(commands):
                if mode=='joint_exact':
                    result=env.step(cmd.copy())
                else:
                    # Same float32 pose and binary gripper conversion as the image export.
                    cmd=cmd.copy() if mode=='raw_saved_ee' else cmd.astype(np.float32).astype(np.float64)
                    fingers=cmd[-2:] if mode=='raw_saved_ee' else np.full(2,.04 if cmd[-1]>.03 else 0.)
                    _,qpos=env.robot.get_qpos_from_ee_pos(physics=env.physics,pos=cmd[:3]+origin,quat=euler_to_quaternion(*cmd[3:6]))
                    result=env.step(np.r_[qpos,fingers])
                if result.last(): break
            finish_video();stream=None;save_rows(mode)
            result=dict(success=any(r['success'] for r in rows),steps=len(rows),initial_integration_state_max_error=restore_error,initial_ee_error_mm=float(np.linalg.norm(np.array(rows[0]['before'])[:3]-initial_ee[:3])*1000))
            if mode=='joint_exact':
                result['max_after_ee_abs_error_vs_collection']=float(np.max(np.abs(np.array([r['after'] for r in rows])-np.array([r['after'] for r in collection_rows[:len(rows)]]))))
            report['runs'][mode]=result
            (args.output/'summary.json').write_text(json.dumps(report,indent=2))
            print(mode,result,flush=True)
    finally:
        if stream is not None: finish_video()
        env.step=original_step;env.robot.get_qpos_from_ee_pos=original_ik
        env.close()
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
