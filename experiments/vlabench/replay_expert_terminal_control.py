"""Test omitted terminal expert waypoint from a saved same-state diagnostic."""
import json
import random
import subprocess
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def main():
    import VLABench
    import VLABench.tasks
    import VLABench.robots
    from VLABench.envs import load_env
    from VLABench.utils.utils import euler_to_quaternion
    root=Path('evaluate_results/vlabench/select_book_collect_replay_same_state_ep1')
    out=root/'terminal_action_control'
    out.mkdir(exist_ok=False)
    saved=np.load(root/'initial_state.npz')
    trajectory=np.load(root/'expert_trajectory.npz')
    records=[json.loads(x) for x in (root/'collection_steps.jsonl').read_text().splitlines()]
    last=records[-1]
    assert last['success'] and last['ik'] is not None
    final_quat=np.array(last['ik']['quat'])
    final_euler=Rotation.from_quat(final_quat[[1,2,3,0]]).as_euler('xyz')
    final_world=np.r_[last['ik']['xyz'],final_euler,last['control'][-2:]]
    summary={}
    for mode in ('raw_plus_terminal','binary_plus_terminal'):
        random.seed(42);np.random.seed(42)
        track=json.loads((Path(VLABench.__file__).parent/'configs/evaluation/tracks/track_1_in_distribution.json').read_text())
        scene=track['select_book'][1]
        scene['task']['target_entity']=scene['task']['target_entity'][0]
        env=load_env('select_book',episode_config=scene,random_init=False,eval=False,run_mode='eval')
        env.reset()
        for name in ('body_pos','body_quat','geom_pos','geom_quat'):
            assert np.array_equal(getattr(env.physics.model,name),saved[name]),name
        spec=int(saved['state_spec'])
        mujoco.mj_setState(env.physics.model.ptr,env.physics.data.ptr,saved['integration_state'],spec)
        warm=env.physics.data.qacc_warmstart.copy()
        env.physics.forward();env.physics.data.qacc_warmstart[:]=warm
        check=np.empty_like(saved['integration_state'])
        mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,check,spec)
        assert np.array_equal(check,saved['integration_state'])
        env.timestep=0
        commands=np.vstack((trajectory['world_waypoints'],final_world))
        stream=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1440x480','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(out/f'{mode}.mp4')],stdin=subprocess.PIPE)
        rows=[]
        try:
            for i,command in enumerate(commands):
                canvas=np.concatenate([env.physics.render(height=480,width=480,camera_id=k) for k in (2,0,3)],axis=1)
                stream.stdin.write(canvas.astype('uint8').tobytes())
                before=np.asarray(env.robot.get_ee_state(env.physics)).copy()
                cmd=command.copy()
                if mode=='binary_plus_terminal':
                    origin=np.asarray(env.get_robot_frame_position())
                    cmd[:3]=(cmd[:3]-origin).astype(np.float32).astype(np.float64)+origin
                    cmd[3:6]=cmd[3:6].astype(np.float32).astype(np.float64)
                    cmd[-2:]=.04 if cmd[-1]>.03 else 0.
                ok,qpos=env.robot.get_qpos_from_ee_pos(physics=env.physics,pos=cmd[:3],quat=euler_to_quaternion(*cmd[3:6]))
                result=env.step(np.r_[qpos,cmd[-2:]])
                rows.append(dict(step=i,success=bool(result.last()),ik_success=bool(ok),before=before.tolist(),after=np.asarray(env.robot.get_ee_state(env.physics)).tolist(),target=cmd.tolist()))
                if i%20==0:print(mode,i,'success',bool(result.last()),flush=True)
                if result.last():break
            summary[mode]=dict(steps=len(rows),success=any(r['success'] for r in rows),initial_state_max_error=0.,ik_failures=sum(not r['ik_success'] for r in rows))
            (out/f'{mode}_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print(mode,summary[mode],flush=True)
        finally:
            stream.stdin.close();code=stream.wait();env.close()
            if code:raise RuntimeError('ffmpeg failed')


if __name__=='__main__':main()
