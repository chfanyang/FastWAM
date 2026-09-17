"""Compare raw and exported expert commands in their original recorded scene."""
import io
import json
import os
import random
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation


def rotation_error(a, b):
    return float(np.degrees((Rotation.from_quat(a[[1,2,3,0]]).inv() *
                            Rotation.from_quat(b[[1,2,3,0]])).magnitude()))


def main():
    import VLABench.robots
    import VLABench.tasks
    from VLABench.envs import load_env
    from VLABench.utils.utils import euler_to_quaternion

    output = Path('evaluate_results/vlabench/expert_replay_select_poker_raw472_image1194')
    output.mkdir(parents=True, exist_ok=False)
    raw_path = Path('data/vlabench_image_video_audit/raw_sample/select_poker_episode_472.hdf5')
    with h5py.File(raw_path) as f:
        g = next(iter(f['data'].values()))
        config = json.loads(g['meta_info/episode_config'][()].decode())
        actions = g['trajectory'][:].astype(np.float64)
        reference = g['observation/ee_state'][:].astype(np.float64)
        reference_rgb = g['observation/rgb'][0]
        language = g['instruction'][0].decode()
    table = pq.read_table('data/vlabench_primitive_ft_lerobot/data/chunk-001/episode_001194.parquet', columns=['actions','state']).to_pydict()
    exported = np.asarray(table['actions'], dtype=np.float64)
    converted = np.column_stack((exported[:, :6], np.repeat(np.where(exported[:,6:7] >= .5, .04, 0.), 2, axis=1)))
    assert len(converted) == len(actions)
    assert np.allclose(converted[:, :6], actions[:, :6], atol=1e-6)
    Image.fromarray(np.concatenate([reference_rgb[2], reference_rgb[0], reference_rgb[3]], axis=1)).save(output/'recorded_initial_rgb.png')
    summary = dict(task='select_poker', raw_episode=472, image_episode=1194, language=language,
                   note='No DiT/VAE/Rothko: isolate exported absolute EE command mapping and official IK execution.',
                   command_max_error=np.abs(converted-actions).max(axis=0).tolist(), runs={})
    (output/'source_scene.json').write_text(json.dumps(config, indent=2))
    for mode, commands in [('raw',actions), ('exported',converted)]:
        random.seed(42);np.random.seed(42)
        env = load_env('select_poker', episode_config=config, random_init=False, eval=False, run_mode='eval')
        env.reset()
        writer = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1440x480','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output/f'{mode}.mp4')], stdin=subprocess.PIPE)
        rows=[];success=False
        try:
            origin=np.asarray(env.get_robot_frame_position())
            initial=np.asarray(env.robot.get_ee_state(env.physics))
            with (output/f'{mode}_steps.jsonl').open('w') as log:
                for i, command in enumerate(commands):
                    obs=env.get_observation(require_pcd=False)
                    canvas=np.concatenate([obs['rgb'][2],obs['rgb'][0],obs['rgb'][3]],axis=1).astype('uint8')
                    writer.stdin.write(canvas.tobytes())
                    if i==0: Image.fromarray(canvas).save(output/f'{mode}_initial_rgb.png')
                    before=np.asarray(obs['ee_state']).copy()
                    xyz=command[:3]+origin
                    quat=np.asarray(euler_to_quaternion(*command[3:6]))
                    ik_ok,qpos=env.robot.get_qpos_from_ee_pos(physics=env.physics,pos=xyz,quat=quat)
                    result=env.step(np.concatenate([qpos,command[-2:]]))
                    after=np.asarray(env.robot.get_ee_state(env.physics)).copy()
                    success=bool(result.last()) or success
                    row=dict(step=i,target_xyz=xyz.tolist(),target_wxyz=quat.tolist(),fingers=command[-2:].tolist(),ik_success=bool(ik_ok),before_ee=before.tolist(),after_ee=after.tolist(),target_error_mm=float(np.linalg.norm(after[:3]-xyz)*1000),target_rotation_error_deg=rotation_error(after[3:7],quat),before_vs_recorded_mm=float(np.linalg.norm(before[:3]-reference[i,:3])*1000),success=bool(result.last()))
                    rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    if i%10==0: print(mode,i,'ik',bool(ik_ok),'position_error_mm',row['target_error_mm'],'success',success,flush=True)
                    if result.last(): break
            summary['runs'][mode]=dict(success=success,steps=len(rows),ik_failures=sum(not r['ik_success'] for r in rows),initial_vs_recorded_mm=float(np.linalg.norm(initial[:3]-reference[0,:3])*1000),initial_rotation_error_deg=rotation_error(initial[3:7],reference[0,3:7]),mean_target_error_mm=float(np.mean([r['target_error_mm'] for r in rows])),mean_before_vs_recorded_mm=float(np.mean([r['before_vs_recorded_mm'] for r in rows])))
            (output/'summary.json').write_text(json.dumps(summary,indent=2))
        finally:
            writer.stdin.close()
            code=writer.wait()
            env.close()
            if code: raise RuntimeError(f'ffmpeg exit {code}')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
