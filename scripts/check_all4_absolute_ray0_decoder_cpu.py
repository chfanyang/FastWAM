"""CPU checks of padded GT construction and observed-state decoding, no GPU."""
import json
from pathlib import Path
import torch
import prepare_all4_absolute_ray0_decoder as workflow
from fastwam.representations.rothko import quaternion_wxyz_to_matrix as rotation

def main():
    torch.set_num_threads(2)
    inputs=workflow.Inputs();store=workflow.core.EpisodeStore(4);rows=[]
    for pool_name,pool in [('train',inputs.train),('validation',inputs.val)]:
        for root in inputs.roots:
            ep=next(e for e in pool if e.dataset_root==root)
            arrays=store.get(ep)
            for start in [0,ep.length-16,ep.length-1]:
                ref=workflow.core.WindowRef(ep,start)
                w=workflow.materialize_padded_windows(store,[ref],16)[0]
                pose=torch.from_numpy(w.pose);grip=torch.from_numpy(w.gripper)
                if start==ep.length-1:
                    torch.testing.assert_close(pose[1:],torch.from_numpy(arrays.action_pose[-1:]).expand(16,-1))
                rgb=inputs.target(w)
                pred,g=inputs.codec.decode(rgb,pose[0])
                # Relative translation is clipped by the baseline's fixed bounds.
                r0=rotation(pose[0,3:]);rel=(r0.T@(pose[1:,:3]-pose[0,:3]).T).T
                bound=torch.tensor(inputs.codec.norm_stats.metadata['translation_abs_bounds_xyz_m'])
                expected=(r0@rel.clamp(-bound,bound).T).T+pose[0,:3]
                torch.testing.assert_close(pred[1:,:3],expected,atol=3e-6,rtol=0)
                torch.testing.assert_close(rotation(pred[1:,3:]),rotation(pose[1:,3:]),atol=4e-6,rtol=0)
                torch.testing.assert_close(g,grip,atol=1e-6,rtol=0)
                assert inputs.index(ref)==inputs.offsets[(root,ep.episode_index)][0]+start
                rows.append(dict(pool=pool_name,suite=Path(root).name,episode=ep.episode_index,start=start,valid_targets=int(workflow.valid_steps(w).sum()),passed=True))
    out=workflow.PREP/'cpu_checks.json';out.write_text(json.dumps({'checks':rows,'count':len(rows)},indent=2));print(out)
if __name__=='__main__':main()
