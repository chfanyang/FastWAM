"""Small CPU-only actual-data audit; no pretrained model is instantiated."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    from hydra import compose,initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.representations.rothko import quaternion_wxyz_to_matrix
    torch.set_num_threads(2);register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        cfg=compose(config_name='train',overrides=['task=libero_goal_matched_fastwam_wan21'])
    dataset=instantiate(cfg.data.train)
    source=OmegaConf.load(ROOT/'runs/libero_goal_rothko_absolute_ray0_2cam224_full_wan21_1_3b_1e-4/2026-09-21_18-42-23/config.yaml')
    original=instantiate(source.data.train)
    assert len(dataset)==len(original)==52895
    episodes=dataset._repeat_episodes
    first_tail=episodes[0][3]-1
    records=[]
    for index in [0,first_tail-15,first_tail,26447,52894]:
        a=dataset[index];b=original[index]
        torch.testing.assert_close(a['rgb_latents'],b['rgb_latents'],rtol=0,atol=0)
        for k in ['context','context_mask','action_is_pad','image_is_pad']:
            torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        pose,grip,bad=dataset.numeric_codec.recover(a['action'][None],a['current_endpose'][None])
        assert not bad.any()
        seq=torch.cat([a['current_endpose'][None],a['future_endpose']],0)
        opening=torch.cat([(a['proprio'][0,-1:]+1)/2,a['future_gripper'].flatten()])[:,None]
        image=dataset.raymap_codec.encode(seq,opening)
        expected,eg=dataset.raymap_codec.decode(image,a['current_endpose'])
        torch.testing.assert_close(pose[0,:,:3],expected[1:,:3],rtol=0,atol=2e-6)
        torch.testing.assert_close(quaternion_wxyz_to_matrix(pose[0,:,3:]),quaternion_wxyz_to_matrix(expected[1:,3:]),rtol=0,atol=2e-6)
        torch.testing.assert_close(grip[0],eg[1:],rtol=0,atol=2e-6)
        assert a['action'].shape==(16,8) and a['proprio'].shape==(16,8)
        records.append(dict(index=index,padding_steps=int(a['action_is_pad'].sum()),rgb_and_text_bit_exact=True,pose_targets_match_direct_raymap_decode=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(dataset_length=len(dataset),records=records,scope='CPU actual-data targets and cache parity only; no model run'),indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
