"""Verify complete formal-cache loading and dataset order on the existing 100 probes."""
import argparse
import bisect
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_DATASETS_CACHE',str(a.output.resolve()/'hf_datasets'))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from prepare_goal_axis_angle_cache import Windows
    torch.set_num_threads(2)
    windows=Windows();cfg=windows.cfg
    cached=instantiate(cfg.data.train)
    assert cached.latent_cache_metadata['complete'] and not cached.latent_cache_metadata.get('benchmark_only')
    online_cfg=OmegaConf.create(OmegaConf.to_container(cfg.data.train,resolve=True))
    online_cfg.latent_cache_only=False;online_cfg.latent_cache_dir=None
    online=instantiate(online_cfg)
    old_cfg=OmegaConf.create(OmegaConf.to_container(cfg.data.train,resolve=True))
    old_cfg._target_='fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset'
    old_cfg.latent_cache_dir=old_cfg.pop('source_latent_cache_dir')
    baseline=instantiate(old_cfg)
    selection_path=ROOT/'evaluate_results/libero/offline_audits/axis_angle_vae100_20261108_quaternion/selection.json'
    selection=json.loads(selection_path.read_text())
    by_episode={e['episode_index']:windows.offsets[i] for i,e in enumerate(windows.episodes)}
    indices=[by_episode[r['episode']]+r['start'] for r in selection]
    indices+=sorted(set([0,137,52894])-set(indices))
    counts=Counter();padding=0
    for n,index in enumerate(indices):
        new=cached[index];old=baseline[index];raw=online[index]
        assert new['latent_cache_index']==old['latent_cache_index']==index
        assert new['prompt']==old['prompt']==raw['prompt']
        for key in ('context','context_mask','action','proprio','image_is_pad','raymap_is_pad'):
            torch.testing.assert_close(new[key],old[key],rtol=0,atol=0)
            torch.testing.assert_close(new[key],raw[key],rtol=0,atol=0)
        torch.testing.assert_close(new['rgb_latents'],old['rgb_latents'],rtol=0,atol=0)
        pose,grip=windows.pose_grip(index)
        torch.testing.assert_close(raw['current_endpose'],pose[0],rtol=0,atol=0)
        torch.testing.assert_close(raw['future_endpose'],pose[1:],rtol=0,atol=0)
        torch.testing.assert_close(raw['future_gripper'],grip[1:],rtol=0,atol=0)
        torch.testing.assert_close(raw['raymap'],windows.codec.encode(pose,grip),rtol=0,atol=0)
        assert new['raymap_latents'].shape==(16,5,28,56)
        assert torch.isfinite(new['raymap_latents']).all()
        episode=windows.episodes[bisect.bisect_right(windows.offsets,index)-1]
        counts[episode['tasks'][0]]+=1
        padding+=int(new['raymap_is_pad'].any())
        if n%20==0:print(f'checked {n+1}/{len(indices)}',flush=True)
    assert len(counts)==10
    report=dict(complete=True,dataset_length=len(cached),checked_windows=len(indices),
        historical_selection_windows=len(selection),padding_windows=padding,per_task=dict(counts),
        rgb_context_prompt_action_proprio_masks_unchanged=True,
        raw_parquet_equals_online_raymap=True,formal_cached_dataset_load_passed=True,
        target_cache=str(cached.latent_cache.cache_dir),indices=indices)
    (a.output/'formal_cache_dataset.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
