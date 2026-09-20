#!/usr/bin/env python3
"""Small, read-only four-suite GT -> original Wan2.1 VAE -> action audit.

Default first/middle/last windows are smoke coverage, not an extreme-value audit.
Supply --indices-json {suite_directory_basename: [dataset_local_indices]} to
audit a separately reviewed ordinary/extreme/large-rotation window manifest.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--indices-json', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('HF_DATASETS_CACHE', str(args.output.resolve()/'hf_cache'))
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.models.wan22.helpers.loader import _load_custom_wan_vae
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.utils import misc
    from libero_all_absolute_workflow import ROOT, TASK, check_inputs

    os.chdir(ROOT)
    misc.register_work_dir(args.output.resolve())
    identity = check_inputs()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the original BF16 VAE audit')
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['task='+TASK,
            'data.train.latent_cache_dir=null', 'data.train.latent_cache_only=false'])
    OmegaConf.resolve(cfg)
    roots = list(cfg.data.train.dataset_dirs)
    selected = json.loads(args.indices_json.read_text()) if args.indices_json else None
    if selected is not None and set(selected) != {Path(r).name for r in roots}:
        raise ValueError('Explicit audit manifest must cover exactly the four suite directories')
    vae = _load_custom_wan_vae(ROOT/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
        vae_class=WanVideoVAE, torch_dtype=torch.bfloat16, device='cuda:0')
    rows = []
    with torch.inference_mode():
        for root in roots:
            suite = Path(root).name
            data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
            data_cfg.dataset_dirs = [root]
            dataset = instantiate(data_cfg)
            indices = selected[suite] if selected is not None else sorted({0, len(dataset)//2, len(dataset)-1})
            if not indices or any(not isinstance(i,int) or i < 0 or i >= len(dataset) for i in indices):
                raise ValueError(f'Invalid indices for {suite}: {indices}')
            for index in indices:
                sample = dataset[index]
                video = sample['raymap']
                gt = sample['future_endpose'].double()
                valid = ~sample['action_is_pad'].bool()
                latent = vae.encode(video.unsqueeze(0).to('cuda:0', torch.bfloat16), device='cuda:0', tiled=False)
                reconstructed = vae.decode(latent, device='cuda:0', tiled=False).float().clamp(-1,1)[0].cpu()
                for kind, pixels in [('direct', video), ('vae', reconstructed)]:
                    pred, gripper = dataset.raymap_codec.decode(pixels)
                    pred = pred[1:].double()
                    qp = torch.nn.functional.normalize(pred[:,3:],dim=-1)
                    qg = torch.nn.functional.normalize(gt[:,3:],dim=-1)
                    degrees = torch.rad2deg(2*torch.acos((qp*qg).sum(-1).abs().clamp(0,1)))
                    position = (pred[:,:3]-gt[:,:3]).norm(dim=-1)*1000
                    rows.append(dict(suite=suite,index=index,kind=kind,
                        valid=valid.tolist(),position_l2_mm=position.tolist(),rotation_deg=degrees.tolist(),
                        gripper_abs_error=(gripper[1:]-sample['future_gripper']).abs().flatten().tolist()))
                print(f'{suite} window {index} completed',flush=True)
                del latent, reconstructed
            del dataset
    summary = {}
    for suite in [Path(r).name for r in roots]:
        summary[suite] = {}
        for kind in ['direct','vae']:
            metrics = {}
            for horizon in [8,16]:
                for key in ['position_l2_mm','rotation_deg','gripper_abs_error']:
                    values = [value for row in rows if row['suite']==suite and row['kind']==kind
                              for value,valid in zip(row[key][:horizon],row['valid'][:horizon]) if valid]
                    t = torch.tensor(values,dtype=torch.float64)
                    metrics[f'h{horizon}_{key}'] = dict(count=len(values),mean=t.mean().item(),
                        median=t.quantile(.5).item(),p95=t.quantile(.95).item(),maximum=t.max().item()) if values else dict(count=0)
            summary[suite][kind]=metrics
    report = dict(identity=identity,selection='explicit_manifest' if selected is not None else 'first_middle_last_smoke',
        note='GT reconstruction only; excludes padding; no DiT prediction or closed-loop test; text conditions not audited',
        rows=rows,summary=summary)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
