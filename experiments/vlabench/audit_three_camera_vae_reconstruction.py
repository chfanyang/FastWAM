"""Separate representation/quantization/VAE errors on held-out select_book windows."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.datasets.vlabench_video import pose_xyz_euler_to_wxyz
from fastwam.models.wan22.helpers.loader import _load_registered_model


def metrics(p, g, target, target_g):
    delta = (p[..., :3]-target[..., :3]).double()
    q = torch.nn.functional.normalize(p[..., 3:].double(), dim=-1)
    qt = torch.nn.functional.normalize(target[..., 3:].double(), dim=-1)
    angle = 2*torch.acos((q*qt).sum(-1).abs().clamp(0,1))*180/torch.pi
    return dict(position_l2_mm=delta.norm(dim=-1).mul(1000).tolist(),
                position_axis_mae_mm=delta.abs().mean(-1).mul(1000).tolist(),
                rotation_deg=angle.tolist(),
                gripper_abs_error=(g-target_g).abs().flatten().tolist(),
                gripper_binary_error=((g>=.5)!=(target_g>=.5)).float().flatten().tolist())


def summarize(rows, phase, first_n):
    arrays={}
    for r in rows:
        for key, values in r[phase].items():
            arrays.setdefault(key,[]).extend(v for i,v in enumerate(values)
                if i<first_n and not r['action_is_pad'][i])
    return {k:dict(n=len(v),mean=float(np.mean(v)),p95=float(np.quantile(v,.95)),max=float(np.max(v)))
            for k,v in arrays.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--all-windows',action='store_true')
    parser.add_argument('--shard-rank',type=int,default=0)
    parser.add_argument('--num-shards',type=int,default=1)
    args=parser.parse_args()
    if not 0 <= args.shard_rank < args.num_shards:
        parser.error('Invalid shard rank/count')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    cfg=OmegaConf.load(args.config)
    cfg.data.val.include_text_context=False
    dataset=instantiate(cfg.data.val)
    # Only need EE/gripper: skip reading image columns. Codec/input conventions
    # below are exactly the same as VLABenchVideoDataset.__getitem__.
    dataset.windows.camera_keys=()
    dataset.windows.reader.cameras=()
    codec=dataset.codec
    vae_path=Path('checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth').resolve()
    vae=_load_registered_model(str(vae_path),'wan_video_vae',torch_dtype=torch.bfloat16,device='cuda')
    vae.eval().requires_grad_(False)
    reader=dataset.windows.reader
    windows=[]
    start=0
    for ep,end in zip(reader.episodes,reader.ends):
        # Include early/middle/late starts and tail padding; metrics exclude pads.
        starts = range(end-start) if args.all_windows else np.linspace(0,end-start-1,4,dtype=int)
        for frame in starts:
            windows.append((int(start+frame),int(ep),int(frame)))
        start=end
    windows=windows[args.shard_rank::args.num_shards]
    rows=[]
    with torch.inference_mode():
        for index,ep,frame in windows:
            raw=dataset.windows[index]
            state,action=raw['raw_state']['default'],raw['raw_action']['default']
            base=pose_xyz_euler_to_wxyz(state[0])
            target=pose_xyz_euler_to_wxyz(action)
            poses=torch.cat((base[None],target))
            grips=torch.cat((raw['raw_state']['gripper_open'][:1],raw['raw_action']['gripper_open']))
            video=codec.encode(poses,grips).unsqueeze(0).cuda()
            base=base.cuda()
            direct_p,direct_g=codec.decode(video,base)
            rounded_p,rounded_g=codec.decode(video.bfloat16().float(),base)
            latent=vae.encode(video.bfloat16(),device='cuda',tiled=False)
            rebuilt=vae.decode(latent,device='cuda',tiled=False).detach().float().clamp(-1,1)
            pred_p,pred_g=codec.decode(rebuilt,base)
            direct_p,direct_g=direct_p[0,1:].cpu(),direct_g[0,1:].cpu()
            rounded_p,rounded_g=rounded_p[0,1:].cpu(),rounded_g[0,1:].cpu()
            pred_p,pred_g=pred_p[0,1:].cpu(),pred_g[0,1:].cpu()
            row=dict(episode=ep,frame_start=frame,val_index=index,
                action_is_pad=raw['action_is_pad'].tolist(),
                direct_vs_target=metrics(direct_p,direct_g,target,grips[1:]),
                bf16_pixels_vs_direct=metrics(rounded_p,rounded_g,direct_p,direct_g),
                vae_vs_direct=metrics(pred_p,pred_g,direct_p,direct_g),
                vae_vs_target=metrics(pred_p,pred_g,target,grips[1:]),
                target_pose=target.tolist(),direct_pose=direct_p.tolist(),vae_pose=pred_p.tolist(),
                target_gripper=grips[1:].tolist(),vae_gripper=pred_g.tolist(),
                raymap_psnr_db=float(-10*torch.log10(((rebuilt-video)/2).square().mean())))
            rows.append(row)
            (args.output/'windows.json').write_text(json.dumps(rows,indent=2))
            print(f'window {len(rows)}/{len(windows)} episode={ep} start={frame}',flush=True)
    phases=('direct_vs_target','bf16_pixels_vs_direct','vae_vs_direct','vae_vs_target')
    summary=dict(config=str(args.config),vae=str(vae_path),
        stats_fingerprint=codec.norm_stats.fingerprint(),codec=codec.metadata(),
        windows=len(rows),heldout_episodes=reader.episodes,
        all_windows=args.all_windows,shard_rank=args.shard_rank,num_shards=args.num_shards,
        padding='Excluded from action metrics; included in VAE encoding as in training',
        first8={k:summarize(rows,k,8) for k in phases},
        all16={k:summarize(rows,k,16) for k in phases})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['first8'],indent=2),flush=True)


if __name__=='__main__':
    main()
