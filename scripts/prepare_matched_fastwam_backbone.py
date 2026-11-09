"""Explicitly generate the Wan2.1 Action Expert initialization; no training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--model-config',type=Path,default=ROOT/'configs/model/fastwam_matched_wan21.yaml')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    from omegaconf import OmegaConf
    config_path=args.model_config if args.model_config.is_absolute() else ROOT/args.model_config
    cfg=OmegaConf.load(config_path)
    if args.output is None:
        args.output=Path(str(cfg.action_dit_pretrained_path))
    if not args.output.is_absolute():args.output=ROOT/args.output
    if args.output.exists():raise FileExistsError(args.output)
    if not args.execute:
        print('Would load original Wan2.1 on CPU and interpolate ActionDiT backbone',
              f'(hidden={cfg.action_dit_config.hidden_dim}, FFN={cfg.action_dit_config.ffn_dim}) into',args.output)
        return
    import torch
    from fastwam.models.wan22.helpers.loader import load_wan_video_components
    from fastwam.models.wan22.action_dit import ActionDiT
    from scripts.preprocess_action_dit_backbone import _resize_tensor_to_shape
    torch.set_num_threads(2);torch.manual_seed(42)
    vc=OmegaConf.to_container(cfg.video_dit_config,resolve=True)
    ac=OmegaConf.to_container(cfg.action_dit_config,resolve=True)
    for key in ('num_heads','attn_head_dim','num_layers'):
        if vc[key]!=ac[key]:raise ValueError(f'MoT expert mismatch for {key}')
    src=load_wan_video_components(device='cpu',torch_dtype=torch.bfloat16,
        model_id='Wan-AI/Wan2.1-T2V-1.3B',model_variant='wan2.1-t2v-1.3b',
        load_text_encoder=False,dit_config=vc)
    target=ActionDiT(**ac).to(torch.bfloat16);vs=src.dit.state_dict();ts=target.state_dict()
    backbone={}
    for k in sorted(ActionDiT.backbone_key_set(ts)):
        value=_resize_tensor_to_shape(vs[k],tuple(ts[k].shape))
        if vs[k].ndim>=2 and vs[k].shape[-1]!=ts[k].shape[-1]:
            value=(value.float()*(vs[k].shape[-1]/ts[k].shape[-1])**0.5).to(ts[k].dtype)
        backbone[k]=value.detach().cpu().contiguous()
    payload=dict(source_model='Wan-AI/Wan2.1-T2V-1.3B',model_config=str(config_path),
        policy=dict(alpha_scaling=True,interpolation='sequential_1d_linear_align_corners_true',
                    skip_prefixes=list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES)),
        meta={k:ac[k] for k in ActionDiT.ACTION_BACKBONE_META_KEYS},backbone_state_dict=backbone)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:torch.save(payload,f)
    print('Saved',args.output)


if __name__=='__main__':main()
