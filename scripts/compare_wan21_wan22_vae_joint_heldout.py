"""Paired held-out VAE round trips; four independent shards, no policy inference."""
import argparse
import hashlib
import json
import gc
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import finetune_rothko_vae_decoder as core
from compare_libero_rothko_decoders import errors, summarize
from fastwam.representations.libero_rothko import LiberoRothkoCodec

RUNS = {
    'wan21': ('libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2', 'Wan2.1'),
    'wan22': ('libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2_rerun_fixed', 'Wan2.2'),
}

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wan21-steps', type=int, nargs='+')
    args = parser.parse_args()
    assert 0 <= args.shard < 4
    torch.set_num_threads(2)
    torch.manual_seed(42)
    configs = {k: argparse.Namespace(**json.loads((ROOT/'runs'/v[0]/'training_config.json').read_text())['arguments']) for k,v in RUNS.items()}
    c = configs['wan21']
    _, heldout = core.discover_episodes(c)
    _, other = core.discover_episodes(configs['wan22'])
    assert heldout == other, 'Validation episodes differ'
    for key in ('action_horizon','center_frac','boundary_margin','outer_margin','focal','center_scale','dir_scale','norm_stats_path'):
        assert getattr(c,key) == getattr(configs['wan22'],key), key
    windows = core.sample_uniform_windows_per_episode(core.EpisodeStore(80), heldout, 10, 16)
    assert len(windows) == 800
    manifest = [{'dataset':w.episode.dataset_root,'episode':w.episode.episode_index,'task':w.episode.task,'start':w.start} for w in windows]
    digest = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    selected = [(i,w) for i,w in enumerate(windows) if i%4 == args.shard]
    out = args.output / f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    (out/'manifest.json').write_text(json.dumps({'full_manifest_sha256':digest,'windows':manifest,'shard':args.shard,'decode_mode':'robust_joint','anchor_alpha':0,'precision':'bf16','metric':'translation L2 mm; stable quaternion angle deg; future frames 1..16'},indent=2))
    device=torch.device('cuda:0')
    stats=core.load_norm_stats(c.norm_stats_path,c)
    codec=LiberoRothkoCodec(norm_stats=stats,expected_action_horizon=16,decode_mode='robust_joint',decode_anchor_alpha=0,decode_block_grid=4)
    rows=[{'window_index':i,'suite':next(s for s,d in core.SUITE_DIRS.items() if Path(w.episode.dataset_root).name==d),'errors':{}} for i,w in selected]
    began=time.monotonic()
    evaluations = [(f'wan21_step{step:06d}', 'wan21', step) for step in args.wan21_steps] if args.wan21_steps else [(name,name,7498) for name in RUNS]
    for name,config_key,step in evaluations:
        run,prefix=RUNS[config_key]
        checkpoint=ROOT/'runs'/run/f'{prefix}_VAE_libero_rothko_step{step:06d}.safetensors'
        print('LOAD',name,checkpoint,flush=True)
        vae=core.load_base_vae(str(checkpoint),configs[config_key].vae_variant,device,torch.bfloat16)
        vae.model.eval()
        for j,(_,w) in enumerate(selected):
            target,encoded=core.build_target_batch([w],stats,device,torch.bfloat16,c)
            truth=encoded[0].pose.unsqueeze(0)
            latent=core.vae_encode(vae.model,target,vae.scale)
            recon=vae.model.decode(latent,vae.scale).float().clamp(-1,1)
            pred,grip=codec.decode(recon,truth[:,0])
            pe,re=errors(pred[:,1:],truth[:,1:])
            assert torch.isfinite(pe).all() and torch.isfinite(re).all()
            rows[j]['errors'][name]={'translation_mm':pe[0].cpu().tolist(),'rotation_deg':re[0].cpu().tolist()}
            if j==0 or (j+1)%20==0:
                print(name,j+1,len(selected),'elapsed',round(time.monotonic()-began,1),flush=True)
        del vae,latent,recon,pred,target
        gc.collect()
        torch.cuda.empty_cache()
        (out/'per_window_errors.json').write_text(json.dumps(rows))
    report={'windows':len(rows),'future_poses':len(rows)*16,'manifest_sha256':digest,'overall':summarize(rows),'elapsed_seconds':time.monotonic()-began}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    main()
