"""Bounded read-only autograd audit; never optimizer.step, never edit deployment codec."""
import argparse
import json
import time
from pathlib import Path

import torch
from train_wan21_decoder_cached import ROOT, CachedWindows, core, errors
from fastwam.representations import rothko


def grad_report(g):
    finite = torch.isfinite(g)
    safe = torch.where(finite, g, 0)
    return {'finite': bool(finite.all()), 'nonfinite': int((~finite).sum()),
            'norm': safe.double().norm().item(), 'max': safe.abs().max().item(),
            'nonzero_fraction': (safe != 0).float().mean().item()}


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.7, device)
    run = ROOT/'runs/libero_wan21_vae_decoder_cached_bs128_lr1e5_const_w100_ep2_fast_20260913'
    args = argparse.Namespace(**json.loads((run/'training_config.json').read_text())['arguments'])
    cache = CachedWindows(args)
    _, episodes = core.discover_episodes(args, rank=0)
    store = core.EpisodeStore(100)
    windows = core.sample_uniform_windows_per_episode(store, episodes, 10, 16)
    selected = [windows[i] for i in (0,200,400,600)]
    ckpt = run/'Wan2.1_VAE_libero_rothko_step001800.safetensors'
    vae = core.load_base_vae(str(ckpt), args.vae_variant, device, torch.bfloat16)
    output = ROOT/'evaluate_results/libero_decoder_offline/rothko_action_gradient_audit_20260914'
    output.mkdir(exist_ok=False, parents=True)
    rows = []
    for w in selected:
        truth = torch.from_numpy(w.pose).unsqueeze(0).to(device)
        target = cache.codec.encode(torch.from_numpy(w.pose),torch.from_numpy(w.gripper)).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = cache.latent(core.WindowRef(w.episode,w.start)).unsqueeze(0).to(device)
            decoded = vae.model.decode(latent,vae.scale).float()
        for label, values in [('ideal',target),('vae1800',decoded)]:
            x = values.detach().clone().requires_grad_(True)
            pred,_ = cache.codec.decode(x.clamp(-1,1),truth[:,0])
            pe,re = errors(pred[:,1:],truth[:,1:])
            for loss_name, loss in [('translation_mm',pe.mean()),('rotation_deg',re.mean())]:
                start=time.monotonic()
                g=torch.autograd.grad(loss,x,retain_graph=True)[0]
                row={'suite':w.episode.dataset_root,'episode':w.episode.episode_index,'start':w.start,
                     'input':label,'loss':loss_name,'value':loss.item(),'gradient':grad_report(g),
                     'backward_seconds':time.monotonic()-start}
                rows.append(row)
                print(json.dumps(row),flush=True)
            del x,pred,pe,re,g
    primitives=[]
    for label, mat in [('identity',torch.eye(3,dtype=torch.float64,device=device)),
                       ('diagonal_distinct',torch.diag(torch.tensor([.2,.3,.5],device=device,dtype=torch.float64)))]:
        for op,fn in [('matrix_to_quaternion',rothko.matrix_to_quaternion_wxyz),
                      ('svd_rotation',rothko._proper_rotation_from_correlation)]:
            x=mat.clone().requires_grad_(True)
            y=fn(x)
            g=torch.autograd.grad(y.sum(),x)[0]
            row={'input':label,'op':op,'gradient':grad_report(g)}
            primitives.append(row)
            print('PRIMITIVE',json.dumps(row),flush=True)
    (output/'summary.json').write_text(json.dumps({'checkpoint':str(ckpt),'rows':rows,'primitives':primitives},indent=2))
    print('ALL_COMPLETE',flush=True)


if __name__=='__main__':main()
