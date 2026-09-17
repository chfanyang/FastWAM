"""Four-GPU, no-update decoder backward audit on real cached windows."""
import argparse
import json
import os
import time

import torch
from train_wan21_decoder_cached import ROOT, CachedWindows, core, errors


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rank=int(os.environ['LOCAL_RANK'])
    device=torch.device('cuda',rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.8,device)
    run=ROOT/'runs/libero_wan21_vae_decoder_cached_bs128_lr1e5_const_w100_ep2_fast_20260913'
    args=argparse.Namespace(**json.loads((run/'training_config.json').read_text())['arguments'])
    cache=CachedWindows(args)
    _,episodes=core.discover_episodes(args,rank=rank)
    store=core.EpisodeStore(100)
    windows=core.sample_uniform_windows_per_episode(store,episodes,10,16)
    checkpoint=run/'checkpoint_step001500_full.pt'
    vae=core.load_base_vae(args.base_vae,args.vae_variant,device,torch.float32)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    vae.model.decoder.load_state_dict(state['decoder'])
    vae.model.conv2.load_state_dict(state['conv2'])
    del state
    params=core.prepare_decoder_finetune(vae)
    masks=core.build_loss_masks(args,device)
    output=ROOT/'evaluate_results/libero_decoder_offline/rothko_decoder_backward_audit_gpu4to7_20260914'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index in (rank*200+5, rank*200+97):
        w=windows[index]
        truth=torch.from_numpy(w.pose).unsqueeze(0).to(device)
        target=cache.codec.encode(torch.from_numpy(w.pose),torch.from_numpy(w.gripper)).unsqueeze(0).to(device)
        z=cache.latent(core.WindowRef(w.episode,w.start)).unsqueeze(0).to(device)
        with core.autocast_context(device,True):
            reconstruction=vae.model.decode(z,vae.scale)
        l1,parts=core.compute_reconstruction_loss(reconstruction.float(),target,masks,args)
        pred,_=cache.codec.decode(reconstruction.float().clamp(-1,1),truth[:,0])
        pe,re=errors(pred[:,1:],truth[:,1:])
        for name,loss in [('map_l1',l1),('translation_mm',pe.mean()),('rotation_deg',re.mean())]:
            start=time.monotonic()
            gradients=torch.autograd.grad(loss,params,retain_graph=True)
            finite=all(bool(torch.isfinite(g).all()) for g in gradients)
            norm=torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt().item()
            largest=max(g.detach().abs().max().item() for g in gradients)
            row={'index':index,'suite':w.episode.dataset_root,'episode':w.episode.episode_index,'start':w.start,
                 'checkpoint':str(checkpoint),'physical_gpu':rank+4,'loss':name,'value':loss.item(),
                 'parameter_grad_finite':finite,'parameter_grad_norm':norm,'parameter_grad_max':largest,
                 'backward_seconds':time.monotonic()-start,'peak_memory_gib':torch.cuda.max_memory_allocated(device)/2**30}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del gradients
        del reconstruction,l1,parts,pred,pe,re,loss
        (output/f'rank{rank}.json').write_text(json.dumps(rows,indent=2))
    print('COMPLETE rank',rank,flush=True)


if __name__=='__main__':main()
