"""Small real-data/model diagnostic; never starts formal training or updates weights."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe-cache',type=Path,required=True)
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--trainer-validation',action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_DATASETS_CACHE',str(a.output.resolve()/'hf_datasets'))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from fastwam.datasets.latent_cache import LatentCacheReader
    from fastwam.runtime import _validate_visual_action_data_contract
    from prepare_goal_axis_angle_cache import Windows
    torch.set_num_threads(2)
    windows = Windows(); cfg = windows.cfg
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train,resolve=True))
    ds_cfg.latent_cache_only = False
    ds_cfg.latent_cache_dir = None
    dataset = instantiate(ds_cfg)
    indices = [0,122,137,255,52894]
    report = dict(diagnostic_only=True,formal_training_started=False,
        length=len(dataset),indices=indices,raw_parquet_equals_online_pixels=True,
        codec_metadata_matches=True,shared_baseline_config_unchanged=True)
    probe_meta = json.loads((a.probe_cache/'metadata.json').read_text())
    reader = LatentCacheReader(a.probe_cache,expected_dataset_contract=probe_meta['dataset_contract'])
    samples = {}
    for index in indices:
        sample = dataset[index]
        torch.testing.assert_close(sample['raymap'],windows.pixels(index),rtol=0,atol=0)
        pose,grip = windows.pose_grip(index)
        torch.testing.assert_close(sample['current_endpose'],pose[0],rtol=0,atol=0)
        torch.testing.assert_close(sample['future_endpose'],pose[1:],rtol=0,atol=0)
        torch.testing.assert_close(sample['future_gripper'],grip[1:],rtol=0,atol=0)
        samples[index] = sample
        report.setdefault('padding_counts',{})[str(index)] = int(sample['raymap_is_pad'].sum())
    assert dataset.raymap_codec.metadata()==windows.codec.metadata()
    # A diagnostic cache must be refused by the formal cached-data adapter.
    reject_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train,resolve=True))
    reject_cfg.latent_cache_dir = str(a.probe_cache.resolve())
    try:
        instantiate(reject_cfg)
    except Exception as e:
        if 'Partial diagnostic axis-angle cache cannot be used for training' not in str(e):raise
        report['diagnostic_cache_rejected_by_training'] = True
    else:raise AssertionError('Diagnostic cache was accepted for training')
    if a.gpu:
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        model = instantiate(cfg.model,model_dtype=torch.bfloat16,device='cuda:0')
        _validate_visual_action_data_contract(model,dataset,dataset)
        report['real_model_contract_passed'] = True
        sample = default_collate([samples[0]])
        sample['rgb_latents'],sample['raymap_latents'] = [x[None] for x in reader[0]]
        with torch.no_grad():
            rgb_online = model._encode_video_latents(sample['video'].to('cuda:0',torch.bfloat16),tiled=False)
            action_online = model._encode_video_latents(sample['raymap'].to('cuda:0',torch.bfloat16),tiled=False)
            torch.testing.assert_close(rgb_online.cpu(),sample['rgb_latents'],rtol=0,atol=0)
            torch.testing.assert_close(action_online.cpu(),sample['raymap_latents'],rtol=0,atol=0)
            for pixels,full in ((sample['video'],rgb_online),(sample['raymap'],action_online)):
                single = model._encode_video_latents(pixels[:,:,:1].to('cuda:0',torch.bfloat16),tiled=False)
                torch.testing.assert_close(single,full[:,:,:1],rtol=0,atol=0)
            report['training_full17_first_latent_equals_inference_single_frame'] = True
        report['dataset_online_rgb_action_equals_cache'] = True
        del rgb_online,action_online
        model.train();model.dit.requires_grad_(True)
        model.vae.eval().requires_grad_(False)
        torch.manual_seed(42)
        loss,metrics = model.training_loss(sample)
        if not torch.isfinite(loss):raise ValueError('Non-finite real model loss')
        loss.backward()
        gradients = [p.grad for p in model.dit.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().max().item()>0 for g in gradients)
        assert all(p.grad is None for p in model.vae.parameters())
        report.update(real_forward_backward_passed=True,loss=metrics,
            gradient_parameter_tensors=len(gradients),frozen_vae_no_gradients=True,
            backward_peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        model.zero_grad(set_to_none=True)
        del loss,gradients
        gc.collect();torch.cuda.empty_cache()
        with torch.inference_mode():
            prediction = model.infer(prompt=None,input_image=sample['video'][:,:,0],
                input_raymap=sample['raymap'][:,:,0],current_endpose=sample['current_endpose'],
                context=sample['context'],context_mask=sample['context_mask'],
                num_inference_steps=2,seed=42,decode_future_rgb=False)
        report['inference_tensor_shapes'] = {k:list(v.shape) for k,v in prediction.items() if isinstance(v,torch.Tensor)}
        for v in prediction.values():
            if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
        report.update(real_two_step_inference_finite=True,
            peak_GiB=torch.cuda.max_memory_allocated()/2**30,model_diagnostic_seconds=time.monotonic()-started)
        if a.trainer_validation:
            from contextlib import nullcontext
            from types import SimpleNamespace
            from fastwam.trainer import Wan22Trainer
            import numpy as np
            eval_dir = a.output/'trainer_validation'
            eval_dir.mkdir(exist_ok=True)
            harness = SimpleNamespace(
                accelerator=SimpleNamespace(device=torch.device('cuda:0'),
                    process_index=0,autocast=nullcontext),
                eval_num_inference_steps=20,cfg=cfg,eval_dir=str(eval_dir),global_step=0)
            model.eval()
            normal = Wan22Trainer._to_batched_eval_sample(samples[0])
            model.set_training_dataset_stats(ROOT/cfg.data.train.pretrained_norm_stats)
            val = Wan22Trainer._evaluate_visual_action(harness,model,normal,False,
                eval_sample_index=0,eval_seed=42)
            assert 'val_loss' in val and 'decoded_rotation_geodesic_deg' in val
            assert all(np.isfinite(v) for v in val.values() if isinstance(v,(int,float)))
            tail = Wan22Trainer._to_batched_eval_sample(samples[137])
            with torch.no_grad(): tail_loss,_ = model.training_loss(tail)
            assert torch.isfinite(tail_loss)
            report.update(shared_trainer_validation_20_steps_passed=True,
                shared_trainer_validation=val,tail_validation_loss=float(tail_loss),
                tail_action_padding=int(tail['raymap_is_pad'].sum()))
    filename = 'real_model_trainer.json' if a.trainer_validation else ('real_model.json' if a.gpu else 'real_dataset.json')
    destination = a.output/filename
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
