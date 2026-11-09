"""Profile one unchanged LIBERO rollout with process-local timing wrappers."""
import argparse
import functools
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', default='libero_goal')
    parser.add_argument('--task-id', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    import numpy as np
    import torch
    from omegaconf import OmegaConf, open_dict
    from experiments.libero import eval_libero_single as ev

    torch.set_num_threads(2)
    cfg = OmegaConf.load(args.source / 'manager_config.yaml')
    with open_dict(cfg):
        cfg.gpu_id = 7
        cfg.EVALUATION.output_dir = str(args.output.resolve())
        cfg.EVALUATION.task_suite_name = args.suite
        cfg.EVALUATION.task_id = args.task_id
        cfg.EVALUATION.num_trials = 1
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, args.output / 'resolved_profile_config.yaml')
    assert cfg.model.rothko_decode_mode == 'robust_joint'
    assert cfg.model.rothko_decode_anchor_alpha == 0
    assert not ev._record_prediction_videos(cfg)
    assert cfg.EVALUATION.num_inference_steps == 20
    assert cfg.EVALUATION.replan_steps == 8
    assert torch.cuda.device_count() == 1, 'Expose only the requested physical GPU7'
    props = torch.cuda.get_device_properties(0)
    report = dict(source=str(args.source.resolve()), suite=args.suite, task_id=args.task_id,
                  episode=0, physical_gpu=7, gpu_name=props.name,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  dtype=str(cfg.mixed_precision), checkpoint=str(cfg.ckpt),
                  vae=str(cfg.model.vae_safetensors_path), decoder='robust_joint', anchor=0,
                  horizon=16, replan=8, denoising_steps=20,
                  batch_size=1, first_replan_excluded_from_steady_state=True,
                  timing='CUDA-synchronized host wall time; nested timers are included in parent timings',
                  memory='PyTorch allocated/reserved bytes, including resident model/text encoder/VAE',
                  notes=['One task, one episode; no artificial warm-up rollout.',
                         'Inference follows the saved evaluation path, including online prompt encoding.',
                         'Geometric raymap decode excludes VAE latent-to-image decoding.',
                         'No CUDA empty_cache or model offloading added.'],
                  chunks=[], env_action_conversion_ms=[])
    active = {}
    original_instantiate = ev.instantiate

    def synchronized_timer(original, name):
        @functools.wraps(original)
        def wrapped(*a, **kw):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            value = original(*a, **kw)
            torch.cuda.synchronize()
            active.setdefault(name, []).append((time.perf_counter()-begin)*1000)
            return value
        return wrapped

    def instantiate(*a, **kw):
        begin = time.perf_counter()
        obj = original_instantiate(*a, **kw)
        if hasattr(obj, 'raymap_codec') and hasattr(obj, 'infer'):
            report['model_instantiation_seconds'] = time.perf_counter()-begin
            report['load_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            obj.infer = synchronized_timer(obj.infer, 'model_infer_ms')
            obj.raymap_codec.decode = synchronized_timer(obj.raymap_codec.decode, 'raymap_to_pose_ms')
            obj._decode_video_tensor = synchronized_timer(obj._decode_video_tensor, 'vae_decode_ms')
            obj._encode_video_latents = synchronized_timer(obj._encode_video_latents, 'condition_encode_ms')
            obj.encode_prompt = synchronized_timer(obj.encode_prompt, 'text_encode_ms')
        return obj
    ev.instantiate = instantiate
    original_load = ev._load_model_checkpoint
    def load(*a, **kw):
        torch.cuda.synchronize();begin=time.perf_counter()
        result=original_load(*a, **kw)
        torch.cuda.synchronize()
        report['checkpoint_load_seconds']=time.perf_counter()-begin
        report['load_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        return result
    ev._load_model_checkpoint=load
    original_predict=ev._predict_action_chunk
    def predict(*a, **kw):
        active.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline=torch.cuda.memory_allocated()
        begin=time.perf_counter()
        result=original_predict(*a, **kw)
        torch.cuda.synchronize()
        row=dict(replan_index=len(report['chunks']), chunk_wall_ms=(time.perf_counter()-begin)*1000,
                 baseline_allocated_bytes=baseline, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 stages={k:dict(calls=len(v),total_ms=sum(v),calls_ms=list(v)) for k,v in active.items()})
        report['chunks'].append(row)
        with (args.output/'chunks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('PROFILE',json.dumps(row),flush=True)
        return result
    ev._predict_action_chunk=predict
    original_action=ev._absolute_target_to_env_action
    def action(*a, **kw):
        begin=time.perf_counter();value=original_action(*a, **kw)
        report['env_action_conversion_ms'].append((time.perf_counter()-begin)*1000)
        return value
    ev._absolute_target_to_env_action=action

    def stats(values):
        if not values:return dict(count=0)
        x=np.array(values)
        return dict(count=len(values),mean=float(x.mean()),median=float(np.median(x)),
                    p95=float(np.percentile(x,95)),min=float(x.min()),max=float(x.max()))
    try:
        report['episode_result']=ev.eval_single_process.__wrapped__(cfg)
        report['complete']=True
    except BaseException:
        report['error']=traceback.format_exc();report['complete']=False
        raise
    finally:
        chunks=report['chunks'];steady=chunks[1:]
        report['total_process_seconds']=time.perf_counter()-start
        report['summary_ms']={}
        for label, records in [('all_replans',chunks),('steady_replans',steady)]:
            summary={'chunk_wall':stats([r['chunk_wall_ms'] for r in records])}
            for stage in ('model_infer_ms','raymap_to_pose_ms','vae_decode_ms','condition_encode_ms','text_encode_ms'):
                summary[stage]=stats([r['stages'][stage]['total_ms'] for r in records if stage in r['stages']])
            report['summary_ms'][label]=summary
        report['env_action_conversion_summary_ms']=stats(report['env_action_conversion_ms'])
        report['inference_peak_allocated_bytes']=max((r['peak_allocated_bytes'] for r in chunks),default=0)
        report['inference_peak_reserved_bytes']=max((r['peak_reserved_bytes'] for r in chunks),default=0)
        (args.output/'profile.json').write_text(json.dumps(report,indent=2,cls=ev.NumpyEncoder)+'\n')


if __name__=='__main__':main()
