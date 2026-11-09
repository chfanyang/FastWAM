"""Explicit opt-in Goal500 scheduler using only the matched evaluation entry."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-ids',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if len(set(args.gpu_ids))!=len(args.gpu_ids) or min(args.gpu_ids)<0:raise ValueError('Invalid GPU IDs')
    root=Path(__file__).resolve().parents[2]
    commands=[[sys.executable,str(root/'experiments/libero/eval_matched_fastwam.py'),
        'ckpt='+str(args.checkpoint.resolve()),'EVALUATION.output_dir='+str(args.output.resolve()),
        'EVALUATION.task_id='+str(i)] for i in range(10)]
    if not args.execute:
        print(json.dumps(dict(goal_tasks=10,episodes_per_task=50,gpus=args.gpu_ids,commands=commands),indent=2));return
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'plan.json').write_text(json.dumps(dict(commands=commands,gpus=args.gpu_ids),indent=2)+'\n')
    pending=list(range(10));active={};completed=[]
    def stop(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            for gpu in args.gpu_ids:
                if gpu in active or not pending:continue
                task=pending.pop(0);env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                env['PYTHONPATH']=str(root/'src')+':'+str(root)+':'+env.get('PYTHONPATH','')
                with (args.output/f'task{task:02d}_gpu{gpu}.log').open('w') as log:
                    proc=subprocess.Popen(commands[task]+['gpu_id='+str(gpu)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(task,proc,time.monotonic())
            for gpu,(task,proc,begin) in list(active.items()):
                if time.monotonic()-begin>43200:raise TimeoutError(f'Task {task} exceeded 12h')
                if proc.poll() is not None:
                    if proc.returncode:raise RuntimeError(f'Task {task} failed: {proc.returncode}')
                    completed.append(task);del active[gpu]
            time.sleep(1)
        rows=[json.loads(p.read_text()) for p in args.output.glob('libero_goal/gpu*_task*_results.json')]
        assert len(rows)==10 and {r['task_id'] for r in rows}==set(range(10))
        n=sum(r['total_episodes'] for r in rows);s=sum(r['successes'] for r in rows)
        assert n==500
        (args.output/'summary.json').write_text(json.dumps(dict(complete=True,successes=s,total_episodes=n,success_rate=s/n),indent=2)+'\n')
    finally:
        for _,proc,_ in active.values():
            if proc.poll() is None:proc.terminate()
        for _,proc,_ in active.values():
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':main()
