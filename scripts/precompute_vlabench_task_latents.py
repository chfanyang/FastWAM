"""Task-local resumable cache; multiple independent VAE processes per GPU.

No distributed collectives and no concurrent writers to a shard. A parent lock
prevents two launches writing the same task. Completed shards are atomic and
carry a checksum plus online single-sample verification. Restart repartitions
only unfinished shards; adding another task leaves old task caches untouched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from fastwam.datasets.latent_cache import sha256_file, LatentCacheReader
from fastwam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]


def configure_loader_worker(_worker_id):
    # DataLoader workers must not each spawn a host-sized Arrow/BLAS pool.
    import pyarrow as pa
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    tmp.replace(path)


def encode(vae, sample, device):
    # Same BF16, separate RGB/Rothko encode calls and per-sample VAE loop as training.
    rgb = sample["video"].to(device=device, dtype=torch.bfloat16)
    ray = sample["raymap"].to(device=device, dtype=torch.bfloat16)
    return torch.stack((vae.encode(rgb, device=device, tiled=False),
                        vae.encode(ray, device=device, tiled=False)), 1)


def worker(args):
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    meta = json.loads((args.output / "metadata.json").read_text())
    assigned = json.loads((args.output / "assignments.json").read_text())[args.worker]
    if not assigned:
        return
    dataset = instantiate(OmegaConf.load(args.output / "dataset.yaml"))
    vae = _load_registered_model(meta["vae_path"], "wan_video_vae",
                                 torch_dtype=torch.bfloat16, device=device)
    vae.eval().requires_grad_(False)
    print(f"worker={args.worker} GPU={args.device} shards={len(assigned)} ready", flush=True)
    started = time.monotonic()
    done = 0
    # One loader per process, not one per shard. Its ordered samples span fixed shards.
    indices = [i for s in assigned for i in range(s["start"], s["end"])]
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False,
                        worker_init_fn=configure_loader_worker,
                        **({"prefetch_factor": 1, "multiprocessing_context": "spawn"}
                           if args.num_workers else {}))
    shard_pos, row, array = 0, 0, None
    with torch.inference_mode():
        for sample in loader:
            pair = encode(vae, sample, device)
            if tuple(pair.shape[2:]) != tuple(meta["latent_shape"]) or not torch.isfinite(pair).all():
                raise ValueError("Invalid encoded latent")
            raw = pair.cpu().contiguous().view(torch.uint16).numpy()
            for j in range(len(raw)):
                shard = assigned[shard_pos]
                path = args.output / shard["file"]
                if array is None:
                    array = np.memmap(str(path)+".partial", mode="w+", dtype=np.uint16,
                        shape=(shard["end"]-shard["start"], 2, *meta["latent_shape"]))
                array[row] = raw[j]
                # Check both endpoints of EVERY shard against fresh online encode.
                if row in (0, shard["end"]-shard["start"]-1):
                    one = {k: sample[k][j:j+1] for k in ("video", "raymap")}
                    online = encode(vae, one, device)[0].cpu()
                    if not torch.equal(online, pair[j].cpu()):
                        raise RuntimeError("Online/batch encoding not bit-exact")
                row += 1
                if row == shard["end"]-shard["start"]:
                    array.flush()
                    array._mmap.close()
                    array = None
                    Path(str(path)+".partial").replace(path)
                    write_json(Path(str(path)+".json"), dict(sha256=sha256_file(path),
                        online_bit_exact=True, indices=[shard["start"], shard["end"]-1]))
                    done += row
                    print(f"worker={args.worker} GPU={args.device} completed={shard['file']} samples={done} rate={done/(time.monotonic()-started):.3f}/s", flush=True)
                    shard_pos += 1
                    row = 0
    if array is not None or shard_pos != len(assigned):
        raise RuntimeError("Incomplete shard iteration")


def coordinate(args):
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            f"task={args.task_config}",
            f"data.train.task_names=[{args.task_name}]",
            "data.train.latent_cache_dir=null",
            "data.train.latent_cache_only=false",
            "data.train.include_text_context=false"])
    dataset_cfg = OmegaConf.to_container(cfg.data.train, resolve=True)
    dataset = instantiate(dataset_cfg)
    n = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    vae_path = (ROOT / "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth").resolve()
    shape = [16, 5, 24, 72]
    shards = [dict(index=i//args.shard_size, start=i, end=min(i+args.shard_size,n),
                   file=f"latents-{i//args.shard_size:05d}.bin") for i in range(0,n,args.shard_size)]
    meta = dict(cache_version=1, complete=False, num_samples=n, latent_shape=shape,
        modalities=["rgb", "raymap"], dtype="bfloat16_raw_uint16",
        encoding_torch_dtype="torch.bfloat16", model_variant="wan2.1-t2v-1.3b",
        vae_path=str(vae_path), vae_identity=dict(kind="original_wan21", filename=vae_path.name,
            size_bytes=vae_path.stat().st_size, sha256=sha256_file(vae_path)),
        dataset_contract=dataset.latent_cache_dataset_contract,
        samples_per_shard=args.shard_size, shards=shards, benchmark_only=bool(args.max_samples))
    path = args.output / "metadata.json"
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != v for k,v in meta.items() if k != "complete"):
            raise ValueError("Refusing incompatible cache resume")
    (args.output / "_SUCCESS").unlink(missing_ok=True)
    write_json(path, meta)
    OmegaConf.save(OmegaConf.create(dataset_cfg), args.output / "dataset.yaml")
    pending = []
    for shard in shards:
        file = args.output / shard["file"]
        marker = Path(str(file)+".json")
        size = (shard["end"]-shard["start"]) * 2 * int(np.prod(shape)) * 2
        valid = file.exists() and file.stat().st_size == size and marker.exists()
        if valid:
            record = json.loads(marker.read_text())
            valid = record.get("online_bit_exact") is True and record.get("sha256") == sha256_file(file)
        if not valid:
            pending.append(shard)
    gpus = [int(x) for x in args.gpus.split(",")]
    devices = [gpu for gpu in gpus for _ in range(args.processes_per_gpu)]
    # Adjacent shards preserve episode locality within each worker.
    assignments = [pending[len(pending)*i//len(devices):len(pending)*(i+1)//len(devices)]
                   for i in range(len(devices))]
    write_json(args.output / "assignments.json", assignments)
    print(f"task={args.task_name} samples={n} reused={len(shards)-len(pending)} pending={len(pending)} processes={len(devices)} output={args.output}", flush=True)
    processes, handles = [], []
    try:
        for i, device in enumerate(devices):
            if not assignments[i]:
                continue
            handle = (args.output / f"worker{i:02d}.log").open("a")
            handles.append(handle)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                   "--worker", str(i), "--device", str(device),
                   "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)]
            processes.append(subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            failed = [p for p in processes if p.poll() not in (None,0)]
            if failed:
                raise RuntimeError("Encoding worker failed; see worker logs. Completed shards retained.")
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Encoding worker failed")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for h in handles:
            h.close()
    meta["complete"] = True
    write_json(path, meta)
    (args.output / "_SUCCESS").write_text("complete\n")
    LatentCacheReader(args.output, expected_dataset_contract=dataset.latent_cache_dataset_contract)
    print(f"COMPLETE {n} samples; every shard endpoints verified bit-exact", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default="select_book")
    parser.add_argument("--task-config", default="vlabench_select_book_rothko_3cam192_wan21")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--processes-per-gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--worker", type=int, default=-1)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if min(args.processes_per_gpu,args.batch_size,args.shard_size) < 1 or min(args.num_workers,args.max_samples) < 0:
        parser.error("Invalid sizes")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker >= 0:
        worker(args)
    else:
        with (args.output / ".coordinator.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            coordinate(args)


if __name__ == "__main__":
    main()
