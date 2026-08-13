#!/usr/bin/env python
"""Stratified LIBERO Rothko VAE reconstruction evaluation.

The evaluator uses the training split defined by the VAE fine-tuning config.
For every task in every requested LIBERO suite it samples distinct episodes,
then samples one continuous action window from each episode.  Every VAE
checkpoint is evaluated on the exact same windows.

Only the VAE round trip is measured here:

    ground-truth Rothko -> frozen Wan VAE encoder -> checkpoint decoder

This does not include FastWAM/DiT prediction or simulator rollout error.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import finetune_rothko_vae_decoder as core  # noqa: E402


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "runs/libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/vae/libero_rothko_decoder_all4_h16_bs2_ga8_lr1e-5_ep2.json"
)
DEFAULT_STEPS = (800, 1600, 2400, 3200, 4000, 4800, 5600, 6400)
DEFAULT_OUTPUT_NAME = "vae_reconstruction_train_10episodes_per_task_step0800_to6400.json"


@dataclass
class MetricAccumulator:
    horizon: int
    windows: int = 0
    future_elements: int = 0
    normalized_center_mae_sum: float = 0.0
    normalized_direction_mae_sum: float = 0.0
    normalized_gripper_mae_sum: float = 0.0
    duplicate_mae_sum: float = 0.0
    normalized_abs_sum: float = 0.0
    normalized_squared_sum: float = 0.0
    normalized_elements: int = 0
    position_sum_m: float = 0.0
    rotation_sum_deg: float = 0.0
    gripper_sum: float = 0.0
    position_max_m: float = 0.0
    rotation_max_deg: float = 0.0
    gripper_max: float = 0.0
    position_horizon_sum_m: list[float] = field(default_factory=list)
    rotation_horizon_sum_deg: list[float] = field(default_factory=list)
    gripper_horizon_sum: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.position_horizon_sum_m:
            self.position_horizon_sum_m = [0.0] * self.horizon
            self.rotation_horizon_sum_deg = [0.0] * self.horizon
            self.gripper_horizon_sum = [0.0] * self.horizon

    def add(
        self,
        *,
        center_mae: float,
        direction_mae: float,
        gripper_region_mae: float,
        duplicate_mae: float,
        normalized_abs_sum: float,
        normalized_squared_sum: float,
        normalized_elements: int,
        position_error: torch.Tensor,
        rotation_error: torch.Tensor,
        gripper_error: torch.Tensor,
    ) -> None:
        position = position_error.detach().double().cpu().reshape(-1)
        rotation = rotation_error.detach().double().cpu().reshape(-1)
        gripper = gripper_error.detach().double().cpu().reshape(-1)
        if not (len(position) == len(rotation) == len(gripper) == self.horizon):
            raise ValueError(
                "Expected one error per future action: "
                f"position={len(position)} rotation={len(rotation)} "
                f"gripper={len(gripper)} horizon={self.horizon}"
            )

        self.windows += 1
        self.future_elements += self.horizon
        self.normalized_center_mae_sum += center_mae
        self.normalized_direction_mae_sum += direction_mae
        self.normalized_gripper_mae_sum += gripper_region_mae
        self.duplicate_mae_sum += duplicate_mae
        self.normalized_abs_sum += normalized_abs_sum
        self.normalized_squared_sum += normalized_squared_sum
        self.normalized_elements += normalized_elements
        self.position_sum_m += float(position.sum())
        self.rotation_sum_deg += float(rotation.sum())
        self.gripper_sum += float(gripper.sum())
        self.position_max_m = max(self.position_max_m, float(position.max()))
        self.rotation_max_deg = max(self.rotation_max_deg, float(rotation.max()))
        self.gripper_max = max(self.gripper_max, float(gripper.max()))
        for index in range(self.horizon):
            self.position_horizon_sum_m[index] += float(position[index])
            self.rotation_horizon_sum_deg[index] += float(rotation[index])
            self.gripper_horizon_sum[index] += float(gripper[index])

    def result(self) -> dict[str, Any]:
        if self.windows == 0 or self.future_elements == 0 or self.normalized_elements == 0:
            raise ValueError("Cannot finalize an empty metric accumulator.")
        normalized_mse = self.normalized_squared_sum / self.normalized_elements
        psnr = float("inf") if normalized_mse == 0 else 10.0 * math.log10(4.0 / normalized_mse)
        return {
            "num_windows": self.windows,
            "num_future_poses": self.future_elements,
            "normalized_full_mae": self.normalized_abs_sum / self.normalized_elements,
            "normalized_full_mse": normalized_mse,
            "normalized_full_psnr_db": psnr,
            "normalized_center_mae": self.normalized_center_mae_sum / self.windows,
            "normalized_direction_mae": self.normalized_direction_mae_sum / self.windows,
            "normalized_gripper_mae": self.normalized_gripper_mae_sum / self.windows,
            "duplicate_mae": self.duplicate_mae_sum / self.windows,
            "future_position_mae_m": self.position_sum_m / self.future_elements,
            "future_rotation_mean_deg": self.rotation_sum_deg / self.future_elements,
            "future_gripper_mae": self.gripper_sum / self.future_elements,
            "future_position_max_m": self.position_max_m,
            "future_rotation_max_deg": self.rotation_max_deg,
            "future_gripper_max": self.gripper_max,
            "future_position_mae_by_horizon_m": [
                value / self.windows for value in self.position_horizon_sum_m
            ],
            "future_rotation_mean_by_horizon_deg": [
                value / self.windows for value in self.rotation_horizon_sum_deg
            ],
            "future_gripper_mae_by_horizon": [
                value / self.windows for value in self.gripper_horizon_sum
            ],
        }


def _load_config(path: Path) -> argparse.Namespace:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return argparse.Namespace(**payload)


def _suite_for_episode(episode: core.EpisodeRef) -> str:
    dataset_name = Path(episode.dataset_root).name
    reverse = {directory: suite for suite, directory in core.SUITE_DIRS.items()}
    if dataset_name not in reverse:
        raise ValueError(f"Unknown LIBERO dataset directory: {dataset_name}")
    return reverse[dataset_name]


def _stable_rng(seed: int, suite: str, task: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{suite}\0{task}".encode()).digest()
    return random.Random(seed + int.from_bytes(digest[:8], "big"))


def select_windows(
    training_episodes: Sequence[core.EpisodeRef],
    *,
    episodes_per_task: int,
    horizon: int,
    seed: int,
) -> tuple[list[core.WindowRef], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[core.EpisodeRef]] = defaultdict(list)
    for episode in training_episodes:
        grouped[(_suite_for_episode(episode), episode.task)].append(episode)

    refs: list[core.WindowRef] = []
    records: list[dict[str, Any]] = []
    for (suite, task), episodes in sorted(grouped.items()):
        eligible = [episode for episode in episodes if episode.length >= horizon]
        if len(eligible) < episodes_per_task:
            raise ValueError(
                f"{suite}/{task!r} has only {len(eligible)} eligible training episodes; "
                f"requested {episodes_per_task}."
            )
        rng = _stable_rng(seed, suite, task)
        chosen = rng.sample(eligible, episodes_per_task)
        for episode in sorted(chosen, key=lambda item: item.episode_index):
            valid_starts = episode.length - horizon + 1
            start = rng.randrange(valid_starts)
            refs.append(core.WindowRef(episode=episode, start=start))
            records.append(
                {
                    "suite": suite,
                    "task": task,
                    "episode_index": episode.episode_index,
                    "episode_length": episode.length,
                    "window_start": start,
                    "action_horizon": horizon,
                }
            )
    return refs, records


def _checkpoint_paths(run_dir: Path, steps: Sequence[int]) -> list[tuple[int, Path]]:
    result = []
    for step in steps:
        path = run_dir / f"Wan2.2_VAE_libero_rothko_step{step:06d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append((step, path))
    return result


def _masked_mae_per_window(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = (reconstruction - target).abs()[..., mask]
    return values.reshape(values.shape[0], -1).mean(-1)


@torch.no_grad()
def precompute_latents(
    windows: Sequence[core.ActionWindow],
    stats: core.RothkoNormStats,
    config: argparse.Namespace,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    print(f"Loading frozen base encoder: {config.base_vae}", flush=True)
    vae = core.load_base_vae(config.base_vae, device, torch.float32)
    cached: list[torch.Tensor] = []
    for offset in range(0, len(windows), batch_size):
        group = windows[offset : offset + batch_size]
        targets, _ = core.build_target_batch(group, stats, device, torch.float32, config)
        with core.autocast_context(device, bool(config.bf16)):
            latents = core.vae_encode(vae.model, targets, vae.scale)
        cached.extend(tensor.detach().cpu() for tensor in latents)
        if offset == 0 or min(offset + batch_size, len(windows)) % 40 == 0:
            print(f"Encoded {min(offset + batch_size, len(windows))}/{len(windows)} windows", flush=True)
        del targets, latents
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return cached


@torch.no_grad()
def evaluate_checkpoint(
    *,
    checkpoint: Path,
    windows: Sequence[core.ActionWindow],
    cached_latents: Sequence[torch.Tensor],
    sample_records: Sequence[dict[str, Any]],
    stats: core.RothkoNormStats,
    config: argparse.Namespace,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    vae = core.load_base_vae(str(checkpoint), device, torch.float32)
    vae.eval()
    masks = core.build_loss_masks(config, device)
    overall = MetricAccumulator(config.action_horizon)
    by_suite: dict[str, MetricAccumulator] = {}
    by_task: dict[tuple[str, str], MetricAccumulator] = {}

    for offset in range(0, len(windows), batch_size):
        group = windows[offset : offset + batch_size]
        records = sample_records[offset : offset + batch_size]
        targets, encoded = core.build_target_batch(group, stats, device, torch.float32, config)
        latents = torch.stack(cached_latents[offset : offset + len(group)]).to(device)
        with core.autocast_context(device, bool(config.bf16)):
            reconstruction = vae.model.decode(latents, vae.scale)
        reconstruction = reconstruction.float().clamp(-1, 1)
        target_float = targets.float()
        center = _masked_mae_per_window(reconstruction, target_float, masks["center"])
        direction = _masked_mae_per_window(reconstruction, target_float, masks["direction"])
        gripper_region = _masked_mae_per_window(
            reconstruction, target_float, masks["gripper"]
        )
        difference = reconstruction - target_float
        normalized_abs = difference.abs().reshape(len(group), -1).sum(-1)
        normalized_squared = difference.square().reshape(len(group), -1).sum(-1)
        normalized_elements = difference[0].numel()

        raw = core.denormalize_rothko(reconstruction, stats)
        duplicate = (raw[..., : core.TILE_W] - raw[..., core.TILE_W :]).abs()
        duplicate = duplicate.reshape(len(group), -1).mean(-1)
        current = torch.stack([item.pose[0] for item in encoded])
        pose_prediction = core.decode_pose(raw, current, config)
        pose_target = torch.stack([item.pose for item in encoded]).to(pose_prediction)
        position = (pose_prediction[:, 1:, :3] - pose_target[:, 1:, :3]).norm(dim=-1)
        rotation = core.rotation_angle_degrees(
            core.quaternion_wxyz_to_matrix(pose_prediction[:, 1:, 3:7].float()),
            core.quaternion_wxyz_to_matrix(pose_target[:, 1:, 3:7].float()),
        )
        gripper_prediction = core.read_gripper(reconstruction, config)
        gripper_target = torch.stack([item.gripper for item in encoded]).to(gripper_prediction)
        gripper = (gripper_prediction[:, 1:] - gripper_target[:, 1:]).abs().squeeze(-1)

        for index, record in enumerate(records):
            suite = record["suite"]
            task = record["task"]
            suite_acc = by_suite.setdefault(suite, MetricAccumulator(config.action_horizon))
            task_acc = by_task.setdefault((suite, task), MetricAccumulator(config.action_horizon))
            values = {
                "center_mae": float(center[index]),
                "direction_mae": float(direction[index]),
                "gripper_region_mae": float(gripper_region[index]),
                "duplicate_mae": float(duplicate[index]),
                "normalized_abs_sum": float(normalized_abs[index]),
                "normalized_squared_sum": float(normalized_squared[index]),
                "normalized_elements": normalized_elements,
                "position_error": position[index],
                "rotation_error": rotation[index],
                "gripper_error": gripper[index],
            }
            overall.add(**values)
            suite_acc.add(**values)
            task_acc.add(**values)

        completed = min(offset + batch_size, len(windows))
        if completed % 40 == 0 or completed == len(windows):
            print(f"  decoded {completed}/{len(windows)} windows", flush=True)
        del targets, encoded, latents, reconstruction, target_float, raw

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "duration_seconds": time.time() - started,
        "overall": overall.result(),
        "by_suite": {},
    }
    for suite in sorted(by_suite):
        tasks = {
            task: by_task[(suite, task)].result()
            for current_suite, task in sorted(by_task)
            if current_suite == suite
        }
        result["by_suite"][suite] = {
            **by_suite[suite].result(),
            "by_task": tasks,
        }
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260806)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes_per_task < 1 or args.batch_size < 1:
        raise ValueError("episodes-per-task and batch-size must be positive")
    config_path = args.config.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else run_dir / DEFAULT_OUTPUT_NAME
    )
    config = _load_config(config_path)
    checkpoints = _checkpoint_paths(run_dir, args.steps)
    training_episodes, held_out_episodes = core.discover_episodes(config)
    refs, sample_records = select_windows(
        training_episodes,
        episodes_per_task=args.episodes_per_task,
        horizon=config.action_horizon,
        seed=args.sample_seed,
    )
    task_count = len({(record["suite"], record["task"]) for record in sample_records})
    suite_count = len({record["suite"] for record in sample_records})
    print(
        f"Selected {len(refs)} windows from {len(refs)} distinct training episodes: "
        f"suites={suite_count}, tasks={task_count}, episodes/task={args.episodes_per_task}",
        flush=True,
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Wan VAE round-trip reconstruction on stratified LIBERO training episodes; "
            "one deterministic continuous 17-frame window per sampled episode."
        ),
        "config": {
            "source_config": str(config_path),
            "data_root": str(Path(config.data_root).expanduser().resolve()),
            "suites": list(config.suites),
            "training_split_seed": config.split_seed,
            "held_out_episodes_per_task": config.eval_episodes_per_task,
            "available_training_episodes": len(training_episodes),
            "available_held_out_episodes": len(held_out_episodes),
            "sample_seed": args.sample_seed,
            "episodes_per_task": args.episodes_per_task,
            "windows_per_episode": 1,
            "num_suites": suite_count,
            "num_tasks": task_count,
            "num_windows": len(refs),
            "action_horizon": config.action_horizon,
            "pixel_frames": config.action_horizon + 1,
            "norm_stats_path": str(Path(config.norm_stats_path).expanduser().resolve()),
            "base_encoder": str(Path(config.base_vae).expanduser().resolve()),
            "bf16_autocast": bool(config.bf16),
            "batch_size": args.batch_size,
            "device": args.device,
            "checkpoint_steps": list(args.steps),
        },
        "samples": sample_records,
        "results": {},
    }
    _atomic_json_dump(payload, output_json)
    print(f"Selection manifest written to {output_json}", flush=True)
    if args.dry_run:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan VAE reconstruction evaluation.")
    device = torch.device(args.device)
    store = core.EpisodeStore(config.episode_cache_size)
    windows = core.materialize_windows(store, refs, config.action_horizon)
    stats = core.load_norm_stats(config.norm_stats_path, config)
    cached_latents = precompute_latents(
        windows, stats, config, device, args.batch_size
    )

    for step, checkpoint in checkpoints:
        print(f"Evaluating step {step}: {checkpoint}", flush=True)
        result = evaluate_checkpoint(
            checkpoint=checkpoint,
            windows=windows,
            cached_latents=cached_latents,
            sample_records=sample_records,
            stats=stats,
            config=config,
            device=device,
            batch_size=args.batch_size,
        )
        result["step"] = step
        payload["results"][str(step)] = result
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json_dump(payload, output_json)
        metrics = result["overall"]
        print(
            f"step={step} position={metrics['future_position_mae_m'] * 1000:.4f}mm "
            f"rotation={metrics['future_rotation_mean_deg']:.5f}deg "
            f"PSNR={metrics['normalized_full_psnr_db']:.3f}dB",
            flush=True,
        )
    print(f"Completed all checkpoints. Results: {output_json}", flush=True)


if __name__ == "__main__":
    main()
