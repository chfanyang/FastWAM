#!/usr/bin/env python
"""Finetune the Wan2.2 VAE decoder for the 17-frame Rothko representation.

Each training window is constructed as:

    frame 0:    observation.state.endpose[start] and current grippers
    frames 1-16: action.endpose[start:start + 16] and future grippers

For each arm, a 192x160 Rothko tile stores relative translation in its central
rectangle, rotated canonical ray directions in its periphery, and ``2*g-1`` in
the outer border.  The left/right 192x320 image is duplicated vertically to
produce the Wan-compatible 384x320 input.  The pretrained encoder is frozen;
only ``conv2`` and the decoder are optimized.

Run from the repository root:

    python -m research.rothko.finetune_rothko_vae_decoder --max-steps 2 \
        --eval-every 2 --eval-windows 4 --fresh

    torchrun --standalone --nproc_per_node=4 \
        -m research.rothko.finetune_rothko_vae_decoder \
        --output-dir experiments/rothko_vae_decoder_finetune/q99p95_h16
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import random
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, SequentialLR
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.rothko import rothko_raymap_vae_sim_replay as rothko  # noqa: E402


DATASET_ROOT = str(REPO_ROOT / "robotwin2.0-fastwam/robotwin2.0")
DEFAULT_NORM_STATS_PATH = REPO_ROOT / "rothko_region_symmetric_q99p95_h16_384x320.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/rothko_vae_decoder_finetune/q99p95_h16"

EPISODES_PER_TASK = 550
NUM_TASKS = 50
DEFAULT_EVAL_EPISODES_PER_TASK = 10

ACTION_HORIZON = 16
PIXEL_FRAMES = ACTION_HORIZON + 1
ARM_H = 192
ARM_W = 160
FULL_H = 384
FULL_W = 320
FOCAL = 0.2
CENTER_SCALE = 1.0
DIR_SCALE = 1.0
CENTER_FRAC = 0.5
BOUNDARY_MARGIN = 8
OUTER_MARGIN = 8

_LOG_FILE: Optional[Any] = None
_CHECKPOINT_STATE: Dict[str, Any] = {}


@dataclass
class ActionWindow:
    episode_index: int
    start: int
    pose: np.ndarray  # [17,14]: current state pose followed by 16 action poses
    gripper: np.ndarray  # [17,2]: current state grippers followed by action grippers


@dataclass
class EncodedWindow:
    video: Tensor  # normalized [3,17,384,320]
    raw: Tensor  # raw Rothko [17,3,384,320]
    left: rothko._ArmPoseInfo
    right: rothko._ArmPoseInfo


def _init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def barrier(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()


def setup_logging(output_dir: str, log_file: Optional[str], rank: int) -> str:
    if rank != 0:
        return ""
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    path = log_file or os.path.join(output_dir, "train.log")
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.close()
    _LOG_FILE = open(path, "a", encoding="utf-8", buffering=1)
    return path


def log(message: str, rank: int = 0) -> None:
    if rank != 0:
        return
    print(message, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(message + "\n")
        _LOG_FILE.flush()


def load_dataset_info(dataset_root: str) -> Dict[str, Any]:
    with open(os.path.join(dataset_root, "meta", "info.json"), encoding="utf-8") as handle:
        return json.load(handle)


def episode_parquet_path(
    dataset_root: str,
    info: Dict[str, Any],
    episode_index: int,
) -> str:
    chunk = episode_index // int(info["chunks_size"])
    relative = info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    return os.path.join(dataset_root, relative)


def _stack_column(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float32) for value in series.values])


def load_episode_arrays(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = [
        "observation.state",
        "action",
        "observation.state.endpose",
        "action.endpose",
    ]
    frame = pd.read_parquet(path, columns=columns)
    state = _stack_column(frame["observation.state"])
    action = _stack_column(frame["action"])
    state_pose = _stack_column(frame["observation.state.endpose"])
    action_pose = _stack_column(frame["action.endpose"])
    lengths = {len(state), len(action), len(state_pose), len(action_pose)}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent episode column lengths in {path}")
    if state.shape[1] < 14 or action.shape[1] < 14:
        raise ValueError(f"Expected at least 14 state/action dimensions in {path}")
    return state, action, state_pose, action_pose


def eval_episode_indices(
    total_episodes: int,
    eval_episodes_per_task: int,
) -> Set[int]:
    return {
        task_id * EPISODES_PER_TASK + seed
        for task_id in range(NUM_TASKS)
        for seed in range(eval_episodes_per_task)
        if task_id * EPISODES_PER_TASK + seed < total_episodes
    }


def sample_random_windows(
    dataset_root: str,
    episode_pool: Sequence[int],
    num_windows: int,
    horizon: int,
    seed: int,
    info: Dict[str, Any],
) -> List[ActionWindow]:
    """Sample valid current-state + future-action windows from episode parquets."""
    rng = random.Random(seed)
    pool = list(episode_pool)
    if not pool:
        raise ValueError("episode_pool is empty")

    windows: List[ActionWindow] = []
    attempts = 0
    max_attempts = max(100, num_windows * 20)
    while len(windows) < num_windows and attempts < max_attempts:
        attempts += 1
        episode_index = rng.choice(pool)
        path = episode_parquet_path(dataset_root, info, episode_index)
        if not os.path.isfile(path):
            continue
        try:
            state, action, state_pose, action_pose = load_episode_arrays(path)
        except Exception:
            continue
        if len(action_pose) < horizon:
            continue
        start = rng.randrange(len(action_pose) - horizon + 1)
        stop = start + horizon
        pose = np.concatenate(
            [state_pose[start : start + 1], action_pose[start:stop]],
            axis=0,
        )
        gripper = np.concatenate(
            [state[start : start + 1, [6, 13]], action[start:stop, [6, 13]]],
            axis=0,
        )
        windows.append(
            ActionWindow(
                episode_index=episode_index,
                start=start,
                pose=pose.astype(np.float32, copy=False),
                gripper=gripper.astype(np.float32, copy=False),
            )
        )
    if len(windows) != num_windows:
        raise RuntimeError(
            f"Collected {len(windows)} Rothko windows after {attempts} attempts; "
            f"need {num_windows}"
        )
    return windows


def outer_border_mask(height: int, width: int, margin: int, device=None) -> Tensor:
    if margin <= 0 or 2 * margin >= min(height, width):
        raise ValueError(f"Invalid outer margin {margin} for {height}x{width}")
    mask = torch.zeros(height, width, dtype=torch.bool, device=device)
    mask[:margin] = True
    mask[-margin:] = True
    mask[:, :margin] = True
    mask[:, -margin:] = True
    return mask


def write_gripper_border(tile: Tensor, gripper: Tensor, margin: int) -> Tensor:
    """Write ``2*g-1`` to every channel of an arm tile's outer border."""
    if tile.ndim != 4 or gripper.shape != (tile.shape[0],):
        raise ValueError("Expected tile [T,3,H,W] and gripper [T]")
    result = tile.clone()
    mask = outer_border_mask(tile.shape[-2], tile.shape[-1], margin, tile.device)
    code = (2.0 * gripper - 1.0).to(dtype=tile.dtype)
    pixels = result.permute(0, 2, 3, 1)
    pixels[:, mask, :] = code[:, None, None]
    return result


def decode_gripper_border(tile: Tensor, margin: int) -> Tensor:
    mask = outer_border_mask(tile.shape[-2], tile.shape[-1], margin, tile.device)
    values = tile.permute(0, 2, 3, 1)[:, mask, :].reshape(tile.shape[0], -1)
    code = values.median(dim=1).values
    return ((code + 1.0) * 0.5).clamp(0.0, 1.0)


def load_and_validate_norm_stats(
    path: str,
    *,
    horizon: int,
    focal: float,
    center_scale: float,
    dir_scale: float,
    center_frac: float,
    boundary_margin: int,
    outer_margin: int,
) -> Tuple[rothko.RothkoNormStats, Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Rothko norm stats not found: {path}")
    stats, metadata = rothko.load_rothko_norm_stats(path)
    expected_shape = (1, 3, FULL_H, FULL_W)
    if tuple(stats.lo.shape) != expected_shape or tuple(stats.hi.shape) != expected_shape:
        raise ValueError(
            f"Expected norm stats shape {expected_shape}, got "
            f"lo={tuple(stats.lo.shape)} hi={tuple(stats.hi.shape)}"
        )
    expected = {
        "representation": "rothko",
        "encoding": "state_frame0_plus_future_action_endpose",
        "quaternion_order": "wxyz",
        "action_horizon": horizon,
        "pixel_frames": horizon + 1,
        "focal": focal,
        "center_scale": center_scale,
        "dir_scale": dir_scale,
        "center_frac": center_frac,
        "boundary_margin": boundary_margin,
        "outer_margin": outer_margin,
        "duplicate_vertical": True,
        "gripper_encoding": "normalized_outer_border_2g_minus_1",
        "active_shape": [FULL_H, FULL_W],
    }
    mismatches = []
    for key, value in expected.items():
        if key not in metadata:
            mismatches.append(f"{key}: missing")
        elif metadata[key] != value:
            mismatches.append(f"{key}: stats={metadata[key]!r}, expected={value!r}")
    if mismatches:
        raise ValueError("Norm stats metadata mismatch: " + "; ".join(mismatches))
    return stats, dict(metadata)


def encode_window(
    window: ActionWindow,
    stats: rothko.RothkoNormStats,
    device: torch.device,
    dtype: torch.dtype,
    *,
    focal: float,
    center_scale: float,
    dir_scale: float,
    center_frac: float,
    outer_margin: int,
) -> EncodedWindow:
    pose = torch.from_numpy(window.pose).to(device=device, dtype=torch.float32)
    gripper = torch.from_numpy(window.gripper).to(device=device, dtype=torch.float32)
    if len(pose) != PIXEL_FRAMES:
        raise ValueError(f"Expected {PIXEL_FRAMES} frames, got {len(pose)}")

    left_map, l_pr, l_rr, l_pa, l_ra, l_p0, l_r0 = rothko.pose7_to_rothko(
        pose[:, :7],
        focal=focal,
        H=ARM_H,
        W=ARM_W,
        center_scale=center_scale,
        dir_scale=dir_scale,
        center_frac=center_frac,
    )
    right_map, r_pr, r_rr, r_pa, r_ra, r_p0, r_r0 = rothko.pose7_to_rothko(
        pose[:, 7:14],
        focal=focal,
        H=ARM_H,
        W=ARM_W,
        center_scale=center_scale,
        dir_scale=dir_scale,
        center_frac=center_frac,
    )
    left = rothko._ArmPoseInfo(l_pr, l_rr, l_pa, l_ra, l_p0, l_r0)
    right = rothko._ArmPoseInfo(r_pr, r_rr, r_pa, r_ra, r_p0, r_r0)

    left_map = write_gripper_border(left_map, gripper[:, 0], outer_margin)
    right_map = write_gripper_border(right_map, gripper[:, 1], outer_margin)
    top = torch.cat([left_map, right_map], dim=-1)
    raw = torch.cat([top, top], dim=-2)
    if tuple(raw.shape) != (PIXEL_FRAMES, 3, FULL_H, FULL_W):
        raise RuntimeError(f"Unexpected Rothko tensor shape: {tuple(raw.shape)}")

    stats_dev = rothko._stats_to_device(stats, device, torch.float32)
    normalized = rothko.normalize_rothko(raw, stats_dev)
    video = normalized.permute(1, 0, 2, 3).contiguous().to(dtype=dtype)
    return EncodedWindow(video=video, raw=raw.to(dtype=dtype), left=left, right=right)


def build_target_batch(
    windows: Sequence[ActionWindow],
    stats: rothko.RothkoNormStats,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Tuple[Tensor, List[EncodedWindow]]:
    encoded = [
        encode_window(
            window,
            stats,
            device,
            dtype,
            focal=args.focal,
            center_scale=args.center_scale,
            dir_scale=args.dir_scale,
            center_frac=args.center_frac,
            outer_margin=args.outer_margin,
        )
        for window in windows
    ]
    return torch.stack([item.video for item in encoded], dim=0), encoded


def build_full_region_masks(
    center_frac: float,
    outer_margin: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    center, _, _ = rothko.rothko_region_masks(
        ARM_H,
        ARM_W,
        center_frac=center_frac,
        boundary_margin=0,
        outer_margin=0,
        device=device,
    )
    border = outer_border_mask(ARM_H, ARM_W, outer_margin, device)
    center = center & ~border
    direction = ~center & ~border

    def combine(tile_mask: Tensor) -> Tensor:
        top = torch.cat([tile_mask, tile_mask], dim=-1)
        return torch.cat([top, top], dim=-2)

    return {
        "center": combine(center),
        "direction": combine(direction),
        "gripper": combine(border),
    }


def masked_l1(recon: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    values = (recon - target).abs()
    return values[..., mask].mean()


def compute_recon_loss(
    recon: Tensor,
    target: Tensor,
    masks: Dict[str, Tensor],
    center_weight: float,
    direction_weight: float,
    gripper_weight: float,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    losses = {
        name: masked_l1(recon, target, mask)
        for name, mask in masks.items()
    }
    total = (
        center_weight * losses["center"]
        + direction_weight * losses["direction"]
        + gripper_weight * losses["gripper"]
    )
    return total, losses


class VaeDecodeWrapper(nn.Module):
    """Ensure DDP synchronizes decoder gradients through ``forward``."""

    def __init__(self, vae_model: nn.Module):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents: Tensor, scale) -> Tensor:
        return self.vae_model.decode(latents, scale).clamp(-1.0, 1.0)


def prepare_vae_for_decoder_finetune(vae) -> List[nn.Parameter]:
    vae.model.requires_grad_(False)
    for module in (vae.model.encoder, vae.model.conv1):
        module.eval()
        module.requires_grad_(False)
    for module in (vae.model.decoder, vae.model.conv2):
        module.train()
        module.requires_grad_(True)
    return [parameter for parameter in vae.model.parameters() if parameter.requires_grad]


@torch.no_grad()
def vae_encode_batch(vae_model, videos: Tensor, scale) -> Tensor:
    return vae_model.encode(videos, scale)


def resolve_warmup_steps(
    warmup_steps: Optional[int],
    warmup_ratio: float,
    max_steps: int,
) -> int:
    if warmup_steps is not None:
        return min(max(int(warmup_steps), 0), max(max_steps - 1, 0))
    return min(max(int(max_steps * warmup_ratio), 0), max(max_steps - 1, 0))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    scheduler_type: str,
) -> Optional[LRScheduler]:
    if scheduler_type == "constant":
        return None
    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported LR scheduler: {scheduler_type}")
    remaining = max(max_steps - warmup_steps, 1)
    base_lr = float(optimizer.param_groups[0]["lr"])
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=remaining,
        eta_min=base_lr * min_lr_ratio,
    )
    if warmup_steps <= 0:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def save_checkpoint(
    path: str,
    vae_model,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LRScheduler],
    step: int,
    metadata: Dict[str, Any],
    *,
    include_optimizer: bool = True,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "step": step,
        "decoder": vae_model.decoder.state_dict(),
        "conv2": vae_model.conv2.state_dict(),
        "optimizer": (
            optimizer.state_dict()
            if include_optimizer and optimizer is not None
            else None
        ),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "weights_only": not include_optimizer,
        "metadata": metadata,
    }
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str,
    vae_model,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LRScheduler],
) -> Tuple[int, Dict[str, bool]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vae_model.decoder.load_state_dict(payload["decoder"])
    vae_model.conv2.load_state_dict(payload["conv2"])
    step = int(payload.get("step", 0))
    result = {"optimizer": False, "scheduler": False}
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
        result["optimizer"] = True
    if scheduler is not None:
        if payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
            result["scheduler"] = True
        else:
            for _ in range(step):
                scheduler.step()
    return step, result


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    latest = os.path.join(output_dir, "checkpoint_latest.pt")
    if os.path.isfile(latest):
        return latest
    candidates = []
    for path in glob.glob(os.path.join(output_dir, "checkpoint_step*.pt")):
        stem = Path(path).stem
        token = stem.removeprefix("checkpoint_step")
        if token.isdigit():
            candidates.append((int(token), path))
    return max(candidates, default=(-1, None))[1]


def resolve_resume_path(args: argparse.Namespace) -> Optional[str]:
    if args.fresh:
        return None
    if args.resume:
        if args.resume.lower() in {"latest", "auto"}:
            result = find_latest_checkpoint(args.output_dir)
            if result is None:
                raise FileNotFoundError(f"No checkpoint found in {args.output_dir}")
            return result
        candidates = [args.resume]
        if not os.path.isabs(args.resume):
            candidates.insert(0, os.path.join(args.output_dir, args.resume))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    return find_latest_checkpoint(args.output_dir) if args.auto_resume else None


def install_signal_handlers() -> None:
    def handler(signum: int, _frame: Any) -> None:
        state = _CHECKPOINT_STATE
        if state and state.get("rank") == 0:
            log(f"Received signal {signum}; saving emergency checkpoint...")
            try:
                # A sharded optimizer needs a collective consolidation, which is
                # unsafe inside an asynchronous signal handler. Save recoverable
                # model weights and scheduler state; optimizer state is omitted.
                emergency_optimizer = (
                    None if state.get("optimizer_is_sharded") else state["optimizer"]
                )
                save_checkpoint(
                    state["path"],
                    state["vae_model"],
                    emergency_optimizer,
                    state["scheduler"],
                    state["step"],
                    state["metadata"],
                    include_optimizer=emergency_optimizer is not None,
                )
                log(f"Emergency checkpoint saved to {state['path']}")
            except Exception:
                log(f"Emergency checkpoint failed:\n{traceback.format_exc()}")
        if state.get("is_distributed") and dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def average_vertical_duplicates(raw: Tensor) -> Tensor:
    if raw.shape[-2:] != (FULL_H, FULL_W):
        raise ValueError(f"Expected Rothko spatial shape {(FULL_H, FULL_W)}")
    return 0.5 * (raw[..., :ARM_H, :] + raw[..., ARM_H:, :])


def rotation_angle_degrees(pred_rotation: Tensor, true_rotation: Tensor) -> Tensor:
    relative = pred_rotation.transpose(-1, -2) @ true_rotation
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cosine))


@torch.no_grad()
def evaluate(
    vae_model,
    scale,
    dataset_root: str,
    episode_pool: Sequence[int],
    stats: rothko.RothkoNormStats,
    info: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Dict[str, Any]:
    vae_model.decoder.eval()
    vae_model.conv2.eval()
    windows = sample_random_windows(
        dataset_root,
        episode_pool,
        args.eval_windows,
        args.action_horizon,
        seed,
        info,
    )
    masks = build_full_region_masks(args.center_frac, args.outer_margin, device)
    stats_dev = rothko._stats_to_device(stats, device, torch.float32)

    totals: Dict[str, float] = {
        "normalized_center_mae": 0.0,
        "normalized_direction_mae": 0.0,
        "normalized_gripper_border_mae": 0.0,
        "duplicate_mae": 0.0,
        "future_position_mae_m": 0.0,
        "future_position_max_m": 0.0,
        "future_rotation_mean_deg": 0.0,
        "future_rotation_max_deg": 0.0,
        "future_gripper_mae": 0.0,
        "future_gripper_max": 0.0,
    }

    for offset in range(0, len(windows), args.eval_batch_size):
        group = windows[offset : offset + args.eval_batch_size]
        targets, encoded = build_target_batch(group, stats, device, dtype, args)
        latents = vae_encode_batch(vae_model, targets, scale)
        recon = vae_model.decode(latents, scale).clamp(-1.0, 1.0)

        for index, (window, item) in enumerate(zip(group, encoded)):
            recon_i = recon[index]
            target_i = targets[index]
            for region, key in (
                ("center", "normalized_center_mae"),
                ("direction", "normalized_direction_mae"),
                ("gripper", "normalized_gripper_border_mae"),
            ):
                totals[key] += float(masked_l1(recon_i, target_i, masks[region]).item())

            recon_t = recon_i.float().permute(1, 0, 2, 3)
            raw_recon = rothko.denormalize_rothko(recon_t, stats_dev)
            totals["duplicate_mae"] += float(
                (raw_recon[:, :, :ARM_H] - raw_recon[:, :, ARM_H:]).abs().mean().item()
            )
            top = average_vertical_duplicates(raw_recon)
            left_tile = top[..., :ARM_W]
            right_tile = top[..., ARM_W:]
            left_pose = rothko.rothko_arm_to_pose7(
                left_tile,
                item.left.pos_0,
                item.left.rot_0,
                center_scale=args.center_scale,
                dir_scale=args.dir_scale,
                center_frac=args.center_frac,
                boundary_margin=args.boundary_margin,
            )
            right_pose = rothko.rothko_arm_to_pose7(
                right_tile,
                item.right.pos_0,
                item.right.rot_0,
                center_scale=args.center_scale,
                dir_scale=args.dir_scale,
                center_frac=args.center_frac,
                boundary_margin=args.boundary_margin,
            )
            pose_pred = torch.cat([left_pose, right_pose], dim=-1)
            pose_true = torch.from_numpy(window.pose).to(device=device, dtype=torch.float32)

            pos_error = torch.stack(
                [
                    (pose_pred[1:, :3] - pose_true[1:, :3]).norm(dim=-1),
                    (pose_pred[1:, 7:10] - pose_true[1:, 7:10]).norm(dim=-1),
                ],
                dim=-1,
            )
            left_rot_error = rotation_angle_degrees(
                quaternion_to_matrix(pose_pred[1:, 3:7].float()),
                quaternion_to_matrix(pose_true[1:, 3:7]),
            )
            right_rot_error = rotation_angle_degrees(
                quaternion_to_matrix(pose_pred[1:, 10:14].float()),
                quaternion_to_matrix(pose_true[1:, 10:14]),
            )
            rot_error = torch.stack([left_rot_error, right_rot_error], dim=-1)

            gripper_pred = torch.stack(
                [
                    decode_gripper_border(left_tile, args.outer_margin),
                    decode_gripper_border(right_tile, args.outer_margin),
                ],
                dim=-1,
            )
            gripper_true = torch.from_numpy(window.gripper).to(
                device=device,
                dtype=torch.float32,
            )
            grip_error = (gripper_pred[1:] - gripper_true[1:]).abs()

            totals["future_position_mae_m"] += float(pos_error.mean().item())
            totals["future_position_max_m"] = max(
                totals["future_position_max_m"],
                float(pos_error.max().item()),
            )
            totals["future_rotation_mean_deg"] += float(rot_error.mean().item())
            totals["future_rotation_max_deg"] = max(
                totals["future_rotation_max_deg"],
                float(rot_error.max().item()),
            )
            totals["future_gripper_mae"] += float(grip_error.mean().item())
            totals["future_gripper_max"] = max(
                totals["future_gripper_max"],
                float(grip_error.max().item()),
            )

        del targets, latents, recon, encoded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    count = len(windows)
    max_keys = {
        "future_position_max_m",
        "future_rotation_max_deg",
        "future_gripper_max",
    }
    result = {
        key: value if key in max_keys else value / count
        for key, value in totals.items()
    }
    result["num_windows"] = count
    result["encoding"] = "state_frame0_plus_future_action_endpose"
    vae_model.decoder.train()
    vae_model.conv2.train()
    return result


def train(args: argparse.Namespace) -> None:
    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if args.action_horizon != ACTION_HORIZON:
        raise ValueError(
            f"This representation and norm checkpoint require horizon {ACTION_HORIZON}; "
            f"got {args.action_horizon}"
        )
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    barrier(is_distributed)

    info = load_dataset_info(args.dataset_root)
    total_episodes = int(info["total_episodes"])
    eval_set = eval_episode_indices(total_episodes, args.eval_episodes_per_task)
    eval_episodes = sorted(eval_set)
    train_episodes = [index for index in range(total_episodes) if index not in eval_set]
    log(
        f"Dataset: {total_episodes} episodes; train={len(train_episodes)} "
        f"eval={len(eval_episodes)} ({args.eval_episodes_per_task}/task)",
        rank,
    )
    if is_distributed:
        log(f"Distributed: world_size={world_size}", rank)

    norm_path = args.norm_stats_path
    if not os.path.isabs(norm_path):
        norm_path = str(REPO_ROOT / norm_path)
    stats, stats_metadata = load_and_validate_norm_stats(
        norm_path,
        horizon=args.action_horizon,
        focal=args.focal,
        center_scale=args.center_scale,
        dir_scale=args.dir_scale,
        center_frac=args.center_frac,
        boundary_margin=args.boundary_margin,
        outer_margin=args.outer_margin,
    )
    log(
        f"Norm stats: {norm_path}; translation bounds="
        f"{stats_metadata.get('translation_abs_bounds_xyz_m')}",
        rank,
    )

    vae = rothko.load_wan22_vae(str(device), dtype)
    vae_model = vae.model
    trainable = prepare_vae_for_decoder_finetune(vae)
    wrapper = VaeDecodeWrapper(vae_model)
    ddp_wrapper: Optional[DDP] = None
    if is_distributed:
        ddp_wrapper = DDP(
            wrapper,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        decode_fn: nn.Module = ddp_wrapper
    else:
        decode_fn = wrapper
    log(f"Trainable conv2+decoder params: {sum(p.numel() for p in trainable):,}", rank)

    optimizer_is_sharded = is_distributed and args.zero_optimizer
    if optimizer_is_sharded:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optimizer = ZeroRedundancyOptimizer(
            trainable,
            optimizer_class=torch.optim.AdamW,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        log("Optimizer: ZeroRedundancyOptimizer(AdamW), state sharded across ranks", rank)
    else:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        log("Optimizer: AdamW with local optimizer state", rank)
    warmup_steps = resolve_warmup_steps(
        args.warmup_steps,
        args.warmup_ratio,
        args.max_steps,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        args.max_steps,
        warmup_steps,
        args.min_lr_ratio,
        args.lr_scheduler,
    )

    step = 0
    resume_path = resolve_resume_path(args)
    if resume_path:
        step, loaded = load_checkpoint(resume_path, vae_model, optimizer, scheduler)
        log(
            f"Resumed {resume_path} at step {step} "
            f"(optimizer={loaded['optimizer']} scheduler={loaded['scheduler']})",
            rank,
        )
    elif args.auto_resume and not args.fresh:
        log("No checkpoint found; starting from pretrained Wan2.2 VAE.", rank)
    barrier(is_distributed)

    metadata = {
        "representation": "rothko",
        "encoding": "state_frame0_plus_future_action_endpose",
        "quaternion_order": "wxyz",
        "action_horizon": args.action_horizon,
        "pixel_frames": PIXEL_FRAMES,
        "image_height": FULL_H,
        "image_width": FULL_W,
        "arm_height": ARM_H,
        "arm_width": ARM_W,
        "duplicate_vertical": True,
        "focal": args.focal,
        "center_scale": args.center_scale,
        "dir_scale": args.dir_scale,
        "center_frac": args.center_frac,
        "boundary_margin": args.boundary_margin,
        "outer_margin": args.outer_margin,
        "gripper_encoding": "normalized_outer_border_2g_minus_1",
        "norm_stats_path": norm_path,
        "center_loss_weight": args.center_loss_weight,
        "direction_loss_weight": args.direction_loss_weight,
        "gripper_loss_weight": args.gripper_loss_weight,
        "batch_size_per_gpu": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "world_size": world_size,
        "zero_optimizer": optimizer_is_sharded,
        "eval_seed": args.eval_seed,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": warmup_steps,
        "norm_stats_metadata": stats_metadata,
    }
    if rank == 0:
        with open(
            os.path.join(args.output_dir, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2)

    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    _CHECKPOINT_STATE.clear()
    _CHECKPOINT_STATE.update(
        rank=rank,
        is_distributed=is_distributed,
        path=latest_path,
        vae_model=vae_model,
        optimizer=optimizer,
        optimizer_is_sharded=optimizer_is_sharded,
        scheduler=scheduler,
        step=step,
        metadata=metadata,
    )
    if rank == 0:
        install_signal_handlers()

    masks = build_full_region_masks(args.center_frac, args.outer_margin, device)
    grad_accum = max(1, args.grad_accum_steps)
    log(
        f"Effective batch={args.batch_size * world_size * grad_accum}; "
        f"loss weights center={args.center_loss_weight} "
        f"direction={args.direction_loss_weight} gripper={args.gripper_loss_weight}",
        rank,
    )
    progress = None
    if rank == 0 and not args.no_progress_bar:
        progress = tqdm(
            total=args.max_steps,
            initial=step,
            desc="Rothko decoder",
            unit="step",
            dynamic_ncols=True,
            file=sys.stdout,
        )

    running = {"loss": 0.0, "center": 0.0, "direction": 0.0, "gripper": 0.0}
    log_count = 0
    start_time = time.time()
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_values = {key: 0.0 for key in running}

        for accumulation_index in range(grad_accum):
            micro_step = step * grad_accum + accumulation_index
            sample_seed = args.seed + micro_step * world_size + rank
            windows = sample_random_windows(
                args.dataset_root,
                train_episodes,
                args.batch_size,
                args.action_horizon,
                sample_seed,
                info,
            )
            targets, encoded_windows = build_target_batch(
                windows, stats, device, dtype, args,
            )
            # Training only needs the stacked normalized targets. Release the
            # auxiliary raw maps and pose metadata before the decoder forward.
            del encoded_windows
            with torch.no_grad():
                latents = vae_encode_batch(vae_model, targets, vae.scale)

            sync_context = contextlib.nullcontext()
            if is_distributed and accumulation_index < grad_accum - 1:
                sync_context = ddp_wrapper.no_sync()
            with sync_context:
                reconstruction = decode_fn(latents, vae.scale)
                loss, components = compute_recon_loss(
                    reconstruction,
                    targets,
                    masks,
                    args.center_loss_weight,
                    args.direction_loss_weight,
                    args.gripper_loss_weight,
                )
                (loss / grad_accum).backward()

            step_values["loss"] += float(loss.item()) / grad_accum
            for key in ("center", "direction", "gripper"):
                step_values[key] += float(components[key].item()) / grad_accum
            del windows, targets, latents, reconstruction, loss, components
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        # Gradients are not needed after the update. Releasing them here keeps
        # checkpointing and rank-0 evaluation below the training memory peak.
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        step += 1
        _CHECKPOINT_STATE["step"] = step
        for key in running:
            running[key] += step_values[key]
        log_count += 1
        if progress is not None:
            progress.update(1)

        if step % args.log_every == 0:
            averages = {key: value / log_count for key, value in running.items()}
            lr = float(optimizer.param_groups[0]["lr"])
            log(
                f"step {step}/{args.max_steps} loss={averages['loss']:.6f} "
                f"center={averages['center']:.6f} direction={averages['direction']:.6f} "
                f"gripper={averages['gripper']:.6f} lr={lr:.2e} "
                f"elapsed={time.time() - start_time:.1f}s",
                rank,
            )
            if progress is not None:
                progress.set_postfix(loss=f"{averages['loss']:.4f}", lr=f"{lr:.2e}")
            running = {key: 0.0 for key in running}
            log_count = 0

        if step % args.save_every == 0 or step == args.max_steps:
            if optimizer_is_sharded:
                # Collective: every rank participates, rank 0 receives the full
                # optimizer state used by save_checkpoint().
                optimizer.consolidate_state_dict(to=0)
            if rank == 0:
                save_checkpoint(
                    latest_path,
                    vae_model,
                    optimizer,
                    scheduler,
                    step,
                    metadata,
                )
                log(f"Saved {latest_path}", rank)
                if args.save_step_checkpoints and (
                    step % args.step_checkpoint_every == 0
                    or step == args.max_steps
                ):
                    step_path = os.path.join(
                        args.output_dir,
                        f"checkpoint_step{step:06d}.pt",
                    )
                    save_checkpoint(
                        step_path,
                        vae_model,
                        optimizer,
                        scheduler,
                        step,
                        metadata,
                        include_optimizer=False,
                    )
                    log(f"Saved weights-only {step_path}", rank)
                if optimizer_is_sharded:
                    # consolidate_state_dict() stores a full CPU copy on rank 0.
                    # torch.save() is complete, so release it before evaluation.
                    optimizer._all_state_dicts.clear()
            barrier(is_distributed)

        if step % args.eval_every == 0 or step == args.max_steps:
            barrier(is_distributed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if rank == 0:
                result = evaluate(
                    vae_model,
                    vae.scale,
                    args.dataset_root,
                    eval_episodes,
                    stats,
                    info,
                    args,
                    device,
                    dtype,
                    args.eval_seed,
                )
                path = os.path.join(args.output_dir, f"eval_step{step:06d}.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2)
                log(
                    f"Eval: pos={result['future_position_mae_m']:.6f}m "
                    f"rot={result['future_rotation_mean_deg']:.3f}deg "
                    f"gripper={result['future_gripper_mae']:.6f} "
                    f"center={result['normalized_center_mae']:.6f} "
                    f"direction={result['normalized_direction_mae']:.6f} -> {path}",
                    rank,
                )
            barrier(is_distributed)

    if progress is not None:
        progress.close()
    log("Training complete.", rank)
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--norm-stats-path", default=str(DEFAULT_NORM_STATS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--focal", type=float, default=FOCAL)
    parser.add_argument("--center-scale", type=float, default=CENTER_SCALE)
    parser.add_argument("--dir-scale", type=float, default=DIR_SCALE)
    parser.add_argument("--center-frac", type=float, default=CENTER_FRAC)
    parser.add_argument("--boundary-margin", type=int, default=BOUNDARY_MARGIN)
    parser.add_argument("--outer-margin", type=int, default=OUTER_MARGIN)

    parser.add_argument(
        "--eval-episodes-per-task",
        type=int,
        default=DEFAULT_EVAL_EPISODES_PER_TASK,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--zero-optimizer",
        action="store_true",
        default=True,
        help="Shard AdamW optimizer state across DDP ranks (default: enabled)",
    )
    parser.add_argument(
        "--no-zero-optimizer",
        action="store_false",
        dest="zero_optimizer",
        help="Replicate full AdamW optimizer state on every DDP rank",
    )
    parser.add_argument("--center-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--lr-scheduler",
        choices=["cosine", "constant"],
        default="cosine",
    )
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)

    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-windows", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=12_345,
        help="Fixed validation-window seed shared across steps and configurations",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-step-checkpoints", action="store_true")
    parser.add_argument(
        "--step-checkpoint-every",
        type=int,
        default=500,
        help="Interval for weights-only historical checkpoints",
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--log-file")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.grad_accum_steps < 1:
        parser.error("--batch-size and --grad-accum-steps must be positive")
    if args.eval_every < 1 or args.save_every < 1 or args.log_every < 1:
        parser.error("log/eval/save intervals must be positive")
    if args.step_checkpoint_every < 1:
        parser.error("--step-checkpoint-every must be positive")
    if args.save_step_checkpoints and args.step_checkpoint_every % args.save_every != 0:
        parser.error("--step-checkpoint-every must be divisible by --save-every")
    if args.eval_windows < 1:
        parser.error("--eval-windows must be positive")
    if min(
        args.center_loss_weight,
        args.direction_loss_weight,
        args.gripper_loss_weight,
    ) < 0:
        parser.error("loss weights must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    _, rank, _, _ = _init_distributed()
    log_path = setup_logging(args.output_dir, args.log_file, rank)
    if rank == 0:
        log(f"Logging to {log_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)


if __name__ == "__main__":
    main()
