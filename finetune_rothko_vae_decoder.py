#!/usr/bin/env python
"""Fine-tune Wan2.1/Wan2.2 VAE ``conv2 + decoder`` for LIBERO Rothko maps.

This is a single-script training utility: all LIBERO parquet loading, Rothko
encoding/decoding, validation metrics, checkpointing, and full safetensors
export live in this file.  It only imports the Wan VAE architectures from an
installed FastWAM package; it does not depend on any research/replay script.

The representation exactly follows FastWAM's LIBERO visual-action path:

* frame 0 is ``observation.state.ee_pose_wxyz`` at the window start;
* frames 1..H are ``action.osc_target_pose_wxyz`` for the next H controls;
* the single 224x224 arm map is duplicated horizontally to 224x448;
* the current/future gripper is written as ``2*g-1`` in the outer 8-pixel
  border;
* translation is relative to frame 0 in frame-0 EE coordinates.

The pretrained encoder and ``conv1`` stay frozen.  Trainable parameters and
optimizer states remain FP32; ``--bf16`` enables BF16 autocast for compute
without converting the master weights.  The rolling ``checkpoint_latest.pt``
contains optimizer state for exact resume, while historical
``checkpoint_step*.pt`` files contain decoder/conv2 weights only.  At the end,
the script also exports a complete variant-matched ``.safetensors`` file that
FastWAM can load directly through ``model.vae_safetensors_path``.

Example (all four LIBERO suites with the defaults below, four GPUs):

    torchrun --standalone --nproc_per_node=4 finetune_rothko_vae_decoder.py

Example (LIBERO-Goal only):

    torchrun --standalone --nproc_per_node=4 finetune_rothko_vae_decoder.py \
      --suites libero_goal \
      --output-dir runs/libero_goal_rothko_vae_decoder
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import os
import random
import shlex
import signal
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file as save_safetensors
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, SequentialLR
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_SRC = SCRIPT_DIR / "src"
if LOCAL_SRC.is_dir() and str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

try:
    from fastwam.models.wan22.helpers.io import load_state_dict as load_wan_state_dict
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE, WanVideoVAE38
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "FastWAM must be installed (or this script must remain beside FastWAM's "
        "src/ directory) so the Wan VAE architectures can be imported."
    ) from exc


SUITE_DIRS = {
    "libero_spatial": "libero_spatial_no_noops_lerobot",
    "libero_object": "libero_object_no_noops_lerobot",
    "libero_goal": "libero_goal_no_noops_lerobot",
    "libero_10": "libero_10_no_noops_lerobot",
}
DEFAULT_SUITES = tuple(SUITE_DIRS)
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data/libero_mujoco3.3.2"
DEFAULT_BASE_VAE = (
    SCRIPT_DIR
    / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
)
DEFAULT_NORM_STATS = (
    DEFAULT_DATA_ROOT / "libero_rothko_region_symmetric_q99p95_h16_224x448.pt"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs/libero_rothko_vae_decoder"
DEFAULT_CONFIG_PATH = (
    SCRIPT_DIR
    / "configs/vae/libero_rothko_decoder_all4_h16_bs2_ga8_lr1e-5_ep2.json"
)

VAE_VARIANTS = {
    "wan2.1-t2v-1.3b": (WanVideoVAE, "Wan2.1_VAE"),
    "wan2.2-ti2v-5b": (WanVideoVAE38, "Wan2.2_VAE"),
}
DEFAULT_VAE_VARIANT = "wan2.2-ti2v-5b"

ACTION_HORIZON = 16
PIXEL_FRAMES = ACTION_HORIZON + 1
IMAGE_H = TILE_H = 224
IMAGE_W = 448
TILE_W = 224
FOCAL = 0.2
CENTER_SCALE = 1.0
DIR_SCALE = 1.0
CENTER_FRAC = 0.5
BOUNDARY_MARGIN = 8
OUTER_MARGIN = 8

STATE_POSE_KEY = "observation.state.ee_pose_wxyz"
STATE_GRIPPER_KEY = "observation.state.gripper_open"
ACTION_POSE_KEY = "action.osc_target_pose_wxyz"
ACTION_KEY = "action"

_LOG_FILE: Optional[Any] = None
_CHECKPOINT_STATE: dict[str, Any] = {}


@dataclass(frozen=True)
class EpisodeRef:
    dataset_root: str
    episode_index: int
    task: str
    length: int


@dataclass(frozen=True)
class WindowRef:
    episode: EpisodeRef
    start: int


@dataclass
class EpisodeArrays:
    state_pose: np.ndarray
    state_gripper: np.ndarray
    action_pose: np.ndarray
    action: np.ndarray


@dataclass
class ActionWindow:
    episode: EpisodeRef
    start: int
    pose: np.ndarray  # [17,7]: current state pose + 16 absolute OSC targets
    gripper: np.ndarray  # [17,1]: current state gripper + 16 action grippers


@dataclass
class EncodedWindow:
    video: Tensor  # [3,17,224,448], normalized to [-1,1]
    pose: Tensor  # [17,7], metric/wxyz
    gripper: Tensor  # [17,1], [0,1]


@dataclass
class RothkoNormStats:
    lo: Tensor
    hi: Tensor
    metadata: dict[str, Any]


class EpisodeStore:
    """Small per-process LRU cache to avoid repeatedly reading parquet files."""

    def __init__(self, cache_size: int):
        self.cache_size = max(int(cache_size), 0)
        self.cache: OrderedDict[EpisodeRef, EpisodeArrays] = OrderedDict()

    def get(self, episode: EpisodeRef) -> EpisodeArrays:
        if episode in self.cache:
            value = self.cache.pop(episode)
            self.cache[episode] = value
            return value
        value = load_episode_arrays(episode)
        if self.cache_size:
            self.cache[episode] = value
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return value


def _init_distributed() -> tuple[bool, int, int, int]:
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


def init_wandb(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    rank: int,
) -> Optional[Any]:
    """Initialize one resumable W&B run on rank 0."""
    if rank != 0 or not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging is enabled but `wandb` is not installed. "
            "Install it or pass --no-wandb."
        ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id_path = output_dir / "wandb_run_id.txt"
    if args.fresh or not run_id_path.is_file():
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
    else:
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        if not run_id:
            run_id = wandb.util.generate_id()
            run_id_path.write_text(run_id + "\n", encoding="utf-8")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or output_dir.name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        id=run_id,
        resume="allow",
        dir=str(output_dir),
        config={"arguments": vars(args), "training": metadata},
    )
    log(
        "Initialized W&B: "
        f"project={args.wandb_project} name={args.wandb_name or output_dir.name} "
        f"id={run_id} mode={args.wandb_mode}",
        rank,
    )
    return run


def quaternion_wxyz_to_matrix(quaternion: Tensor) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion [...,4], got {tuple(quaternion.shape)}")
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion_wxyz(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix [...,3,3], got {tuple(matrix.shape)}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    candidates = torch.stack(
        (
            torch.stack((1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20), -1),
            torch.stack((m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21), -1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22), -1),
        ),
        dim=-2,
    )
    denominators = (candidates[..., :, 0].clamp_min(0).sqrt() * 2).clamp_min(1e-8)
    candidates = candidates / denominators.unsqueeze(-1)
    best = denominators.argmax(dim=-1, keepdim=True)
    quaternion = torch.gather(
        candidates,
        -2,
        best[..., None].expand(best.shape + (4,)),
    ).squeeze(-2)
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def region_masks(
    height: int,
    width: int,
    center_frac: float,
    boundary_margin: int,
    outer_margin: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    center_h = max(1, int(round(height * center_frac)))
    center_w = max(1, int(round(width * center_frac)))
    y0, x0 = (height - center_h) // 2, (width - center_w) // 2
    y1, x1 = y0 + center_h, x0 + center_w
    center = torch.zeros(height, width, dtype=torch.bool, device=device)
    center[y0:y1, x0:x1] = True

    origin = torch.zeros_like(center)
    origin[y0 + boundary_margin : y1 - boundary_margin,
           x0 + boundary_margin : x1 - boundary_margin] = True
    expanded_center = torch.zeros_like(center)
    expanded_center[
        max(y0 - boundary_margin, 0) : min(y1 + boundary_margin, height),
        max(x0 - boundary_margin, 0) : min(x1 + boundary_margin, width),
    ] = True
    direction = ~expanded_center
    border = torch.ones_like(center)
    if outer_margin:
        border[outer_margin : height - outer_margin,
               outer_margin : width - outer_margin] = False
        direction &= ~border
    else:
        border.zero_()
    center &= ~border
    if not origin.any() or not direction.any():
        raise ValueError("Rothko decode mask is empty.")
    return center, origin, direction, border


def canonical_directions(device: torch.device, dtype: torch.dtype, focal: float) -> Tensor:
    dx, dy = 1.0 / TILE_W, 1.0 / TILE_H
    y, x = torch.meshgrid(
        torch.linspace(1 - dy, -(1 - dy), TILE_H, device=device, dtype=dtype),
        torch.linspace(1 - dx, -(1 - dx), TILE_W, device=device, dtype=dtype),
        indexing="ij",
    )
    directions = torch.stack((x / focal, y / focal, torch.ones_like(x)), dim=-1)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def load_norm_stats(path: str, args: argparse.Namespace) -> RothkoNormStats:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "lo" not in payload or "hi" not in payload:
        raise ValueError(f"Invalid Rothko norm stats: {path}")
    stats = RothkoNormStats(
        lo=torch.as_tensor(payload["lo"], dtype=torch.float32),
        hi=torch.as_tensor(payload["hi"], dtype=torch.float32),
        metadata=dict(payload.get("metadata") or {}),
    )
    if stats.lo.shape != stats.hi.shape or stats.lo.ndim != 4 or stats.lo.shape[:2] != (1, 3):
        raise ValueError(f"Invalid norm tensor shapes: {stats.lo.shape}, {stats.hi.shape}")
    expected = {
        "environment": "libero",
        "representation": "rothko",
        "raymap_representation": "libero_rothko",
        "encoding": "current_ee_plus_future_absolute_osc_targets",
        "layout": "single_arm_duplicated_horizontal",
        "image_size": [IMAGE_H, IMAGE_W],
        "tile_size": [TILE_H, TILE_W],
        "quaternion_order": "wxyz",
        "action_horizon": args.action_horizon,
        "pixel_frames": args.action_horizon + 1,
        "focal": args.focal,
        "center_scale": args.center_scale,
        "dir_scale": args.dir_scale,
        "center_frac": args.center_frac,
        "boundary_margin": args.boundary_margin,
        "outer_margin": args.outer_margin,
    }
    mismatches = []
    for key, expected_value in expected.items():
        if key not in stats.metadata:
            mismatches.append(f"{key}: missing")
            continue
        actual = stats.metadata[key]
        if isinstance(expected_value, float):
            matches = abs(float(actual) - expected_value) <= 1e-8
        else:
            matches = actual == expected_value
        if not matches:
            mismatches.append(f"{key}: stats={actual!r}, expected={expected_value!r}")
    if mismatches:
        raise ValueError("LIBERO Rothko norm metadata mismatch: " + "; ".join(mismatches))
    if stats.lo.shape[-2:] not in {(IMAGE_H, IMAGE_W), (TILE_H, TILE_W)}:
        raise ValueError(f"Unexpected norm stats spatial shape: {stats.lo.shape[-2:]}")
    return stats


def expanded_stats(stats: RothkoNormStats, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    lo = stats.lo.to(device=device, dtype=dtype)
    hi = stats.hi.to(device=device, dtype=dtype)
    if lo.shape[-2:] == (TILE_H, TILE_W):
        lo, hi = torch.cat((lo, lo), -1), torch.cat((hi, hi), -1)
    return lo, hi


def encode_rothko(pose: Tensor, gripper: Tensor, stats: RothkoNormStats, args: argparse.Namespace) -> Tensor:
    """Encode one window and return [3,T,224,448] normalized Rothko video."""
    position = pose[:, :3]
    rotation = quaternion_wxyz_to_matrix(pose[:, 3:7])
    base_position = position[:1]
    base_rotation = rotation[:1]
    relative_position = torch.einsum(
        "tij,tj->ti", base_rotation.transpose(-1, -2).expand_as(rotation), position - base_position
    )
    relative_rotation = base_rotation.transpose(-1, -2) @ rotation
    canonical = canonical_directions(pose.device, pose.dtype, args.focal)
    tile = torch.einsum("tij,hwj->tihw", relative_rotation, canonical) * args.dir_scale
    center, _, _, border = region_masks(
        TILE_H, TILE_W, args.center_frac, 0, args.outer_margin, pose.device
    )
    tile[..., center] = (relative_position * args.center_scale)[..., :, None]
    raw = torch.cat((tile, tile), dim=-1)
    lo, hi = expanded_stats(stats, pose.device, pose.dtype)
    normalized = (2 * (raw - lo) / (hi - lo).clamp_min(1e-6) - 1).clamp(-1, 1)
    code = gripper[:, 0].clamp(0, 1).mul(2).sub(1).to(normalized.dtype)
    for x_offset in (0, TILE_W):
        target_tile = normalized[..., x_offset : x_offset + TILE_W]
        target_tile[..., border] = code[:, None, None].expand(-1, 3, int(border.sum()))
    return normalized.permute(1, 0, 2, 3).contiguous()


def denormalize_rothko(video: Tensor, stats: RothkoNormStats) -> Tensor:
    """Convert [B,3,T,H,W] normalized video to [B,T,3,H,W] raw maps."""
    normalized = video.permute(0, 2, 1, 3, 4).contiguous()
    lo, hi = expanded_stats(stats, normalized.device, normalized.dtype)
    return (normalized + 1) * 0.5 * (hi - lo).unsqueeze(1) + lo.unsqueeze(1)


def read_gripper(video: Tensor, args: argparse.Namespace) -> Tensor:
    normalized = video.permute(0, 2, 1, 3, 4).contiguous()
    _, _, _, border = region_masks(
        TILE_H, TILE_W, args.center_frac, args.boundary_margin, args.outer_margin, video.device
    )
    observations = []
    for x_offset in (0, TILE_W):
        tile = normalized[..., x_offset : x_offset + TILE_W]
        observations.append(tile[..., border].reshape(*tile.shape[:2], -1))
    code = torch.cat(observations, -1).median(-1).values
    return ((code + 1) * 0.5).clamp(0, 1).unsqueeze(-1)


def decode_pose(raw: Tensor, current_pose: Tensor, args: argparse.Namespace) -> Tensor:
    """Decode [B,T,3,224,448] raw maps to absolute [B,T,7] poses."""
    tile = 0.5 * (raw[..., :TILE_W] + raw[..., TILE_W:])
    _, origin_mask, direction_mask, _ = region_masks(
        TILE_H, TILE_W, args.center_frac, args.boundary_margin, args.outer_margin, raw.device
    )
    values = tile.permute(0, 1, 3, 4, 2).to(torch.float64)
    origin_code = values[:, :, origin_mask].median(2).values
    relative_position = (origin_code - origin_code[:, :1]) / args.center_scale
    directions = values[:, :, direction_mask]
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    reference = directions[:, 0]
    correlation = torch.einsum("btpi,bpj->btij", directions, reference) / directions.shape[2]
    u, _, vh = torch.linalg.svd(correlation)
    determinant = torch.linalg.det(u @ vh)
    correction = torch.eye(3, device=raw.device, dtype=torch.float64).repeat(
        raw.shape[0], raw.shape[1], 1, 1
    )
    correction[..., 2, 2] = torch.where(determinant >= 0, 1.0, -1.0)
    relative_rotation = u @ correction @ vh
    relative_position[:, 0] = 0
    relative_rotation[:, 0] = torch.eye(3, device=raw.device, dtype=torch.float64)
    base_position = current_pose[:, None, :3].to(torch.float64)
    base_rotation = quaternion_wxyz_to_matrix(current_pose[:, 3:7].to(torch.float64))[:, None]
    position = base_position + torch.einsum("btij,btj->bti", base_rotation, relative_position)
    rotation = base_rotation @ relative_rotation
    return torch.cat((position, matrix_to_quaternion_wxyz(rotation)), -1).to(raw.dtype)


def load_dataset_info(dataset_root: str) -> dict[str, Any]:
    with open(Path(dataset_root) / "meta/info.json", encoding="utf-8") as handle:
        return json.load(handle)


def episode_parquet_path(episode: EpisodeRef) -> Path:
    info = load_dataset_info(episode.dataset_root)
    relative = info["data_path"].format(
        episode_chunk=episode.episode_index // int(info["chunks_size"]),
        episode_index=episode.episode_index,
    )
    return Path(episode.dataset_root) / relative


def _column_to_numpy(table, key: str) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)


def load_episode_arrays(episode: EpisodeRef) -> EpisodeArrays:
    path = episode_parquet_path(episode)
    table = pq.read_table(
        path,
        columns=[STATE_POSE_KEY, STATE_GRIPPER_KEY, ACTION_POSE_KEY, ACTION_KEY],
    )
    result = EpisodeArrays(
        state_pose=_column_to_numpy(table, STATE_POSE_KEY),
        state_gripper=_column_to_numpy(table, STATE_GRIPPER_KEY),
        action_pose=_column_to_numpy(table, ACTION_POSE_KEY),
        action=_column_to_numpy(table, ACTION_KEY),
    )
    lengths = {len(result.state_pose), len(result.state_gripper), len(result.action_pose), len(result.action)}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent episode column lengths: {path}")
    if result.state_pose.shape[1:] != (7,) or result.action_pose.shape[1:] != (7,):
        raise ValueError(f"Expected wxyz pose7 fields in {path}")
    if result.state_gripper.ndim == 1:
        result.state_gripper = result.state_gripper[:, None]
    if result.state_gripper.shape[1:] != (1,) or result.action.shape[-1] < 1:
        raise ValueError(f"Unexpected gripper/action shapes in {path}")
    return result


def discover_episodes(
    args: argparse.Namespace,
    rank: int = 0,
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    if args.dataset_roots:
        roots = [Path(path).expanduser().resolve() for path in args.dataset_roots]
    else:
        base = Path(args.data_root).expanduser().resolve()
        roots = [(base / SUITE_DIRS[name]).resolve() for name in args.suites]
    all_train: list[EpisodeRef] = []
    all_eval: list[EpisodeRef] = []
    for root in roots:
        info = load_dataset_info(str(root))
        records = []
        with open(root / "meta/episodes.jsonl", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                task_list = record.get("tasks") or ["unknown"]
                records.append(
                    (
                        int(record["episode_index"]),
                        str(task_list[0]),
                        int(record["length"]),
                    )
                )
        if len(records) != int(info["total_episodes"]):
            raise ValueError(f"Episode metadata count mismatch in {root}")
        by_task: dict[str, list[int]] = defaultdict(list)
        episode_lengths = {}
        for episode_index, task, length in records:
            by_task[task].append(episode_index)
            episode_lengths[episode_index] = length
        root_train = root_eval = 0
        for task, indices in sorted(by_task.items()):
            stable = int.from_bytes(
                hashlib.sha256(f"{root.name}\0{task}".encode()).digest()[:8], "big"
            )
            rng = random.Random(args.split_seed + stable)
            indices = list(indices)
            rng.shuffle(indices)
            eval_count = min(args.eval_episodes_per_task, max(len(indices) - 1, 0))
            eval_indices = set(indices[:eval_count])
            for episode_index in sorted(indices):
                ref = EpisodeRef(
                    str(root), episode_index, task, episode_lengths[episode_index]
                )
                if episode_index in eval_indices:
                    all_eval.append(ref)
                    root_eval += 1
                else:
                    all_train.append(ref)
                    root_train += 1
        log(
            f"Split {root.name}: tasks={len(by_task)} "
            f"train={root_train} eval={root_eval}",
            rank,
        )
    if not all_train or not all_eval:
        raise ValueError("Training and evaluation episode pools must both be non-empty.")
    return all_train, all_eval


def enumerate_window_refs(
    episode_pool: Sequence[EpisodeRef], horizon: int
) -> list[WindowRef]:
    """Enumerate every valid continuous action window exactly once."""
    refs = []
    for episode in episode_pool:
        valid = episode.length - horizon + 1
        refs.extend(WindowRef(episode, start) for start in range(max(valid, 0)))
    if not refs:
        raise ValueError("No valid training windows were found.")
    return refs


def materialize_windows(
    store: EpisodeStore, refs: Sequence[WindowRef], horizon: int
) -> list[ActionWindow]:
    windows = []
    for ref in refs:
        arrays = store.get(ref.episode)
        stop = ref.start + horizon
        pose = np.concatenate(
            (
                arrays.state_pose[ref.start : ref.start + 1],
                arrays.action_pose[ref.start:stop],
            ),
            axis=0,
        )
        future_gripper = np.clip(arrays.action[ref.start:stop, -1:], 0, 1)
        gripper = np.concatenate(
            (
                np.clip(arrays.state_gripper[ref.start : ref.start + 1], 0, 1),
                future_gripper,
            ),
            axis=0,
        )
        windows.append(ActionWindow(ref.episode, ref.start, pose, gripper))
    return windows


def sample_random_windows(
    store: EpisodeStore,
    episode_pool: Sequence[EpisodeRef],
    num_windows: int,
    horizon: int,
    seed: int,
) -> list[ActionWindow]:
    rng = random.Random(seed)
    windows: list[ActionWindow] = []
    attempts = 0
    max_attempts = max(100, num_windows * 20)
    while len(windows) < num_windows and attempts < max_attempts:
        attempts += 1
        episode = rng.choice(episode_pool)
        try:
            arrays = store.get(episode)
        except Exception:
            continue
        valid = len(arrays.action_pose) - horizon + 1
        if valid <= 0:
            continue
        start = rng.randrange(valid)
        stop = start + horizon
        pose = np.concatenate(
            (arrays.state_pose[start : start + 1], arrays.action_pose[start:stop]),
            axis=0,
        )
        future_gripper = np.clip(arrays.action[start:stop, -1:], 0, 1)
        gripper = np.concatenate(
            (np.clip(arrays.state_gripper[start : start + 1], 0, 1), future_gripper),
            axis=0,
        )
        windows.append(ActionWindow(episode, start, pose, gripper))
    if len(windows) != num_windows:
        raise RuntimeError(f"Collected only {len(windows)}/{num_windows} windows")
    return windows


def sample_uniform_windows_per_episode(
    store: EpisodeStore,
    episode_pool: Sequence[EpisodeRef],
    windows_per_episode: int,
    horizon: int,
) -> list[ActionWindow]:
    """Select fixed, uniformly spaced windows from every validation episode.

    Unlike the legacy global random sampler, this guarantees coverage of every
    held-out episode (and therefore every task represented by the stratified
    episode split).  The selected starts include the beginning and end of each
    episode whenever at least two windows are requested.
    """
    refs: list[WindowRef] = []
    ordered_episodes = sorted(
        episode_pool,
        key=lambda episode: (
            episode.dataset_root,
            episode.task,
            episode.episode_index,
        ),
    )
    for episode in ordered_episodes:
        valid = episode.length - horizon + 1
        if valid <= 0:
            continue
        if valid <= windows_per_episode:
            starts = np.arange(valid, dtype=np.int64)
        else:
            starts = np.rint(
                np.linspace(0, valid - 1, windows_per_episode)
            ).astype(np.int64)
        refs.extend(WindowRef(episode, int(start)) for start in np.unique(starts))
    if not refs:
        raise RuntimeError("No valid uniformly sampled validation windows were found")
    return materialize_windows(store, refs, horizon)


def build_target_batch(
    windows: Sequence[ActionWindow],
    stats: RothkoNormStats,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[Tensor, list[EncodedWindow]]:
    encoded = []
    for window in windows:
        pose = torch.from_numpy(window.pose).to(device=device, dtype=torch.float32)
        gripper = torch.from_numpy(window.gripper).to(device=device, dtype=torch.float32)
        video = encode_rothko(pose, gripper, stats, args).to(dtype=dtype)
        encoded.append(EncodedWindow(video, pose, gripper))
    return torch.stack([item.video for item in encoded]), encoded


def build_loss_masks(args: argparse.Namespace, device: torch.device) -> dict[str, Tensor]:
    center, _, direction, border = region_masks(
        TILE_H, TILE_W, args.center_frac, args.boundary_margin, args.outer_margin, device
    )
    return {
        "center": torch.cat((center, center), -1),
        "direction": torch.cat((direction, direction), -1),
        "gripper": torch.cat((border, border), -1),
    }


def masked_l1(reconstruction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return (reconstruction - target).abs()[..., mask].mean()


def compute_reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
    masks: dict[str, Tensor],
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor]]:
    losses = {name: masked_l1(reconstruction, target, mask) for name, mask in masks.items()}
    total = (
        args.center_loss_weight * losses["center"]
        + args.direction_loss_weight * losses["direction"]
        + args.gripper_loss_weight * losses["gripper"]
    )
    return total, losses


def load_base_vae(
    path: str,
    vae_variant: str,
    device: torch.device,
    dtype: torch.dtype,
) -> WanVideoVAE:
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.is_file() or path_obj.suffix.lower() not in {
        ".safetensors", ".pth", ".pt", ".bin"
    }:
        raise FileNotFoundError(f"Complete base VAE weights not found: {path_obj}")
    if vae_variant not in VAE_VARIANTS:
        raise ValueError(
            f"Unsupported VAE variant {vae_variant!r}; expected one of {sorted(VAE_VARIANTS)}"
        )
    vae_class, _ = VAE_VARIANTS[vae_variant]
    vae = vae_class().to(dtype=dtype)
    expected = set(vae.state_dict())
    loaded = load_wan_state_dict(str(path_obj), torch_dtype=dtype, device="cpu")
    provided = set(loaded)
    if provided == expected:
        state = loaded
    elif {f"model.{key}" for key in provided} == expected:
        state = {f"model.{key}": value for key, value in loaded.items()}
    else:
        direct_missing = sorted(expected - provided)
        direct_unexpected = sorted(provided - expected)
        prefixed = {f"model.{key}" for key in provided}
        prefixed_missing = sorted(expected - prefixed)
        prefixed_unexpected = sorted(prefixed - expected)
        if len(prefixed_missing) < len(direct_missing):
            direct_missing, direct_unexpected = prefixed_missing, prefixed_unexpected
        raise ValueError(
            f"Base VAE is not a complete {vae_class.__name__} state dict for "
            f"variant={vae_variant}: missing={direct_missing[:8]}, "
            f"unexpected={direct_unexpected[:8]}, path={path_obj}"
        )
    vae.load_state_dict(state, strict=True)
    del loaded, state
    return vae.to(device=device, dtype=dtype).eval()


def prepare_decoder_finetune(vae: WanVideoVAE) -> list[nn.Parameter]:
    vae.model.requires_grad_(False)
    vae.model.encoder.eval()
    vae.model.conv1.eval()
    vae.model.decoder.train().requires_grad_(True)
    vae.model.conv2.train().requires_grad_(True)
    return [parameter for parameter in vae.model.parameters() if parameter.requires_grad]


class VaeDecodeWrapper(nn.Module):
    def __init__(self, vae_model: nn.Module):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents: Tensor, scale) -> Tensor:
        # Do not clamp here: clamp has zero gradient outside [-1,1].
        return self.vae_model.decode(latents, scale)


@torch.no_grad()
def vae_encode(vae_model: nn.Module, video: Tensor, scale) -> Tensor:
    return vae_model.encode(video, scale)


def autocast_context(device: torch.device, enabled: bool):
    """Use BF16 compute while retaining FP32 model parameters and optimizer state."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    scheduler_type: str,
) -> Optional[LRScheduler]:
    if scheduler_type == "constant":
        return None
    remaining = max(max_steps - warmup_steps, 1)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=remaining,
        eta_min=float(optimizer.param_groups[0]["lr"]) * min_lr_ratio,
    )
    if warmup_steps <= 0:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_steps, 1),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def save_checkpoint(
    path: str,
    vae: WanVideoVAE,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LRScheduler],
    step: int,
    metadata: dict[str, Any],
    include_optimizer: bool = True,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "step": step,
        "decoder": vae.model.decoder.state_dict(),
        "conv2": vae.model.conv2.state_dict(),
        "optimizer": optimizer.state_dict() if include_optimizer and optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metadata": metadata,
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str,
    vae: WanVideoVAE,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LRScheduler],
    *,
    allow_weights_only_resume: bool = False,
) -> tuple[int, bool]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vae.model.decoder.load_state_dict(payload["decoder"], strict=True)
    vae.model.conv2.load_state_dict(payload["conv2"], strict=True)
    optimizer_state = payload.get("optimizer")
    scheduler_state = payload.get("scheduler")
    if optimizer is not None and optimizer_state is None and not allow_weights_only_resume:
        raise RuntimeError(
            f"Checkpoint {path} contains model weights but no optimizer state, so it "
            "cannot perform an exact training resume. Pass "
            "--allow-weights-only-resume only when resetting Adam state is intentional."
        )
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        # Constructing SequentialLR initializes the optimizer at the first
        # warmup LR.  Normally optimizer.load_state_dict() replaces that LR,
        # but an emergency weights-only checkpoint has no optimizer state.
        # Restore the scheduler's last LR explicitly so the next update
        # continues from the saved training step instead of restarting at the
        # tiny warmup-start learning rate.
        if optimizer is not None and optimizer_state is None:
            saved_lrs = scheduler_state.get("_last_lr")
            if saved_lrs is None or len(saved_lrs) != len(optimizer.param_groups):
                raise ValueError(
                    "Weights-only checkpoint cannot restore optimizer LR: "
                    "scheduler `_last_lr` is missing or has the wrong length."
                )
            for param_group, saved_lr in zip(optimizer.param_groups, saved_lrs):
                param_group["lr"] = float(saved_lr)
    return int(payload.get("step", 0)), optimizer_state is not None


def export_complete_vae(
    path: str,
    vae: WanVideoVAE,
    step: int,
    metadata: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Training uses FP32 master parameters, but FastWAM inference runs this VAE
    # in BF16 and the original Wan VAEs are distributed in BF16. Quantize
    # only the final deployment export; resumable .pt checkpoints remain FP32.
    state = {
        key.removeprefix("model."): value.detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        for key, value in vae.state_dict().items()
    }
    temporary = path + ".tmp"
    save_safetensors(
        state,
        temporary,
        metadata={
            "model": VAE_VARIANTS[metadata["vae_variant"]][1],
            "format": "pt",
            "dtype": "bfloat16",
            "finetuned_modules": "conv2,decoder",
            "checkpoint_step": str(step),
            "training_metadata_json": json.dumps(metadata, separators=(",", ":")),
        },
    )
    os.replace(temporary, path)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    latest = os.path.join(output_dir, "checkpoint_latest.pt")
    if os.path.isfile(latest):
        return latest
    candidates = []
    for path in glob.glob(os.path.join(output_dir, "checkpoint_step*.pt")):
        token = Path(path).stem.removeprefix("checkpoint_step")
        if token.isdigit():
            candidates.append((int(token), path))
    return max(candidates, default=(-1, None))[1]


def resolve_resume(args: argparse.Namespace) -> Optional[str]:
    if args.fresh:
        return None
    if args.resume:
        if args.resume in {"auto", "latest"}:
            path = find_latest_checkpoint(args.output_dir)
            if path is None:
                raise FileNotFoundError(f"No checkpoint in {args.output_dir}")
            return path
        return str(Path(args.resume).expanduser().resolve())
    return find_latest_checkpoint(args.output_dir) if args.auto_resume else None


def rotation_angle_degrees(prediction: Tensor, target: Tensor) -> Tensor:
    relative = prediction.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cosine))


@torch.no_grad()
def evaluate(
    vae: WanVideoVAE,
    store: EpisodeStore,
    eval_episodes: Sequence[EpisodeRef],
    stats: RothkoNormStats,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    vae.model.decoder.eval()
    vae.model.conv2.eval()
    if args.eval_sampling == "uniform_per_episode":
        windows = sample_uniform_windows_per_episode(
            store,
            eval_episodes,
            args.eval_windows_per_episode,
            args.action_horizon,
        )
    else:
        windows = sample_random_windows(
            store, eval_episodes, args.eval_windows, args.action_horizon, args.eval_seed
        )
    windows_per_task: dict[str, int] = defaultdict(int)
    for window in windows:
        dataset_name = Path(window.episode.dataset_root).name
        suite_name = next(
            (suite for suite, directory in SUITE_DIRS.items() if directory == dataset_name),
            dataset_name,
        )
        windows_per_task[f"{suite_name}/{window.episode.task}"] += 1
    masks = build_loss_masks(args, device)
    totals = defaultdict(float)
    maxima = defaultdict(float)
    for offset in range(0, len(windows), args.eval_batch_size):
        group = windows[offset : offset + args.eval_batch_size]
        targets, encoded = build_target_batch(group, stats, device, dtype, args)
        with autocast_context(device, args.bf16):
            latents = vae_encode(vae.model, targets, vae.scale)
            reconstruction = vae.model.decode(latents, vae.scale)
        reconstruction = reconstruction.float().clamp(-1, 1)
        for region in ("center", "direction", "gripper"):
            totals[f"normalized_{region}_mae"] += float(
                masked_l1(reconstruction, targets.float(), masks[region]).item()
            ) * len(group)
        raw = denormalize_rothko(reconstruction, stats)
        totals["duplicate_mae"] += float(
            (raw[..., :TILE_W] - raw[..., TILE_W:]).abs().mean().item()
        ) * len(group)
        current = torch.stack([item.pose[0] for item in encoded])
        pose_prediction = decode_pose(raw, current, args)
        pose_target = torch.stack([item.pose for item in encoded]).to(pose_prediction)
        position_error = (pose_prediction[:, 1:, :3] - pose_target[:, 1:, :3]).norm(dim=-1)
        rotation_error = rotation_angle_degrees(
            quaternion_wxyz_to_matrix(pose_prediction[:, 1:, 3:7].float()),
            quaternion_wxyz_to_matrix(pose_target[:, 1:, 3:7].float()),
        )
        gripper_prediction = read_gripper(reconstruction, args)
        gripper_target = torch.stack([item.gripper for item in encoded]).to(gripper_prediction)
        gripper_error = (gripper_prediction[:, 1:] - gripper_target[:, 1:]).abs()
        totals["future_position_mae_m"] += float(position_error.sum().item())
        totals["future_rotation_mean_deg"] += float(rotation_error.sum().item())
        totals["future_gripper_mae"] += float(gripper_error.sum().item())
        totals["future_element_count"] += position_error.numel()
        maxima["future_position_max_m"] = max(maxima["future_position_max_m"], float(position_error.max()))
        maxima["future_rotation_max_deg"] = max(maxima["future_rotation_max_deg"], float(rotation_error.max()))
        maxima["future_gripper_max"] = max(maxima["future_gripper_max"], float(gripper_error.max()))
    count = len(windows)
    result = {
        "normalized_center_mae": totals["normalized_center_mae"] / count,
        "normalized_direction_mae": totals["normalized_direction_mae"] / count,
        "normalized_gripper_mae": totals["normalized_gripper_mae"] / count,
        "duplicate_mae": totals["duplicate_mae"] / count,
        "future_position_mae_m": totals["future_position_mae_m"] / totals["future_element_count"],
        "future_rotation_mean_deg": totals["future_rotation_mean_deg"] / totals["future_element_count"],
        "future_gripper_mae": totals["future_gripper_mae"] / totals["future_element_count"],
        **maxima,
        "num_windows": count,
        "num_eval_episodes": len(eval_episodes),
        "num_eval_tasks": len(windows_per_task),
        "eval_sampling": args.eval_sampling,
        "windows_per_task": dict(sorted(windows_per_task.items())),
    }
    vae.model.decoder.train()
    vae.model.conv2.train()
    return result


def install_signal_handlers() -> None:
    def handler(signum: int, _frame: Any) -> None:
        state = _CHECKPOINT_STATE
        if state and state.get("rank") == 0:
            log(f"Received signal {signum}; saving emergency weights checkpoint...")
            try:
                emergency_path = str(
                    Path(state["path"]).with_name("checkpoint_emergency.pt")
                )
                save_checkpoint(
                    emergency_path, state["vae"], None, state["scheduler"],
                    state["step"], state["metadata"], include_optimizer=False,
                )
                log(f"Saved emergency weights checkpoint: {emergency_path}")
            except Exception:
                log("Emergency checkpoint failed:\n" + traceback.format_exc())
        if state.get("is_distributed") and dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def train(args: argparse.Namespace) -> None:
    is_distributed, rank, world_size, local_rank = _init_distributed()
    device = (
        torch.device(f"cuda:{local_rank}")
        if is_distributed
        else torch.device(args.device if torch.cuda.is_available() else "cpu")
    )
    # Keep master weights, gradients, and AdamW state in FP32.  BF16 is used
    # only through autocast so lr=1e-5 updates are not rounded out of the
    # trainable decoder parameters.
    dtype = torch.float32
    log_path = setup_logging(args.output_dir, args.log_file, rank)
    log(f"Logging to {log_path}", rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    train_episodes, eval_episodes = discover_episodes(args, rank=rank)
    log(f"Total split: train={len(train_episodes)} eval={len(eval_episodes)}", rank)
    train_window_refs = enumerate_window_refs(train_episodes, args.action_horizon)
    effective_batch_size = args.batch_size * world_size * args.grad_accum_steps
    steps_per_epoch = (len(train_window_refs) + effective_batch_size - 1) // effective_batch_size
    args.max_steps = args.epochs * steps_per_epoch
    padded_windows_per_epoch = steps_per_epoch * effective_batch_size
    log(
        f"Epoch training: windows={len(train_window_refs):,}, epochs={args.epochs}, "
        f"effective_batch={effective_batch_size}, steps_per_epoch={steps_per_epoch:,}, "
        f"max_steps={args.max_steps:,}, padding_per_epoch="
        f"{padded_windows_per_epoch - len(train_window_refs)}",
        rank,
    )
    stats = load_norm_stats(str(Path(args.norm_stats_path).expanduser().resolve()), args)
    log(
        f"Norm stats: {args.norm_stats_path}; bounds="
        f"{stats.metadata.get('translation_abs_bounds_xyz_m')}", rank,
    )
    vae = load_base_vae(args.base_vae, args.vae_variant, device, dtype)
    trainable = prepare_decoder_finetune(vae)
    non_fp32_trainable = [parameter.dtype for parameter in trainable if parameter.dtype != torch.float32]
    if non_fp32_trainable:
        raise RuntimeError(
            "Decoder fine-tuning requires FP32 master parameters, got "
            f"non-FP32 dtypes: {sorted({str(value) for value in non_fp32_trainable})}"
        )
    log(f"Trainable conv2+decoder params: {sum(p.numel() for p in trainable):,}", rank)
    log(
        "Precision: FP32 master weights/optimizer with "
        + ("BF16 autocast" if args.bf16 and device.type == "cuda" else "FP32 compute"),
        rank,
    )
    wrapper = VaeDecodeWrapper(vae.model)
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

    optimizer_is_sharded = is_distributed and args.zero_optimizer
    if optimizer_is_sharded:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            trainable,
            optimizer_class=torch.optim.AdamW,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = (
        min(args.warmup_steps, args.max_steps - 1)
        if args.warmup_steps is not None
        else min(int(args.max_steps * args.warmup_ratio), args.max_steps - 1)
    )
    scheduler = build_scheduler(
        optimizer, args.max_steps, warmup_steps, args.min_lr_ratio, args.lr_scheduler
    )

    step = 0
    optimizer_state_restored = False
    resume_path = resolve_resume(args)
    if resume_path:
        step, optimizer_state_restored = load_checkpoint(
            resume_path,
            vae,
            optimizer,
            scheduler,
            allow_weights_only_resume=args.allow_weights_only_resume,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        resume_kind = "full optimizer" if optimizer_state_restored else "WEIGHTS ONLY; Adam reset"
        log(
            f"Resumed {resume_path} at step {step}; state={resume_kind}; "
            f"lr={current_lr:.8e}",
            rank,
        )
    barrier(is_distributed)

    runtime_config = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "python": sys.version,
        "torch": torch.__version__,
        "world_size": world_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
    }
    metadata = {
        "environment": "libero",
        "representation": "rothko",
        "raymap_representation": "libero_rothko",
        "encoding": "current_ee_plus_future_absolute_osc_targets",
        "layout": "single_arm_duplicated_horizontal",
        "action_horizon": args.action_horizon,
        "pixel_frames": PIXEL_FRAMES,
        "image_size": [IMAGE_H, IMAGE_W],
        "tile_size": [TILE_H, TILE_W],
        "focal": args.focal,
        "center_scale": args.center_scale,
        "dir_scale": args.dir_scale,
        "center_frac": args.center_frac,
        "boundary_margin": args.boundary_margin,
        "outer_margin": args.outer_margin,
        "quaternion_order": "wxyz",
        "gripper_encoding": "normalized_outer_border_2g_minus_1",
        "dataset_roots": sorted({episode.dataset_root for episode in train_episodes}),
        "train_episodes": len(train_episodes),
        "eval_episodes": len(eval_episodes),
        "train_windows": len(train_window_refs),
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": args.max_steps,
        "padded_windows_per_epoch": padded_windows_per_epoch,
        "base_vae": str(Path(args.base_vae).expanduser().resolve()),
        "vae_variant": args.vae_variant,
        "norm_stats_path": str(Path(args.norm_stats_path).expanduser().resolve()),
        "norm_stats_metadata": stats.metadata,
        "center_loss_weight": args.center_loss_weight,
        "direction_loss_weight": args.direction_loss_weight,
        "gripper_loss_weight": args.gripper_loss_weight,
        "lr": args.lr,
        "warmup_steps": warmup_steps,
        "batch_size_per_gpu": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "world_size": world_size,
        "precision": (
            "fp32_master_bf16_autocast"
            if args.bf16 and device.type == "cuda"
            else "fp32"
        ),
        "arguments": dict(vars(args)),
        "runtime": runtime_config,
    }
    if rank == 0:
        source_config_path = Path(args.config).expanduser().resolve()
        with source_config_path.open(encoding="utf-8") as handle:
            source_config = json.load(handle)
        copied_config_path = Path(args.output_dir) / "experiment_config.json"
        temporary_copied_config_path = copied_config_path.with_suffix(
            copied_config_path.suffix + ".tmp"
        )
        with temporary_copied_config_path.open("w", encoding="utf-8") as handle:
            json.dump(source_config, handle, indent=2)
        os.replace(temporary_copied_config_path, copied_config_path)

        config_path = Path(args.output_dir) / "training_config.json"
        temporary_config_path = config_path.with_suffix(config_path.suffix + ".tmp")
        with temporary_config_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        os.replace(temporary_config_path, config_path)
        log(f"Copied experiment configuration: {copied_config_path}", rank)
        log(f"Saved resolved training configuration: {config_path}", rank)
    wandb_run = init_wandb(args, metadata, rank)

    latest_path = str(Path(args.output_dir) / "checkpoint_latest.pt")
    _CHECKPOINT_STATE.update(
        rank=rank, is_distributed=is_distributed, path=latest_path, vae=vae,
        optimizer=optimizer, scheduler=scheduler, step=step, metadata=metadata,
    )
    if rank == 0:
        install_signal_handlers()

    store = EpisodeStore(args.episode_cache_size)
    masks = build_loss_masks(args, device)
    grad_accum = args.grad_accum_steps
    log(
        f"Effective batch={effective_batch_size}; "
        f"weights center={args.center_loss_weight} direction={args.direction_loss_weight} "
        f"gripper={args.gripper_loss_weight}", rank,
    )
    progress = tqdm(
        total=args.max_steps, initial=step, desc="LIBERO Rothko VAE", unit="step",
        dynamic_ncols=True, disable=rank != 0 or args.no_progress_bar,
    )
    running = defaultdict(float)
    log_count = 0
    started = time.time()
    cached_epoch_index = -1
    epoch_window_refs: list[WindowRef] = []
    while step < args.max_steps:
        epoch_index = step // steps_per_epoch
        step_in_epoch = step % steps_per_epoch
        if epoch_index != cached_epoch_index:
            epoch_window_refs = list(train_window_refs)
            random.Random(args.seed + epoch_index).shuffle(epoch_window_refs)
            padding = padded_windows_per_epoch - len(epoch_window_refs)
            if padding:
                epoch_window_refs.extend(epoch_window_refs[:padding])
            cached_epoch_index = epoch_index
            log(f"Starting epoch {epoch_index + 1}/{args.epochs}", rank)
        optimizer.zero_grad(set_to_none=True)
        step_values = defaultdict(float)
        for accumulation_index in range(grad_accum):
            global_micro_batch = args.batch_size * world_size
            offset = (
                step_in_epoch * effective_batch_size
                + accumulation_index * global_micro_batch
                + rank * args.batch_size
            )
            refs = epoch_window_refs[offset : offset + args.batch_size]
            windows = materialize_windows(store, refs, args.action_horizon)
            targets, _ = build_target_batch(windows, stats, device, dtype, args)
            with torch.no_grad(), autocast_context(device, args.bf16):
                latents = vae_encode(vae.model, targets, vae.scale)
            sync_context = contextlib.nullcontext()
            if is_distributed and accumulation_index < grad_accum - 1:
                sync_context = ddp_wrapper.no_sync()
            with sync_context:
                with autocast_context(device, args.bf16):
                    reconstruction = decode_fn(latents, vae.scale)
                # Accumulate reconstruction losses in fp32 while preserving
                # gradients back into the FP32 decoder master parameters.
                loss, components = compute_reconstruction_loss(
                    reconstruction.float(), targets.float(), masks, args
                )
                (loss / grad_accum).backward()
            step_values["loss"] += float(loss) / grad_accum
            for key, value in components.items():
                step_values[key] += float(value) / grad_accum
            del targets, latents, reconstruction, loss, components
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        step += 1
        completed_epochs = step / steps_per_epoch
        epoch_completed = step % steps_per_epoch == 0
        _CHECKPOINT_STATE["step"] = step
        for key, value in step_values.items():
            running[key] += value
        log_count += 1
        progress.update(1)

        if step % args.log_every == 0:
            averages = {key: value / log_count for key, value in running.items()}
            current_lr = float(optimizer.param_groups[0]["lr"])
            log(
                f"step {step}/{args.max_steps} loss={averages['loss']:.6f} "
                f"center={averages['center']:.6f} direction={averages['direction']:.6f} "
                f"gripper={averages['gripper']:.6f} lr={current_lr:.2e} "
                f"epoch={completed_epochs:.4f}/{args.epochs} "
                f"elapsed={time.time() - started:.1f}s", rank,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": averages["loss"],
                        "train/center_loss": averages["center"],
                        "train/direction_loss": averages["direction"],
                        "train/gripper_loss": averages["gripper"],
                        "train/learning_rate": current_lr,
                        "train/epoch": completed_epochs,
                    },
                    step=step,
                )
            running.clear()
            log_count = 0

        if step % args.save_every == 0 or epoch_completed or step == args.max_steps:
            if optimizer_is_sharded:
                optimizer.consolidate_state_dict(to=0)
            if rank == 0:
                save_checkpoint(latest_path, vae, optimizer, scheduler, step, metadata)
                if args.save_step_checkpoints and (
                    step % args.step_checkpoint_every == 0 or step == args.max_steps
                ):
                    save_checkpoint(
                        str(Path(args.output_dir) / f"checkpoint_step{step:06d}.pt"),
                        vae, None, scheduler, step, metadata, include_optimizer=False,
                    )
                if step % args.export_every == 0:
                    _, vae_export_name = VAE_VARIANTS[args.vae_variant]
                    periodic_export_path = str(
                        Path(args.output_dir)
                        / f"{vae_export_name}_libero_rothko_step{step:06d}.safetensors"
                    )
                    export_complete_vae(periodic_export_path, vae, step, metadata)
                    log(f"Exported complete periodic VAE: {periodic_export_path}", rank)
                if optimizer_is_sharded:
                    optimizer._all_state_dicts.clear()
            barrier(is_distributed)

        if step % args.eval_every == 0 or epoch_completed or step == args.max_steps:
            barrier(is_distributed)
            if rank == 0:
                result = evaluate(vae, store, eval_episodes, stats, args, device, dtype)
                path = Path(args.output_dir) / f"eval_step{step:06d}.json"
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2)
                log(
                    f"Eval: pos={result['future_position_mae_m']:.6f}m "
                    f"rot={result['future_rotation_mean_deg']:.3f}deg "
                    f"gripper={result['future_gripper_mae']:.6f} -> {path}", rank,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"eval/{key}": value
                            for key, value in result.items()
                            if isinstance(value, (int, float))
                        },
                        step=step,
                    )
            barrier(is_distributed)

    progress.close()
    barrier(is_distributed)
    if rank == 0:
        _, vae_export_name = VAE_VARIANTS[args.vae_variant]
        export_path = args.export_safetensors or str(
            Path(args.output_dir)
            / f"{vae_export_name}_libero_rothko_step{step:06d}.safetensors"
        )
        export_complete_vae(export_path, vae, step, metadata)
        log(f"Exported complete FastWAM-loadable VAE: {export_path}", rank)
        if wandb_run is not None:
            wandb_run.summary["export_safetensors"] = export_path
            wandb_run.finish()
    barrier(is_distributed)
    log("Training complete.", rank)
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    config_args, _ = config_parser.parse_known_args()
    config_path = Path(config_args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"VAE fine-tuning config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config_payload = json.load(handle)
    if not isinstance(config_payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-roots", nargs="+", help="Explicit dataset roots; overrides --data-root/--suites")
    parser.add_argument("--suites", nargs="+", choices=sorted(SUITE_DIRS), default=list(DEFAULT_SUITES))
    parser.add_argument("--base-vae", default=str(DEFAULT_BASE_VAE))
    parser.add_argument(
        "--vae-variant",
        choices=sorted(VAE_VARIANTS),
        default=DEFAULT_VAE_VARIANT,
    )
    parser.add_argument("--norm-stats-path", default=str(DEFAULT_NORM_STATS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--export-safetensors")

    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--focal", type=float, default=FOCAL)
    parser.add_argument("--center-scale", type=float, default=CENTER_SCALE)
    parser.add_argument("--dir-scale", type=float, default=DIR_SCALE)
    parser.add_argument("--center-frac", type=float, default=CENTER_FRAC)
    parser.add_argument("--boundary-margin", type=int, default=BOUNDARY_MARGIN)
    parser.add_argument("--outer-margin", type=int, default=OUTER_MARGIN)

    parser.add_argument("--eval-episodes-per-task", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--episode-cache-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--zero-optimizer", action="store_true", default=True)
    parser.add_argument("--no-zero-optimizer", action="store_false", dest="zero_optimizer")
    parser.add_argument("--center-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)

    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-windows", type=int, default=200)
    parser.add_argument(
        "--eval-sampling",
        choices=("random", "uniform_per_episode"),
        default="random",
        help=(
            "Validation window selection. `random` preserves the legacy global "
            "sampler; `uniform_per_episode` deterministically covers every held-out "
            "episode."
        ),
    )
    parser.add_argument(
        "--eval-windows-per-episode",
        type=int,
        default=None,
        help="Uniformly spaced validation windows per episode.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=12_345)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--export-every",
        type=int,
        default=1_000,
        help="Interval for complete FastWAM-loadable VAE safetensors exports.",
    )
    parser.add_argument("--save-step-checkpoints", action="store_true", default=True)
    parser.add_argument(
        "--no-save-step-checkpoints",
        action="store_false",
        dest="save_step_checkpoints",
    )
    parser.add_argument("--step-checkpoint-every", type=int, default=500)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 autocast compute with FP32 master weights (default).",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_false",
        dest="bf16",
        help="Use FP32 compute and FP32 master weights.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument(
        "--allow-weights-only-resume",
        action="store_true",
        help="Allow resume without optimizer state, explicitly resetting Adam.",
    )
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--log-file")
    parser.add_argument("--no-progress-bar", action="store_true")

    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    parser.add_argument("--wandb-project", default="fast-wam")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name", default="libero_rothko_vae_decoder")
    parser.add_argument("--wandb-group", default="libero_rothko_vae")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")

    configurable = {
        action.dest
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(config_payload) - configurable)
    missing = sorted(configurable - set(config_payload))
    if unknown or missing:
        raise ValueError(
            f"Invalid VAE fine-tuning config {config_path}: "
            f"unknown={unknown}, missing={missing}"
        )
    parser.set_defaults(**config_payload)
    args = parser.parse_args()
    args.config = str(Path(args.config).expanduser().resolve())

    if args.action_horizon != ACTION_HORIZON:
        parser.error(f"This script currently requires --action-horizon={ACTION_HORIZON}")
    invalid_suites = sorted(set(args.suites) - set(SUITE_DIRS))
    if invalid_suites:
        parser.error(f"Unknown LIBERO suites in config: {invalid_suites}")
    positive = (
        args.batch_size, args.grad_accum_steps, args.epochs, args.eval_windows,
        args.eval_batch_size, args.log_every, args.eval_every, args.save_every,
        args.export_every,
    )
    if min(positive) < 1:
        parser.error("Batch/step/eval/save arguments must be positive")
    if args.eval_episodes_per_task < 1:
        parser.error("--eval-episodes-per-task must be positive")
    if args.eval_sampling == "uniform_per_episode":
        if args.eval_windows_per_episode is None or args.eval_windows_per_episode < 1:
            parser.error(
                "--eval-windows-per-episode must be positive when "
                "--eval-sampling=uniform_per_episode"
            )
    if args.save_step_checkpoints and args.step_checkpoint_every % args.save_every != 0:
        parser.error("--step-checkpoint-every must be divisible by --save-every")
    if args.export_every % args.save_every != 0:
        parser.error("--export-every must be divisible by --save-every")
    if min(args.center_loss_weight, args.direction_loss_weight, args.gripper_loss_weight) < 0:
        parser.error("Loss weights must be non-negative")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
