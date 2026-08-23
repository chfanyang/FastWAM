"""Persistent LIBERO-Plus worker: load FastWAM once, evaluate many tasks."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# These must be set before importing either LIBERO or robosuite. Keep all
# Plus-specific cache/config routing local to this entry point.
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(project_root / "data/libero_plus/.libero_config")
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(project_root / "data/libero_plus/.numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(project_root / "data/libero_plus/.matplotlib_cache")
)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from experiments.libero.eval_libero_single import (
    NumpyEncoder,
    _get_num_video_frames,
    _is_libero_rothko,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _record_prediction_videos,
    _repeat_initial_states,
    _resolve_dataset_stats_path,
    _validate_eval_runtime_cfg,
    _validate_visualize_future_video_cfg,
    run_single_episode,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    save_model_prediction_video,
    save_prediction_video,
    save_rollout_video,
)
from experiments.libero_plus.libero_plus_utils import (
    assert_libero_plus_install,
    get_libero_plus_env,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver(
    "split", lambda s, idx: s.split("/")[int(idx)], replace=True
)


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "suite" not in row or "task_id" not in row:
                raise ValueError(f"Invalid task row at {path}:{line_number}: {row}")
            tasks.append(row)
    return tasks


def _result_path(output_dir: Path, suite: str, task_id: int) -> Path:
    return output_dir / suite / "results" / f"task{task_id:04d}.json"


def _prepare_runtime(cfg: DictConfig):
    partial_state = PartialState()
    partial_state.config = cfg
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    _validate_visualize_future_video_cfg(cfg)
    action_horizon = _validate_eval_runtime_cfg(cfg)
    if int(cfg.EVALUATION.get("env_num", 1)) != 1:
        raise ValueError("LIBERO-Plus persistent workers require EVALUATION.env_num=1.")

    if _is_libero_rothko(cfg):
        with open_dict(cfg.model):
            cfg.model.skip_dit_load_from_pretrain = False
            cfg.model.pop("action_dit_pretrained_path", None)

    model_device = str(cfg.EVALUATION.device)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    if hasattr(model, "validate_dataset_stats"):
        model.validate_dataset_stats(dataset_stats_path)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using original-LIBERO dataset stats: %s", dataset_stats_path)

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    return (
        model,
        processor,
        action_horizon,
        int(video_size[1]),
        int(video_size[0]),
        model_device,
        dataset_stats_path,
    )


def _save_prediction_artifacts(
    *,
    predicted_video_dir: Path,
    clips: list[dict[str, Any]],
    task_id: int,
    success: bool,
    task_description: str,
) -> None:
    if not clips:
        return
    rollout_frames = []
    predicted_frames = []
    raymap_frames = []
    for clip in clips:
        rollout_frames.extend(clip["gt_frames"])
        predicted_frames.extend(clip["pred_frames"])
        if clip.get("pred_raymap_frames"):
            raymap_frames.extend(clip["pred_raymap_frames"])
    trial_id = f"task{task_id}_trial0"
    save_prediction_video(
        predicted_video_dir,
        rollout_frames,
        predicted_frames,
        trial_id,
        "all",
        success,
        task_description,
    )
    if raymap_frames:
        save_model_prediction_video(
            predicted_video_dir,
            raymap_frames,
            trial_id,
            "all",
            "rothko",
            success,
            task_description,
        )


def _evaluate_one_task(
    *,
    suite_name: str,
    task_id: int,
    task_suite,
    model,
    processor,
    cfg: DictConfig,
    output_dir: Path,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict[str, Any]:
    started = time.time()
    # ``run_single_episode`` selects the 400/700-step budget from this field.
    # Update it for every manifest row because one persistent worker may cover
    # several suites during the same process lifetime.
    with open_dict(cfg.EVALUATION):
        cfg.EVALUATION.task_suite_name = suite_name
    task = task_suite.get_task(task_id)
    initial_states = _repeat_initial_states(
        task_suite.get_task_init_states(task_id), 1
    )
    env, task_description = get_libero_plus_env(
        task, LIBERO_ENV_RESOLUTION, cfg.get("seed")
    )
    try:
        (
            _loop_success,
            replay_images,
            predicted_clips,
            episode_psnr,
            control_trace,
        ) = run_single_episode(
            env=env,
            initial_state=initial_states[0],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=0,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        # Plus success is checked explicitly instead of assuming that every
        # environment ``done`` has identical semantics across perturbations.
        success = bool(np.asarray(env.check_success()).reshape(-1)[0])
    finally:
        env.close()

    suite_dir = output_dir / suite_name
    if bool(cfg.EVALUATION.get("save_rollout_videos", False)):
        video_dir = suite_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        save_rollout_video(
            video_dir,
            replay_images,
            f"task{task_id}_trial0",
            success,
            task_description,
        )
    if _record_prediction_videos(cfg):
        predicted_dir = suite_dir / "predicted_videos"
        predicted_dir.mkdir(parents=True, exist_ok=True)
        _save_prediction_artifacts(
            predicted_video_dir=predicted_dir,
            clips=predicted_clips,
            task_id=task_id,
            success=success,
            task_description=task_description,
        )
    if bool(cfg.EVALUATION.get("save_control_trace", False)):
        trace_dir = suite_dir / "control_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        with (trace_dir / f"task{task_id:04d}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in control_trace:
                handle.write(json.dumps(row, cls=NumpyEncoder) + "\n")

    return {
        "task_suite": suite_name,
        "task_id": int(task_id),
        "task_description": task_description,
        "language": task_description,
        "model_prompt": DEFAULT_PROMPT.format(task=task_description),
        "problem_folder": str(task.problem_folder),
        "bddl_file": str(task.bddl_file),
        "init_states_file": str(task.init_states_file),
        "success": success,
        "successes": int(success),
        "total_episodes": 1,
        "future_video_psnr": episode_psnr,
        "duration": time.time() - started,
        "worker_id": int(cfg.WORKER.worker_id),
        "gpu_id": int(cfg.gpu_id),
    }


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="sim_libero_plus.yaml"
)
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    if cfg.WORKER.task_file is None:
        raise ValueError("WORKER.task_file must not be None.")
    assert_libero_plus_install()

    output_dir = Path(str(cfg.EVALUATION.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _read_tasks(Path(str(cfg.WORKER.task_file)))
    (
        model,
        processor,
        action_horizon,
        input_w,
        input_h,
        model_device,
        dataset_stats_path,
    ) = _prepare_runtime(cfg)

    suite_cache = {}
    benchmark_dict = benchmark.get_benchmark_dict()
    completed = 0
    errors = 0
    for row in tasks:
        suite_name = str(row["suite"])
        task_id = int(row["task_id"])
        result_path = _result_path(output_dir, suite_name, task_id)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists():
            completed += 1
            continue
        if suite_name not in suite_cache:
            with contextlib.redirect_stdout(io.StringIO()):
                suite_cache[suite_name] = benchmark_dict[suite_name]()
        try:
            result = _evaluate_one_task(
                suite_name=suite_name,
                task_id=task_id,
                task_suite=suite_cache[suite_name],
                model=model,
                processor=processor,
                cfg=cfg,
                output_dir=output_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            result["checkpoint"] = str(cfg.ckpt)
            result["dataset_stats_path"] = str(dataset_stats_path)
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, cls=NumpyEncoder)
            completed += 1
            print(
                f"[worker {cfg.WORKER.worker_id}] {suite_name}/{task_id}: "
                f"success={result['success']} duration={result['duration']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            error_path = result_path.with_suffix(".error.json")
            with error_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "task_suite": suite_name,
                        "task_id": task_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "worker_id": int(cfg.WORKER.worker_id),
                        "gpu_id": int(cfg.gpu_id),
                    },
                    handle,
                    indent=2,
                )
            logging.exception("LIBERO-Plus task failed: %s/%s", suite_name, task_id)
            if not bool(cfg.WORKER.continue_on_error):
                raise

    worker_summary = {
        "worker_id": int(cfg.WORKER.worker_id),
        "gpu_id": int(cfg.gpu_id),
        "assigned": len(tasks),
        "completed_or_skipped": completed,
        "errors": errors,
    }
    with (output_dir / f"worker{int(cfg.WORKER.worker_id):02d}_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(worker_summary, handle, indent=2)
    if errors and not bool(cfg.WORKER.continue_on_error):
        raise RuntimeError(f"Worker encountered {errors} task errors.")


if __name__ == "__main__":
    main()
