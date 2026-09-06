"""Launch persistent multi-GPU workers for LIBERO-Plus evaluation."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# The benchmark is imported below, so route it to Plus before that import.
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(project_root / "data/libero_plus/.libero_config")
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(project_root / "data/libero_plus/.numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(project_root / "data/libero_plus/.matplotlib_cache")
)

from experiments.libero_plus.libero_plus_utils import assert_libero_plus_install
from libero.libero import benchmark


def _task_choice() -> str:
    choice = HydraConfig.get().runtime.choices.get("task")
    if not choice:
        raise ValueError("Pass task=<original-LIBERO training config>.")
    return str(choice)


def _worker_overrides() -> list[str]:
    blocked = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.output_dir",
        "EVALUATION.device",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "WORKER.task_file",
        "WORKER.worker_id",
    }
    result = []
    for override in HydraConfig.get().overrides.task:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in blocked or key.startswith("MULTIRUN.") or key.startswith("hydra."):
            continue
        result.append(override)
    return result


def _gpu_ids(cfg: DictConfig) -> list[int]:
    configured = cfg.MULTIRUN.get("gpu_ids")
    if configured is not None:
        ids = [int(x) for x in configured]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        ids = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        ids = list(range(int(cfg.MULTIRUN.num_gpus)))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid GPU list: {ids}")
    return ids


def _result_path(output_dir: Path, suite: str, task_id: int) -> Path:
    return output_dir / suite / "results" / f"task{task_id:04d}.json"


def _create_task_rows(cfg: DictConfig, output_dir: Path) -> list[dict]:
    rows = []
    benchmark_dict = benchmark.get_benchmark_dict()
    start = int(cfg.MULTIRUN.get("task_start", 0))
    end_cfg = cfg.MULTIRUN.get("task_end")
    limit_cfg = cfg.MULTIRUN.get("max_tasks_per_suite")
    for suite_name in cfg.MULTIRUN.task_suite_names:
        # Plus prints its full 2k-task permutation during construction. Keep
        # the manager log readable; worker logs still show actual task IDs.
        with contextlib.redirect_stdout(io.StringIO()):
            suite = benchmark_dict[str(suite_name)]()
        end = int(suite.n_tasks) if end_cfg is None else min(int(end_cfg), int(suite.n_tasks))
        ids = list(range(start, end))
        if limit_cfg is not None:
            ids = ids[: int(limit_cfg)]
        for task_id in ids:
            if bool(cfg.MULTIRUN.get("resume", True)) and _result_path(
                output_dir, str(suite_name), task_id
            ).exists():
                continue
            task = suite.get_task(task_id)
            rows.append(
                {
                    "suite": str(suite_name),
                    "task_id": task_id,
                    "language": str(task.language),
                    "bddl_file": str(task.bddl_file),
                }
            )
    return rows


def _assign_tasks(
    rows: list[dict],
    gpu_ids: list[int],
    suite_names: list[str],
    *,
    workers_per_gpu: int,
    assignment_mode: str,
) -> list[tuple[int, list[dict]]]:
    if workers_per_gpu <= 0:
        raise ValueError(f"workers_per_gpu must be positive, got {workers_per_gpu}.")

    # Repeat the complete GPU list so worker indices remain evenly distributed
    # across physical devices when more than one persistent worker is used.
    worker_gpu_ids = gpu_ids * workers_per_gpu
    assignments: list[list[dict]] = [[] for _ in worker_gpu_ids]

    if assignment_mode == "round_robin":
        for index, row in enumerate(rows):
            assignments[index % len(assignments)].append(row)
    elif assignment_mode == "suite_parallel":
        if not suite_names or len(suite_names) != len(set(suite_names)):
            raise ValueError(f"Invalid suite list for suite_parallel: {suite_names}")
        known_suites = set(suite_names)
        unknown_suites = sorted({str(row["suite"]) for row in rows} - known_suites)
        if unknown_suites:
            raise ValueError(f"Task rows contain suites not in task_suite_names: {unknown_suites}")

        # With four suites/four GPUs this binds one suite to each GPU. With two
        # workers per GPU, every suite gets two persistent workers on that GPU.
        for suite_index, suite_name in enumerate(suite_names):
            worker_indices = [
                index
                for index in range(len(assignments))
                if index % len(suite_names) == suite_index
            ]
            if not worker_indices:
                # Fewer workers than suites: preserve all tasks by assigning
                # additional suites to workers cyclically.
                worker_indices = [suite_index % len(assignments)]
            suite_rows = [row for row in rows if str(row["suite"]) == suite_name]
            for index, row in enumerate(suite_rows):
                assignments[worker_indices[index % len(worker_indices)]].append(row)
    else:
        raise ValueError(
            "assignment_mode must be 'round_robin' or 'suite_parallel', "
            f"got {assignment_mode!r}."
        )

    return list(zip(worker_gpu_ids, assignments))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _aggregate(output_dir: Path, expected: int) -> dict:
    result_files = sorted(output_dir.glob("*/results/task*.json"))
    error_files = sorted(output_dir.glob("*/results/task*.error.json"))
    per_suite = {}
    total_success = 0
    for path in result_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        suite = str(row["task_suite"])
        stats = per_suite.setdefault(suite, {"completed": 0, "successes": 0})
        stats["completed"] += 1
        stats["successes"] += int(bool(row["success"]))
        total_success += int(bool(row["success"]))
    for stats in per_suite.values():
        stats["success_rate"] = (
            stats["successes"] / stats["completed"] if stats["completed"] else None
        )
    summary = {
        "expected_this_launch": expected,
        "completed_results_in_output": len(result_files),
        "errors_in_output": len(error_files),
        "successes": total_success,
        "success_rate": total_success / len(result_files) if result_files else None,
        "per_suite": per_suite,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="sim_libero_plus.yaml"
)
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    assert_libero_plus_install()
    output_dir = Path(str(cfg.EVALUATION.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "manager_config.yaml")

    rows = _create_task_rows(cfg, output_dir)
    _write_jsonl(output_dir / "tasks.jsonl", rows)
    gpu_ids = _gpu_ids(cfg)
    workers_per_gpu = int(cfg.MULTIRUN.get("workers_per_gpu", 1))
    assignment_mode = str(cfg.MULTIRUN.get("assignment_mode", "round_robin"))
    worker_assignments = _assign_tasks(
        rows,
        gpu_ids,
        [str(name) for name in cfg.MULTIRUN.task_suite_names],
        workers_per_gpu=workers_per_gpu,
        assignment_mode=assignment_mode,
    )
    print(
        f"LIBERO-Plus: {len(rows)} pending tasks, "
        f"{len(worker_assignments)} persistent workers on {len(gpu_ids)} GPUs, "
        f"workers_per_gpu={workers_per_gpu}, assignment_mode={assignment_mode}, "
        f"output={output_dir}",
        flush=True,
    )
    if not rows:
        print(json.dumps(_aggregate(output_dir, 0), indent=2))
        return

    task_choice = _task_choice()
    render_gpu = int(cfg.MULTIRUN.render_gpu_id)
    render_on_worker_gpu = bool(cfg.MULTIRUN.get("render_on_worker_gpu", False))
    processes = []
    logs = []
    worker_script = project_root / "experiments/libero_plus/eval_libero_plus_worker.py"
    extra_overrides = _worker_overrides()
    for worker_id, (gpu_id, assigned) in enumerate(worker_assignments):
        if not assigned:
            continue
        task_file = output_dir / "worker_tasks" / f"worker{worker_id:02d}.jsonl"
        _write_jsonl(task_file, assigned)
        log_path = output_dir / "worker_logs" / f"worker{worker_id:02d}_gpu{gpu_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        logs.append(log_handle)

        if render_on_worker_gpu:
            # The bundled robosuite/MuJoCo EGL backend interprets
            # MUJOCO_EGL_DEVICE_ID as a physical device index. Expose only the
            # assigned physical GPU so both rendering and model inference stay
            # local to that worker's GPU.
            visible = str(gpu_id)
            model_device = "cuda:0"
            egl_device = gpu_id
        elif gpu_id == render_gpu:
            visible = str(gpu_id)
            model_device = "cuda:0"
            egl_device = render_gpu
        else:
            # The bundled old robosuite selects EGL by physical index instead
            # of CUDA's remapped logical index. Expose the render GPU first and
            # put the model on logical cuda:1.
            visible = f"{render_gpu},{gpu_id}"
            model_device = "cuda:1"
            egl_device = render_gpu
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": visible,
                "MUJOCO_EGL_DEVICE_ID": str(egl_device),
                "MUJOCO_GL": "egl",
                "TOKENIZERS_PARALLELISM": "false",
                "LIBERO_CONFIG_PATH": str(
                    project_root / "data/libero_plus/.libero_config"
                ),
                "NUMBA_CACHE_DIR": str(project_root / "data/libero_plus/.numba_cache"),
                "MPLCONFIGDIR": str(
                    project_root / "data/libero_plus/.matplotlib_cache"
                ),
            }
        )
        command = [
            sys.executable,
            str(worker_script),
            f"task={task_choice}",
            f"ckpt={cfg.ckpt}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.device={model_device}",
            f"EVALUATION.output_dir={output_dir}",
            f"WORKER.task_file={task_file}",
            f"WORKER.worker_id={worker_id}",
            *extra_overrides,
        ]
        print(
            f"worker {worker_id}: physical GPU {gpu_id}, tasks={len(assigned)}, "
            f"log={log_path}",
            flush=True,
        )
        processes.append(
            (worker_id, subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT))
        )

    return_codes = {}
    try:
        while processes:
            still_running = []
            for worker_id, process in processes:
                code = process.poll()
                if code is None:
                    still_running.append((worker_id, process))
                else:
                    return_codes[worker_id] = code
                    print(f"worker {worker_id} exited with code {code}", flush=True)
            processes = still_running
            if processes:
                time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping LIBERO-Plus workers...", flush=True)
        for _, process in processes:
            process.terminate()
        for _, process in processes:
            process.wait(timeout=30)
        raise
    finally:
        for handle in logs:
            handle.close()

    summary = _aggregate(output_dir, len(rows))
    print(json.dumps(summary, indent=2), flush=True)
    failed_workers = {key: value for key, value in return_codes.items() if value != 0}
    if failed_workers:
        raise RuntimeError(f"LIBERO-Plus workers failed: {failed_workers}")


if __name__ == "__main__":
    main()
