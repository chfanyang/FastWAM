"""Environment helpers used only by LIBERO-Plus evaluation."""

from __future__ import annotations

import contextlib
import io
import pathlib
from typing import Any

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def assert_libero_plus_install() -> None:
    """Fail early when the active environment imports upstream LIBERO."""
    import libero
    from libero.libero import benchmark

    module_file = getattr(libero, "__file__", None)
    if module_file is not None:
        module_path = pathlib.Path(module_file).resolve()
    else:
        namespace_paths = list(getattr(libero, "__path__", []))
        module_path = (
            pathlib.Path(namespace_paths[0]).resolve()
            if namespace_paths
            else pathlib.Path("<unknown-libero-namespace>")
        )
    benchmark_dict = benchmark.get_benchmark_dict()
    with contextlib.redirect_stdout(io.StringIO()):
        goal_suite = benchmark_dict["libero_goal"]()
    if int(goal_suite.n_tasks) <= 100:
        raise RuntimeError(
            "The active Python environment is not importing LIBERO-Plus: "
            f"libero={module_path}, libero_goal.n_tasks={goal_suite.n_tasks}. "
            "Activate the fastwam_libero_plus environment and set "
            "LIBERO_CONFIG_PATH to data/libero_plus/.libero_config."
        )


def get_libero_plus_env(task: Any, resolution: int, seed: int | None):
    """Create one Plus perturbation environment.

    LIBERO-Plus encodes perturbation metadata in the virtual BDDL path and
    parses it with string operations.  This Plus-only helper therefore passes
    a string without changing the original LIBERO helper.
    """
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=int(resolution),
        camera_widths=int(resolution),
    )
    if seed is not None:
        env.seed(int(seed))
    return env, str(task.language)
