#!/usr/bin/env python3
"""Run translation-only versus ZeroGrasp-depth execution over a seed range."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

from maniskill_codex.zerograsp_inference_worker import (
    READY_PREFIX as ZEROGRASP_READY_PREFIX,
    RESPONSE_PREFIX as ZEROGRASP_RESPONSE_PREFIX,
)


DEFAULT_MANISKILL_PYTHON = Path("maniskill_curobo/envs/maniskill_curobo/bin/python")
DEFAULT_ZEROGRASP_PYTHON = Path("/home/openclaw-server/miniconda3/envs/graduate/bin/python")
DEFAULT_CAMERA_EYE = (-0.30, 0.0, 0.55)
DEFAULT_CAMERA_TARGET = (0.05, 0.0, 0.08)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--reuse-candidate-root",
        default=None,
        help="Optional root containing seedNNN_base/zg_input and seedNNN_base/zg_output.",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    parser.add_argument(
        "--mask-mode",
        choices=("task-target", "all-objects", "visible-area"),
        default="task-target",
    )
    parser.add_argument("--approach-axis", default="positive-x")
    parser.add_argument("--baseline-depth-scale", type=float, default=0.0)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument(
        "--include-corrected-depth",
        action="store_true",
        help=(
            "Also run a corrected_depth treatment: start from "
            "--corrected-depth-scale and progressively fall back to shallower "
            "depth scales during top-K executability selection."
        ),
    )
    parser.add_argument(
        "--corrected-depth-scale",
        type=float,
        default=None,
        help="Requested depth scale for corrected_depth. Defaults to --depth-scale.",
    )
    parser.add_argument("--grasp-depth-max-offset", type=float, default=0.04)
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=20,
        help=(
            "For each ZeroGrasp output, try this many model-ranked grasp "
            "candidates and execute the first pre/grasp-plannable candidate."
        ),
    )
    parser.add_argument(
        "--depth-auto-fallback",
        action="store_true",
        help="Enable progressive depth fallback for the depth treatment.",
    )
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--workspace-z-min", type=float, default=0.01)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument(
        "--settle-before-export-steps",
        type=int,
        default=20,
        help=(
            "Hold the robot and let initially unstable objects settle before "
            "capturing ZeroGrasp input and before planning."
        ),
    )
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-waypoints-per-stage", type=int, default=80)
    parser.add_argument("--robot-config", default="franka.yml")
    parser.add_argument("--scene-source", choices=("maniskill", "fixed"), default="maniskill")
    parser.add_argument("--scene-min-cuboid-dimension", type=float, default=0.005)
    parser.add_argument("--scene-model", default="collision_test.yml")
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
    )
    parser.add_argument("--config", default="configs/maniskill.yaml")
    parser.add_argument("--maniskill-python", default=str(DEFAULT_MANISKILL_PYTHON))
    parser.add_argument("--zerograsp-python", default=str(DEFAULT_ZEROGRASP_PYTHON))
    parser.add_argument(
        "--persistent-zerograsp-worker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep one ZeroGrasp model loaded across generated candidates. "
            "Disable with --no-persistent-zerograsp-worker."
        ),
    )
    parser.add_argument(
        "--zerograsp-worker-timeout-sec",
        type=float,
        default=300.0,
        help="Maximum wait for a persistent ZeroGrasp worker response.",
    )
    parser.add_argument(
        "--persistent-worker",
        action="store_true",
        help=(
            "Reuse one PickSingle ManiSkill environment and one warmed cuRobo planner "
            "across all executions. Requires --reuse-candidate-root."
        ),
    )
    parser.add_argument("--persistent-child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    if args.seed_start > args.seed_end:
        raise ValueError("--seed-start must not exceed --seed-end.")
    if args.persistent_worker and not args.persistent_child:
        validate_persistent_worker_args(args)
        command = [
            str(Path(args.maniskill_python).expanduser()),
            "-m",
            "maniskill_curobo.scripts.run_zerograsp_depth_ab_batch",
            *raw_argv,
            "--persistent-child",
        ]
        proc = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env_with_pythonpath(),
        )
        return int(proc.returncode)

    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_root = (
        Path(args.reuse_candidate_root).expanduser().resolve()
        if args.reuse_candidate_root
        else None
    )

    records = load_existing_records(output_root) if args.reuse_existing else {}
    seeds = list(range(args.seed_start, args.seed_end + 1))
    persistent_runner = None
    persistent_export_runner = None
    zerograsp_runner = None
    if args.persistent_child:
        validate_persistent_worker_args(args)
        persistent_runner = PersistentExecutionRunner(repo_root=repo_root)
        print(
            "persistent_worker=enabled "
            "environment_reuse=true planner_reuse=true "
            "supported_env=PickSingleYCB-v1",
            flush=True,
        )
        if candidate_root is None:
            persistent_export_runner = PersistentInputExportRunner(
                repo_root=repo_root
            )
            print(
                "persistent_input_export_worker=enabled "
                "environment_reuse=true",
                flush=True,
            )
    if (
        args.persistent_zerograsp_worker
        and candidate_root is None
        and generated_candidates_are_needed(
            output_root=output_root,
            seeds=seeds,
            reuse_existing=bool(args.reuse_existing),
        )
    ):
        zerograsp_runner = PersistentZeroGraspRunner(
            repo_root=repo_root,
            python=Path(args.zerograsp_python).expanduser(),
            checkpoint=args.checkpoint,
            config=args.config,
            collision_detection=True,
            timeout_sec=float(args.zerograsp_worker_timeout_sec),
            logs_dir=output_root / "worker_logs",
        )
        print(
            "persistent_zerograsp_worker=enabled "
            f"startup_sec={zerograsp_runner.ready_report.get('startup_sec', 0.0):.3f} "
            f"pid={zerograsp_runner.ready_report.get('pid')}",
            flush=True,
        )
    try:
        for index, seed in enumerate(seeds, start=1):
            print(
                f"[{args.env_id}] seed {seed} ({index}/{len(seeds)})",
                flush=True,
            )
            record = run_seed(
                args=args,
                repo_root=repo_root,
                output_root=output_root,
                candidate_root=candidate_root,
                seed=seed,
                existing=records.get(seed),
                execute_runner=persistent_runner,
                export_runner=persistent_export_runner,
                zerograsp_runner=zerograsp_runner,
            )
            records[seed] = record
            write_results(
                output_root,
                args,
                seeds,
                records,
                persistent_zerograsp_worker_active=zerograsp_runner is not None,
                persistent_input_export_worker_active=(
                    persistent_export_runner is not None
                ),
            )
            comparison = record.get("comparison")
            if comparison:
                variant_text = " ".join(
                    f"{label}={variant.get('outcome')}"
                    for label, variant in (record.get("variants") or {}).items()
                )
                comparisons = record.get("comparisons") or {}
                change_text = " ".join(
                    f"{label}_change={value.get('change')}"
                    for label, value in comparisons.items()
                )
                print(f"  {variant_text} {change_text}", flush=True)
            else:
                print(f"  status={record['status']}", flush=True)
    finally:
        if persistent_runner is not None:
            persistent_runner.close()
        if persistent_export_runner is not None:
            persistent_export_runner.close()
        if zerograsp_runner is not None:
            zerograsp_runner.close()

    summary = write_results(
        output_root,
        args,
        seeds,
        records,
        persistent_zerograsp_worker_active=zerograsp_runner is not None,
        persistent_input_export_worker_active=(
            persistent_export_runner is not None
        ),
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2), flush=True)
    return 0


def run_seed(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    candidate_root: Path | None,
    seed: int,
    existing: dict[str, Any] | None,
    execute_runner: PersistentExecutionRunner | None = None,
    export_runner: PersistentInputExportRunner | None = None,
    zerograsp_runner: PersistentZeroGraspRunner | None = None,
) -> dict[str, Any]:
    seed_dir = output_root / f"seed{seed:03d}"
    setup_dir = seed_dir / "setup"
    logs_dir = setup_dir / "logs"
    setup_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    source_base = reused_source_base(candidate_root, seed) if candidate_root else setup_dir
    input_dir = source_base / "zg_input"
    zg_output_dir = source_base / "zg_output"
    candidate_path = zg_output_dir / "recommended_grasp_top1.json"

    setup: dict[str, Any] = {
        "source": "reused_collision_on_candidate" if candidate_root else "generated",
        "input_dir": str(input_dir),
        "zerograsp_output_dir": str(zg_output_dir),
        "candidate_path": str(candidate_path),
    }
    if candidate_root:
        if not candidate_path.is_file():
            return {
                "seed": seed,
                "status": "missing_reused_candidate",
                "setup": setup,
            }
    else:
        export_complete = input_bundle_is_valid(input_dir)
        if not (args.reuse_existing and export_complete):
            command = export_command(args, seed, input_dir)
            if export_runner is None:
                setup["export"] = run_command(
                    command,
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    name="export_input",
                )
            else:
                setup["export"] = export_runner.run(
                    command,
                    logs_dir=logs_dir,
                    name="export_input",
                )
        else:
            setup["export"] = {"exit_code": 0, "skipped": True}
        if setup["export"]["exit_code"] != 0:
            return {"seed": seed, "status": "export_failed", "setup": setup}

        if not (args.reuse_existing and candidate_path.is_file()):
            if zerograsp_runner is None:
                setup["zerograsp"] = run_command(
                    zerograsp_command(args, seed, input_dir, zg_output_dir),
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    name="zerograsp",
                )
            else:
                setup["zerograsp"] = zerograsp_runner.run(
                    input_dir=input_dir,
                    output_dir=zg_output_dir,
                    random_seed=seed,
                    logs_dir=logs_dir,
                    name="zerograsp",
                )
        else:
            setup["zerograsp"] = {"exit_code": 0, "skipped": True}
        if setup["zerograsp"]["exit_code"] != 0 or not candidate_path.is_file():
            return {"seed": seed, "status": "zerograsp_failed", "setup": setup}

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    variants = {}
    for spec in variant_specs(args):
        label = spec["label"]
        scale = float(spec["scale"])
        run_dir = seed_dir / label
        if args.reuse_existing and variant_is_complete(
            existing,
            label=label,
            scale=scale,
            run_dir=run_dir,
        ):
            variants[label] = summarize_run(run_dir, reused=True)
            continue
        command = execute_command(
            args,
            seed=seed,
            zg_output_dir=zg_output_dir,
            run_dir=run_dir,
            depth_scale=scale,
            depth_auto_fallback=bool(spec["depth_auto_fallback"]),
        )
        if execute_runner is None:
            result = run_command(
                command,
                cwd=repo_root,
                logs_dir=run_dir / "logs",
                name="execute",
            )
        else:
            result = execute_runner.run(
                command,
                logs_dir=run_dir / "logs",
                name="execute",
            )
        variants[label] = summarize_run(run_dir, command=result)

    return {
        "seed": seed,
        "status": "complete",
        "setup": setup,
        "candidate": {
            "score": candidate.get("score"),
            "width_m": candidate.get("width_m"),
            "depth_m": candidate.get("depth_m"),
            "object_id": candidate.get("object_id"),
            "source_file": candidate.get("source_file"),
        },
        "variants": variants,
        "comparison": compare_variants(variants["baseline"], variants["depth"]),
        "comparisons": {
            label: compare_variants(variants["baseline"], variant)
            for label, variant in variants.items()
            if label != "baseline"
        },
    }


def variant_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = [
        {
            "label": "baseline",
            "scale": float(args.baseline_depth_scale),
            "depth_auto_fallback": False,
        },
        {
            "label": "depth",
            "scale": float(args.depth_scale),
            "depth_auto_fallback": bool(args.depth_auto_fallback),
        },
    ]
    if args.include_corrected_depth:
        corrected_scale = (
            float(args.corrected_depth_scale)
            if args.corrected_depth_scale is not None
            else float(args.depth_scale)
        )
        specs.append(
            {
                "label": "corrected_depth",
                "scale": corrected_scale,
                "depth_auto_fallback": True,
            }
        )
    return specs


def reused_source_base(candidate_root: Path | None, seed: int) -> Path:
    if candidate_root is None:
        raise ValueError("candidate_root is required.")
    candidates = [
        candidate_root / f"seed{seed:03d}_base",
        candidate_root / f"seed{seed:03d}" / "setup",
        candidate_root / f"seed{seed:03d}",
    ]
    for path in candidates:
        if (path / "zg_output" / "recommended_grasp_top1.json").is_file():
            return path
    return candidates[0]


def validate_persistent_worker_args(args: argparse.Namespace) -> None:
    if args.env_id != "PickSingleYCB-v1":
        raise ValueError(
            "--persistent-worker currently supports only PickSingleYCB-v1. "
            "PickClutter requires dynamic cuRobo world updates."
        )
    if args.scene_source != "maniskill":
        raise ValueError("--persistent-worker currently requires --scene-source maniskill.")


class NonClosingEnvProxy:
    """Delegate to a Gym environment while keeping it alive between episodes."""

    def __init__(self, env: Any):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._env, name, value)

    def close(self) -> None:
        return None


class PersistentExecutionRunner:
    """Run the existing single-episode entry point with cached heavy resources."""

    def __init__(
        self,
        *,
        repo_root: Path,
        execute_module: Any | None = None,
    ):
        if execute_module is None:
            from maniskill_curobo.scripts import execute_curobo_pick as execute_module

        self.repo_root = Path(repo_root).resolve()
        self.execute_module = execute_module
        self._original_build_env: Callable[..., Any] = execute_module.build_env
        self._original_build_planner: Callable[..., Any] = execute_module.build_planner
        self._env: Any | None = None
        self._env_proxy: NonClosingEnvProxy | None = None
        self._env_signature: tuple[Any, ...] | None = None
        self._planner: Any | None = None
        self._scene_signature: str | None = None
        self._episode_count = 0
        execute_module.build_env = self._build_env
        execute_module.build_planner = self._build_planner

    def _build_env(self, args: argparse.Namespace) -> NonClosingEnvProxy:
        signature = environment_signature(args)
        if self._env is None:
            self._env = self._original_build_env(args)
            self._env_proxy = NonClosingEnvProxy(self._env)
            self._env_signature = signature
        elif signature != self._env_signature:
            raise RuntimeError(
                "Persistent worker environment configuration changed between episodes: "
                f"expected={self._env_signature}, received={signature}"
            )
        assert self._env_proxy is not None
        return self._env_proxy

    def _build_planner(
        self,
        args: argparse.Namespace,
        scene_model: str | dict[str, Any] | None = None,
    ) -> Any:
        signature = scene_model_signature(scene_model)
        if self._planner is None:
            self._planner = self._original_build_planner(args, scene_model=scene_model)
            self._scene_signature = signature
        else:
            if signature != self._scene_signature:
                raise RuntimeError(
                    "Persistent worker cuRobo scene changed between episodes. "
                    "Dynamic worlds are not supported by this worker."
                )
            reset_seed = getattr(self._planner, "reset_seed", None)
            if callable(reset_seed):
                reset_seed()
        return self._planner

    def run(
        self,
        command: list[str],
        *,
        logs_dir: Path,
        name: str,
    ) -> dict[str, Any]:
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = logs_dir / f"{name}.stdout.log"
        stderr_path = logs_dir / f"{name}.stderr.log"
        command_path = logs_dir / f"{name}.command.sh"
        command_path.write_text(
            "PYTHONPATH=. " + shlex.join(command) + "\n",
            encoding="utf-8",
        )
        argv = in_process_execute_argv(command)
        env_was_initialized = self._env is not None
        planner_was_initialized = self._planner is not None
        started = time.time()
        exit_code = 0
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = self.execute_module.main(argv)
                exit_code = int(result or 0)
            except Exception:
                exit_code = 1
                traceback.print_exc(file=stderr)
        self._episode_count += 1
        return {
            "exit_code": exit_code,
            "runtime_sec": float(time.time() - started),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "command_path": str(command_path),
            "persistent_worker": True,
            "episode_index": self._episode_count,
            "environment_reused": env_was_initialized,
            "planner_reused": planner_was_initialized,
        }

    def close(self) -> None:
        self.execute_module.build_env = self._original_build_env
        self.execute_module.build_planner = self._original_build_planner
        if self._planner is not None:
            destroy = getattr(self._planner, "destroy", None)
            if callable(destroy):
                destroy()
            self._planner = None
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_proxy = None


class PersistentInputExportRunner:
    """Reuse one ManiSkill sensor environment for repeated RGB-D exports."""

    def __init__(
        self,
        *,
        repo_root: Path,
        export_module: Any | None = None,
    ):
        if export_module is None:
            from maniskill_codex import export_zerograsp_input as export_module

        self.repo_root = Path(repo_root).resolve()
        self.export_module = export_module
        self._original_build_env: Callable[..., Any] = export_module.build_env
        self._env: Any | None = None
        self._env_proxy: NonClosingEnvProxy | None = None
        self._env_signature: tuple[Any, ...] | None = None
        self._export_count = 0
        export_module.build_env = self._build_env

    def _build_env(
        self,
        width: int,
        height: int,
        render_width: int = 1280,
        render_height: int = 1024,
        env_id: str = "PickSingleYCB-v1",
        camera_name: str = "base_camera",
        camera_eye: Iterable[float] | None = None,
        camera_target: Iterable[float] | None = None,
        control_mode: str = "pd_ee_pose",
    ) -> NonClosingEnvProxy:
        signature = (
            int(width),
            int(height),
            int(render_width),
            int(render_height),
            str(env_id),
            str(camera_name),
            tuple(float(value) for value in camera_eye) if camera_eye else None,
            (
                tuple(float(value) for value in camera_target)
                if camera_target
                else None
            ),
            str(control_mode),
        )
        if self._env is None:
            self._env = self._original_build_env(
                width=width,
                height=height,
                render_width=render_width,
                render_height=render_height,
                env_id=env_id,
                camera_name=camera_name,
                camera_eye=camera_eye,
                camera_target=camera_target,
                control_mode=control_mode,
            )
            self._env_proxy = NonClosingEnvProxy(self._env)
            self._env_signature = signature
        elif signature != self._env_signature:
            raise RuntimeError(
                "Persistent export environment configuration changed: "
                f"expected={self._env_signature}, received={signature}"
            )
        assert self._env_proxy is not None
        return self._env_proxy

    def run(
        self,
        command: list[str],
        *,
        logs_dir: Path,
        name: str,
    ) -> dict[str, Any]:
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = logs_dir / f"{name}.stdout.log"
        stderr_path = logs_dir / f"{name}.stderr.log"
        command_path = logs_dir / f"{name}.command.sh"
        command_path.write_text(
            "PYTHONPATH=. " + shlex.join(command) + "\n",
            encoding="utf-8",
        )
        argv = in_process_export_argv(command)
        env_was_initialized = self._env is not None
        started = time.time()
        exit_code = 0
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = self.export_module.main(argv)
                exit_code = int(result or 0)
            except Exception:
                exit_code = 1
                traceback.print_exc(file=stderr)
        self._export_count += 1
        return {
            "exit_code": exit_code,
            "runtime_sec": float(time.time() - started),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "command_path": str(command_path),
            "persistent_input_export_worker": True,
            "export_index": self._export_count,
            "environment_reused": env_was_initialized,
        }

    def close(self) -> None:
        self.export_module.build_env = self._original_build_env
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_proxy = None


class PersistentZeroGraspRunner:
    """Keep ZeroGrasp and its CUDA context alive across batch items."""

    def __init__(
        self,
        *,
        repo_root: Path,
        python: Path,
        checkpoint: str,
        config: str,
        collision_detection: bool,
        timeout_sec: float,
        logs_dir: Path,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.timeout_sec = float(timeout_sec)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stderr_path = self.logs_dir / "zerograsp_worker.stderr.log"
        self.stdout_path = self.logs_dir / "zerograsp_worker.stdout.log"
        self._stderr = self.stderr_path.open("a", encoding="utf-8")
        self._stdout = self.stdout_path.open("a", encoding="utf-8")
        command = [
            str(python),
            "-m",
            "maniskill_codex.zerograsp_inference_worker",
            "--checkpoint",
            checkpoint,
            "--config",
            config,
        ]
        if collision_detection:
            command.append("--enable-collision-detection")
        self.command = command
        self.process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            env=env_with_pythonpath(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        self._request_count = 0
        self.ready_report = self._read_protocol_message(
            ZEROGRASP_READY_PREFIX,
            request_id=None,
        )

    def run(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        random_seed: int,
        logs_dir: Path,
        name: str,
    ) -> dict[str, Any]:
        logs_dir.mkdir(parents=True, exist_ok=True)
        command_path = logs_dir / f"{name}.command.sh"
        stdout_path = logs_dir / f"{name}.stdout.log"
        stderr_path = logs_dir / f"{name}.stderr.log"
        self._request_count += 1
        request_id = self._request_count
        request = {
            "request_id": request_id,
            "input_dir": str(Path(input_dir).resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "random_seed": int(random_seed),
        }
        command_path.write_text(
            "# Persistent ZeroGrasp worker request\n"
            + json.dumps(request, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        started = time.time()
        try:
            self._send(request)
            response = self._read_protocol_message(
                ZEROGRASP_RESPONSE_PREFIX,
                request_id=request_id,
            )
            exit_code = 0 if response.get("ok") else 2
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            exit_code = 1
        stdout_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text(
            (
                "Persistent worker diagnostics: "
                f"{self.stderr_path}\n"
                if exit_code
                else ""
            ),
            encoding="utf-8",
        )
        return {
            "exit_code": exit_code,
            "runtime_sec": float(time.time() - started),
            "model_runtime_sec": response.get("model_runtime_sec"),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "command_path": str(command_path),
            "persistent_zerograsp_worker": True,
            "request_id": request_id,
            "worker_pid": self.ready_report.get("pid"),
            "response": response,
        }

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"ZeroGrasp worker exited with code {self.process.returncode}"
            )
        if self.process.stdin is None:
            raise RuntimeError("ZeroGrasp worker stdin is unavailable")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _read_protocol_message(
        self,
        prefix: str,
        *,
        request_id: int | None,
    ) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("ZeroGrasp worker stdout is unavailable")
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"ZeroGrasp worker exited with code {self.process.returncode}"
                )
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select(
                [self.process.stdout],
                [],
                [],
                remaining,
            )
            if not readable:
                break
            line = self.process.stdout.readline()
            if not line:
                continue
            self._stdout.write(line)
            self._stdout.flush()
            text = line.strip()
            if not text.startswith(prefix):
                continue
            payload = json.loads(text[len(prefix) :])
            if request_id is None or payload.get("request_id") == request_id:
                return payload
        raise TimeoutError(
            f"Timed out after {self.timeout_sec:.1f}s waiting for ZeroGrasp worker"
        )

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request_count += 1
                request_id = self._request_count
                self._send({"request_id": request_id, "command": "shutdown"})
                self._read_protocol_message(
                    ZEROGRASP_RESPONSE_PREFIX,
                    request_id=request_id,
                )
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._stderr.close()
        self._stdout.close()


def generated_candidates_are_needed(
    *,
    output_root: Path,
    seeds: list[int],
    reuse_existing: bool,
) -> bool:
    if not reuse_existing:
        return bool(seeds)
    return any(
        not (
            output_root
            / f"seed{seed:03d}"
            / "setup"
            / "zg_output"
            / "recommended_grasp_top1.json"
        ).is_file()
        for seed in seeds
    )


def environment_signature(args: argparse.Namespace) -> tuple[Any, ...]:
    return (
        args.env_id,
        args.camera,
        int(args.width),
        int(args.height),
        int(args.render_width),
        int(args.render_height),
        tuple(float(value) for value in args.camera_eye),
        tuple(float(value) for value in args.camera_target),
    )


def scene_model_signature(scene_model: str | dict[str, Any] | None) -> str:
    if isinstance(scene_model, dict):
        return json.dumps(scene_model, sort_keys=True)
    if scene_model is None:
        return "none"
    path = Path(scene_model).expanduser()
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return str(scene_model)


def in_process_execute_argv(command: list[str]) -> list[str]:
    if len(command) < 3 or command[1:3] != [
        "-m",
        "maniskill_curobo.scripts.execute_curobo_pick",
    ]:
        raise ValueError(f"Unsupported persistent worker command: {command}")
    return command[3:]


def in_process_export_argv(command: list[str]) -> list[str]:
    if len(command) < 3 or command[1:3] != [
        "-m",
        "maniskill_codex.export_zerograsp_input",
    ]:
        raise ValueError(f"Unsupported persistent export command: {command}")
    return command[3:]


def export_command(args: argparse.Namespace, seed: int, input_dir: Path) -> list[str]:
    return [
        str(Path(args.maniskill_python).expanduser()),
        "-m",
        "maniskill_codex.export_zerograsp_input",
        "--env-id",
        args.env_id,
        "--seed",
        str(seed),
        "--camera",
        args.camera,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--control-mode",
        "pd_joint_pos",
        "--settle-before-export-steps",
        str(args.settle_before_export_steps),
        "--camera-eye",
        *[str(float(value)) for value in args.camera_eye],
        "--camera-target",
        *[str(float(value)) for value in args.camera_target],
        "--mask-mode",
        args.mask_mode,
        "--output-dir",
        str(input_dir),
    ]


def zerograsp_command(
    args: argparse.Namespace,
    seed: int,
    input_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        str(Path(args.zerograsp_python).expanduser()),
        "-m",
        "maniskill_codex.run_zerograsp_inference",
        "--img-path",
        str(input_dir / "rgb.png"),
        "--depth-path",
        str(input_dir / "depth.png"),
        "--mask-path",
        str(input_dir / "mask.png"),
        "--camera-info-path",
        str(input_dir / "camera.json"),
        "--output-dir",
        str(output_dir),
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--random-seed",
        str(seed),
        "--enable-collision-detection",
    ]


def execute_command(
    args: argparse.Namespace,
    *,
    seed: int,
    zg_output_dir: Path,
    run_dir: Path,
    depth_scale: float,
    depth_auto_fallback: bool = False,
) -> list[str]:
    command = [
        str(Path(args.maniskill_python).expanduser()),
        "-m",
        "maniskill_curobo.scripts.execute_curobo_pick",
        "--env-id",
        args.env_id,
        "--zerograsp-output",
        str(zg_output_dir),
        "--seed",
        str(seed),
        "--camera",
        args.camera,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--mask-mode",
        args.mask_mode,
        "--camera-eye",
        *[str(float(value)) for value in args.camera_eye],
        "--camera-target",
        *[str(float(value)) for value in args.camera_target],
        "--render-width",
        str(args.render_width),
        "--render-height",
        str(args.render_height),
        "--approach-axis",
        args.approach_axis,
        "--pregrasp-offset",
        str(args.pregrasp_offset),
        "--lift-offset",
        str(args.lift_offset),
        "--workspace-z-min",
        str(args.workspace_z_min),
        "--grasp-depth-scale",
        str(depth_scale),
        "--grasp-depth-max-offset",
        str(args.grasp_depth_max_offset),
        "--candidate-top-k",
        str(args.candidate_top_k),
        "--close-steps",
        str(args.close_steps),
        "--settle-steps",
        str(args.settle_steps),
        "--settle-before-export-steps",
        str(args.settle_before_export_steps),
        "--action-repeat",
        str(args.action_repeat),
        "--max-waypoints-per-stage",
        str(args.max_waypoints_per_stage),
        "--robot-config",
        args.robot_config,
        "--scene-source",
        args.scene_source,
        "--scene-min-cuboid-dimension",
        str(args.scene_min_cuboid_dimension),
        "--scene-model",
        args.scene_model,
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--video-fps",
        str(args.video_fps),
        "--video-out",
        str(run_dir / "execution.mp4"),
        "--output-dir",
        str(run_dir),
        "--no-grasp-marker",
    ]
    if args.no_video:
        command.append("--no-video")
    if depth_auto_fallback:
        command.append("--grasp-depth-auto-fallback")
    return command


def run_command(
    command: list[str],
    *,
    cwd: Path,
    logs_dir: Path,
    name: str,
) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.stdout.log"
    stderr_path = logs_dir / f"{name}.stderr.log"
    command_path = logs_dir / f"{name}.command.sh"
    command_path.write_text(
        "PYTHONPATH=. " + shlex.join(command) + "\n",
        encoding="utf-8",
    )
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=env_with_pythonpath(),
        )
    return {
        "exit_code": int(proc.returncode),
        "runtime_sec": float(time.time() - started),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command_path": str(command_path),
    }


def summarize_run(
    run_dir: Path,
    *,
    command: dict[str, Any] | None = None,
    reused: bool = False,
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return {
            "outcome": "missing_run_manifest",
            "object_lift_success": False,
            "run_dir": str(run_dir),
            "command": command,
            "reused": reused,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = manifest.get("object_lift_metrics") or {}
    grasp = manifest.get("grasp") or {}
    return {
        "outcome": manifest_outcome(manifest),
        "object_lift_success": bool(metrics.get("object_lift_success")),
        "execution_completed": not bool(manifest.get("failure_reason")),
        "height_delta_m": metrics.get("height_delta_m"),
        "max_height_delta_m": metrics.get("max_height_delta_m"),
        "final_object_tcp_distance_m": metrics.get("final_object_tcp_distance_m"),
        "depth_m": grasp.get("depth_m"),
        "depth_offset": grasp.get("grasp_depth_offset"),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "video_path": manifest.get("video_saved"),
        "partial_video_saved": manifest.get("partial_video_saved"),
        "command": command,
        "reused": reused,
    }


def manifest_outcome(manifest: dict[str, Any]) -> str:
    metrics = manifest.get("object_lift_metrics") or {}
    if metrics.get("object_lift_success"):
        return "success"
    reason = str(manifest.get("failure_reason") or "")
    if "stage=pre" in reason:
        return "planning_failed_pre"
    if "stage=grasp" in reason:
        return "planning_failed_grasp"
    if "stage=lift" in reason:
        return "planning_failed_lift"
    if reason:
        return "execution_failed"
    return str(metrics.get("failure_reason") or "failed")


def input_bundle_is_valid(input_dir: Path) -> bool:
    required = ("rgb.png", "depth.png", "mask.png", "camera.json", "scene.json")
    if not all((input_dir / name).is_file() for name in required):
        return False
    from PIL import Image

    with Image.open(input_dir / "mask.png") as mask:
        return mask.getbbox() is not None


def compare_variants(
    baseline: dict[str, Any],
    depth: dict[str, Any],
) -> dict[str, Any]:
    baseline_success = bool(baseline.get("object_lift_success"))
    depth_success = bool(depth.get("object_lift_success"))
    if not baseline_success and depth_success:
        change = "improved"
    elif baseline_success and not depth_success:
        change = "regressed"
    elif baseline_success:
        change = "both_success"
    else:
        change = "both_failed"
    return {
        "baseline_outcome": baseline.get("outcome"),
        "depth_outcome": depth.get("outcome"),
        "change": change,
    }


def variant_is_complete(
    existing: dict[str, Any] | None,
    *,
    label: str,
    scale: float,
    run_dir: Path,
) -> bool:
    if not existing or existing.get("status") != "complete":
        return False
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grasp = manifest.get("grasp") or {}
    selection = grasp.get("candidate_selection") or {}
    if selection.get("enabled") and selection.get("requested_depth_scale") is not None:
        configured = selection.get("requested_depth_scale")
    else:
        fallback = grasp.get("grasp_depth_auto_fallback") or {}
        configured = (
            fallback.get("requested_scale")
            if fallback.get("enabled")
            else (grasp.get("grasp_depth_offset") or {}).get("scale")
        )
    return configured is not None and abs(float(configured) - float(scale)) < 1e-9


def load_existing_records(output_root: Path) -> dict[int, dict[str, Any]]:
    path = output_root / "records.jsonl"
    if not path.is_file():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["seed"])] = record
    return records


def write_results(
    output_root: Path,
    args: argparse.Namespace,
    seeds: list[int],
    records: dict[int, dict[str, Any]],
    *,
    persistent_zerograsp_worker_active: bool = False,
    persistent_input_export_worker_active: bool = False,
) -> dict[str, Any]:
    ordered = [records[seed] for seed in seeds if seed in records]
    complete = [record for record in ordered if record.get("status") == "complete"]
    counts = {
        "requested_seeds": len(seeds),
        "processed_seeds": len(ordered),
        "complete_seeds": len(complete),
        "setup_failures": sum(record.get("status") != "complete" for record in ordered),
        "baseline_successes": sum(
            bool(record["variants"]["baseline"].get("object_lift_success"))
            for record in complete
        ),
        "depth_successes": sum(
            bool(record["variants"]["depth"].get("object_lift_success"))
            for record in complete
        ),
        "improved": sum(record["comparison"]["change"] == "improved" for record in complete),
        "regressed": sum(record["comparison"]["change"] == "regressed" for record in complete),
        "both_success": sum(record["comparison"]["change"] == "both_success" for record in complete),
        "both_failed": sum(record["comparison"]["change"] == "both_failed" for record in complete),
        "variant_successes": {
            spec["label"]: sum(
                bool(record["variants"][spec["label"]].get("object_lift_success"))
                for record in complete
                if spec["label"] in record.get("variants", {})
            )
            for spec in variant_specs(args)
        },
        "variant_outcomes": {
            spec["label"]: outcome_counts_for_variant(complete, spec["label"])
            for spec in variant_specs(args)
        },
        "comparison_by_variant": {
            spec["label"]: comparison_counts_for_variant(complete, spec["label"])
            for spec in variant_specs(args)
            if spec["label"] != "baseline"
        },
    }
    payload = {
        "env_id": args.env_id,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "baseline_depth_scale": args.baseline_depth_scale,
        "depth_scale": args.depth_scale,
        "include_corrected_depth": bool(args.include_corrected_depth),
        "corrected_depth_scale": (
            args.corrected_depth_scale
            if args.corrected_depth_scale is not None
            else args.depth_scale
        ),
        "grasp_depth_max_offset": args.grasp_depth_max_offset,
        "candidate_top_k": int(args.candidate_top_k),
        "depth_auto_fallback": bool(args.depth_auto_fallback),
        "persistent_worker": bool(args.persistent_worker),
        "persistent_zerograsp_worker_requested": bool(
            args.persistent_zerograsp_worker
        ),
        "persistent_zerograsp_worker_active": bool(
            persistent_zerograsp_worker_active
        ),
        "persistent_input_export_worker_active": bool(
            persistent_input_export_worker_active
        ),
        "collision_detection": True,
        "reuse_candidate_root": args.reuse_candidate_root,
        "video_enabled": not args.no_video,
        "camera_eye": [float(value) for value in args.camera_eye],
        "camera_target": [float(value) for value in args.camera_target],
        "mask_mode": args.mask_mode,
        "counts": counts,
        "records": ordered,
    }
    write_json_atomic(output_root / "summary.json", payload)
    records_path = output_root / "records.jsonl"
    temp_path = records_path.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for record in ordered:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(records_path)
    write_json_atomic(
        output_root / "progress.json",
        {
            "env_id": args.env_id,
            "processed": len(ordered),
            "total": len(seeds),
            "last_seed": ordered[-1]["seed"] if ordered else None,
            "counts": counts,
        },
    )
    return payload


def outcome_counts_for_variant(
    complete_records: list[dict[str, Any]],
    label: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in complete_records:
        variant = (record.get("variants") or {}).get(label)
        if not variant:
            continue
        outcome = str(variant.get("outcome"))
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def comparison_counts_for_variant(
    complete_records: list[dict[str, Any]],
    label: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in complete_records:
        comparisons = record.get("comparisons") or {}
        comparison = comparisons.get(label)
        if comparison is None and label == "depth":
            comparison = record.get("comparison")
        if not comparison:
            continue
        change = str(comparison.get("change"))
        counts[change] = counts.get(change, 0) + 1
    return counts


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def env_with_pythonpath() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "." if not current else f".:{current}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


if __name__ == "__main__":
    raise SystemExit(main())
