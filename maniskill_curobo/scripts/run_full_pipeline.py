#!/usr/bin/env python3
"""Run ManiSkill -> ZeroGrasp -> projection -> cuRobo execution."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from maniskill_curobo.scripts.execute_curobo_pick import (
    DEFAULT_CAMERA_EYE,
    DEFAULT_CAMERA_TARGET,
)


@dataclass(frozen=True)
class PipelineLayout:
    run_dir: Path
    input_dir: Path
    output_dir: Path
    logs_dir: Path
    projection_path: Path
    video_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class PipelineStep:
    name: str
    runner: str
    module: str
    module_args: list[str]
    command: list[str]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one self-contained run folder containing ManiSkill RGB-D/mask input, "
            "ZeroGrasp outputs, grasp projection, and cuRobo execution video."
        )
    )
    parser.add_argument("--output-root", default="maniskill_curobo/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--input-dir", default=None, help="Optional existing ZeroGrasp input dir.")
    parser.add_argument("--env-id", default="PickSingleYCB-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    parser.add_argument(
        "--mask-mode",
        choices=("task-target", "all-objects", "visible-area"),
        default="task-target",
    )
    parser.add_argument(
        "--approach-axis",
        choices=("negative-x", "positive-x", "flip-world-z"),
        default="positive-x",
    )
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--workspace-z-min", type=float, default=0.02)
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-waypoints-per-stage", type=int, default=120)
    parser.add_argument("--robot-config", default="franka.yml")
    parser.add_argument("--scene-source", choices=("maniskill", "fixed"), default="maniskill")
    parser.add_argument("--scene-include-target-object", action="store_true")
    parser.add_argument("--scene-min-cuboid-dimension", type=float, default=0.005)
    parser.add_argument("--scene-model", default="collision_test.yml")
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
    )
    parser.add_argument("--config", default="configs/maniskill.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--enable-collision-detection",
        action="store_true",
        help="Enable ZeroGrasp collision filtering.",
    )
    parser.add_argument("--no-grasp-marker", action="store_true")
    parser.add_argument(
        "--maniskill-python",
        default=str(Path("maniskill_curobo/envs/maniskill_curobo/bin/python")),
        help="Python executable for ManiSkill + cuRobo commands.",
    )
    parser.add_argument("--zerograsp-env-name", default="graduate")
    parser.add_argument("--no-conda", action="store_true", help="Run ZeroGrasp with the current shell Python.")
    return parser.parse_args(argv)


def prepare_run_layout(output_root: str | Path, run_name: str | None = None) -> PipelineLayout:
    root = Path(output_root).expanduser().resolve()
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / name
    layout = PipelineLayout(
        run_dir=run_dir,
        input_dir=run_dir / "zg_input",
        output_dir=run_dir / "zg_output",
        logs_dir=run_dir / "logs",
        projection_path=run_dir / "grasp_projection.png",
        video_path=run_dir / "execution.mp4",
        manifest_path=run_dir / "pipeline_manifest.json",
    )
    layout.input_dir.mkdir(parents=True, exist_ok=True)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    return layout


def build_pipeline_steps(args: argparse.Namespace, layout: PipelineLayout) -> list[PipelineStep]:
    cwd = Path.cwd()
    steps: list[PipelineStep] = []
    if args.input_dir is None:
        export_args = [
            "--env-id",
            args.env_id,
            "--seed",
            str(args.seed),
            "--camera",
            args.camera,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            *camera_view_cli_args(args.camera_eye, args.camera_target),
            "--mask-mode",
            args.mask_mode,
            "--output-dir",
            str(layout.input_dir),
        ]
        steps.append(
            make_python_step(
                name="export_input",
                runner=args.maniskill_python,
                module="maniskill_codex.export_zerograsp_input",
                module_args=export_args,
                cwd=cwd,
            )
        )

    zg_args = [
        "--img-path",
        str(layout.input_dir / "rgb.png"),
        "--depth-path",
        str(layout.input_dir / "depth.png"),
        "--mask-path",
        str(layout.input_dir / "mask.png"),
        "--camera-info-path",
        str(layout.input_dir / "camera.json"),
        "--output-dir",
        str(layout.output_dir),
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
    ]
    if args.device:
        zg_args.extend(["--device", args.device])
    if args.enable_collision_detection:
        zg_args.append("--enable-collision-detection")
    steps.append(
        make_zerograsp_step(
            name="zerograsp",
            conda_env=args.zerograsp_env_name,
            module="maniskill_codex.run_zerograsp_inference",
            module_args=zg_args,
            cwd=cwd,
            no_conda=args.no_conda,
        )
    )

    projection_args = [
        "--rgb",
        str(layout.input_dir / "rgb.png"),
        "--camera",
        str(layout.input_dir / "camera.json"),
        "--grasp",
        str(layout.output_dir / "recommended_grasp_top1.json"),
        "--output",
        str(layout.projection_path),
        "--approach-axis",
        args.approach_axis,
    ]
    steps.append(
        make_python_step(
            name="projection",
            runner=args.maniskill_python,
            module="maniskill_codex.grasp_projection",
            module_args=projection_args,
            cwd=cwd,
        )
    )

    execute_args = [
        "--env-id",
        args.env_id,
        "--zerograsp-output",
        str(layout.output_dir),
        "--seed",
        str(args.seed),
        "--camera",
        args.camera,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--mask-mode",
        args.mask_mode,
        *camera_view_cli_args(args.camera_eye, args.camera_target),
        "--render-width",
        str(args.render_width),
        "--render-height",
        str(args.render_height),
        "--pregrasp-offset",
        str(args.pregrasp_offset),
        "--lift-offset",
        str(args.lift_offset),
        "--workspace-z-min",
        str(args.workspace_z_min),
        "--close-steps",
        str(args.close_steps),
        "--settle-steps",
        str(args.settle_steps),
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
        str(layout.video_path),
        "--output-dir",
        str(layout.run_dir),
        "--approach-axis",
        args.approach_axis,
    ]
    if args.no_grasp_marker:
        execute_args.append("--no-grasp-marker")
    if args.scene_include_target_object:
        execute_args.append("--scene-include-target-object")
    steps.append(
        make_python_step(
            name="execute",
            runner=args.maniskill_python,
            module="maniskill_curobo.scripts.execute_curobo_pick",
            module_args=execute_args,
            cwd=cwd,
        )
    )
    return steps


def make_python_step(
    name: str,
    runner: str,
    module: str,
    module_args: list[str],
    cwd: Path,
) -> PipelineStep:
    command = python_module_command(runner, module, module_args, cwd=cwd)
    return PipelineStep(name=name, runner=runner, module=module, module_args=module_args, command=command)


def make_zerograsp_step(
    name: str,
    conda_env: str,
    module: str,
    module_args: list[str],
    cwd: Path,
    no_conda: bool,
) -> PipelineStep:
    command = (
        python_module_command("python", module, module_args, cwd=cwd)
        if no_conda
        else conda_python_command(conda_env, module, module_args, cwd=cwd)
    )
    runner = "python" if no_conda else f"conda:{conda_env}"
    return PipelineStep(name=name, runner=runner, module=module, module_args=module_args, command=command)


def python_module_command(
    python_executable: str,
    module: str,
    module_args: list[str],
    cwd: Path,
) -> list[str]:
    parts = [
        "PYTHONPATH=.",
        shlex.quote(str(Path(python_executable).expanduser())),
        "-m",
        shlex.quote(module),
        *[shlex.quote(str(arg)) for arg in module_args],
    ]
    return ["bash", "-lc", " ".join(parts)]


def conda_python_command(
    conda_env: str,
    module: str,
    module_args: list[str],
    cwd: Path,
) -> list[str]:
    quoted_args = " ".join(shlex.quote(str(arg)) for arg in module_args)
    command = (
        'source "$(conda info --base)/etc/profile.d/conda.sh" && '
        f"conda activate {shlex.quote(conda_env)} && "
        f"cd {shlex.quote(str(cwd))} && "
        f"PYTHONPATH=. python -m {shlex.quote(module)} {quoted_args}"
    )
    return ["bash", "-lc", command]


def copy_existing_input(input_dir: str | Path, layout: PipelineLayout) -> None:
    source = Path(input_dir).expanduser().resolve()
    required = ["rgb.png", "depth.png", "mask.png", "camera.json"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source} is missing required ZeroGrasp input files: {', '.join(missing)}")
    for name in required:
        shutil.copy2(source / name, layout.input_dir / name)
    for name in ("rgbd.npz", "scene.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, layout.input_dir / name)


def run_command(step: PipelineStep, layout: PipelineLayout, cwd: Path) -> dict[str, object]:
    started = time.time()
    proc = subprocess.run(step.command, cwd=str(cwd), text=True, capture_output=True)
    runtime_sec = time.time() - started
    stdout_path = layout.logs_dir / f"{step.name}.stdout.log"
    stderr_path = layout.logs_dir / f"{step.name}.stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    result: dict[str, object] = {
        "name": step.name,
        "runner": step.runner,
        "module": step.module,
        "command": step.command,
        "exit_code": int(proc.returncode),
        "runtime_sec": float(runtime_sec),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"Pipeline step {step.name!r} failed with exit code {proc.returncode}. "
            f"See {stderr_path}."
        )
    return result


def write_manifest(layout: PipelineLayout, args: argparse.Namespace, steps: list[dict[str, object]]) -> None:
    manifest = {
        "run_dir": str(layout.run_dir),
        "env_id": args.env_id,
        "seed": args.seed,
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "camera_eye": args.camera_eye,
        "camera_target": args.camera_target,
        "mask_mode": args.mask_mode,
        "approach_axis": args.approach_axis,
        "input_dir": str(layout.input_dir),
        "zerograsp_output_dir": str(layout.output_dir),
        "recommended_grasp": str(layout.output_dir / "recommended_grasp_top1.json"),
        "projection": str(layout.projection_path),
        "video": str(layout.video_path),
        "steps": steps,
    }
    layout.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def camera_view_cli_args(camera_eye: Iterable[float], camera_target: Iterable[float]) -> list[str]:
    return [
        "--camera-eye",
        *[str(float(v)) for v in camera_eye],
        "--camera-target",
        *[str(float(v)) for v in camera_target],
    ]


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cwd = Path(__file__).resolve().parents[2]
    layout = prepare_run_layout(args.output_root, args.run_name)
    if args.input_dir is not None:
        copy_existing_input(args.input_dir, layout)

    steps = build_pipeline_steps(args, layout)
    step_results: list[dict[str, object]] = []
    try:
        for step in steps:
            print(f"[{step.name}] {step.module}")
            step_results.append(run_command(step, layout, cwd=cwd))
    finally:
        write_manifest(layout, args, step_results)

    print(f"Run folder: {layout.run_dir}")
    print(f"Recommended grasp: {layout.output_dir / 'recommended_grasp_top1.json'}")
    print(f"Projection: {layout.projection_path}")
    print(f"Video: {layout.video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
