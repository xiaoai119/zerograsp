"""Run the full ManiSkill -> ZeroGrasp -> projection -> execution pipeline."""

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

from maniskill_codex.camera_views import add_camera_view_args, camera_view_cli_args


@dataclass(frozen=True)
class PipelineLayout:
    run_dir: Path
    input_dir: Path
    output_dir: Path
    logs_dir: Path
    projection_path: Path
    video_path: Path
    rl_tuning_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class PipelineStep:
    name: str
    conda_env: str
    module: str
    module_args: list[str]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one self-contained run folder containing ManiSkill RGB-D/mask input, "
            "ZeroGrasp outputs, grasp projection, and execution video."
        )
    )
    parser.add_argument("--output-root", default="maniskill_codex/runs", help="Parent directory for run folders.")
    parser.add_argument("--run-name", default=None, help="Run folder name. Defaults to a timestamp.")
    parser.add_argument("--input-dir", default=None, help="Optional existing rgb.png/depth.png/mask.png/camera.json input dir.")
    parser.add_argument("--env-id", default="PickClutterYCB-v1", help="ManiSkill environment id.")
    parser.add_argument("--seed", type=int, default=42, help="ManiSkill reset seed.")
    parser.add_argument("--camera", default="base_camera", help="ManiSkill sensor name.")
    parser.add_argument("--width", type=int, default=1280, help="ZeroGrasp sensor width.")
    parser.add_argument("--height", type=int, default=1024, help="ZeroGrasp sensor height.")
    add_camera_view_args(parser)
    parser.add_argument(
        "--mask-mode",
        choices=("task-target", "all-objects", "visible-area"),
        default="task-target",
        help=(
            "Which ManiSkill segmentation ids to pass to ZeroGrasp: task-target, "
            "all-objects, or legacy visible-area filtering."
        ),
    )
    parser.add_argument(
        "--approach-axis",
        choices=("negative-x", "positive-x", "flip-world-z"),
        default="negative-x",
        help=(
            "Which ZeroGrasp local X direction should be treated as the approach direction. "
            "negative-x preserves the previous convention; positive-x flips the full X axis; "
            "flip-world-z keeps XY and flips only the world/base Z component."
        ),
    )
    parser.add_argument("--render-width", type=int, default=1280, help="Execution video render width.")
    parser.add_argument("--render-height", type=int, default=1024, help="Execution video render height.")
    parser.add_argument("--stage-steps", type=int, default=20, help="Simulation steps per execution stage.")
    parser.add_argument("--pregrasp-max-steps", type=int, default=200, help="Maximum simulation steps for pre-grasp settling.")
    parser.add_argument("--stage-max-steps", type=int, default=80, help="Maximum simulation steps per motion stage.")
    parser.add_argument("--settle-pos-tolerance", type=float, default=0.01, help="TCP position tolerance before advancing stages.")
    parser.add_argument("--descend-settle-pos-tolerance", type=float, default=0.02, help="TCP position tolerance before closing after descend.")
    parser.add_argument("--workspace-z-min", type=float, default=0.02, help="Minimum base-frame z target.")
    parser.add_argument("--pregrasp-offset", type=float, default=0.10, help="Meters to retreat from grasp along approach.")
    parser.add_argument("--lift-offset", type=float, default=0.15, help="Meters to lift after closing.")
    parser.add_argument("--video-fps", type=int, default=20, help="Execution video FPS.")
    parser.add_argument(
        "--rl-tune",
        action="store_true",
        help="Run residual CEM policy search before final ManiSkill execution.",
    )
    parser.add_argument("--rl-iters", type=int, default=3, help="CEM iterations for --rl-tune.")
    parser.add_argument("--rl-population", type=int, default=8, help="Rollouts per CEM iteration for --rl-tune.")
    parser.add_argument("--rl-elite-fraction", type=float, default=0.25, help="Elite fraction used by CEM.")
    parser.add_argument("--rl-seed", type=int, default=None, help="Random seed for residual policy search.")
    parser.add_argument(
        "--rl-stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop residual policy search after the first successful rollout.",
    )
    parser.add_argument(
        "--rl-approach-offset-range",
        type=float,
        nargs=2,
        default=(0.0, 0.05),
        metavar=("MIN", "MAX"),
        help="Meters to search along the grasp approach direction.",
    )
    parser.add_argument(
        "--rl-tcp-x-offset-range",
        type=float,
        nargs=2,
        default=(-0.02, 0.02),
        metavar=("MIN", "MAX"),
        help="Meters to search along the TCP local x axis.",
    )
    parser.add_argument(
        "--rl-tcp-y-offset-range",
        type=float,
        nargs=2,
        default=(-0.02, 0.02),
        metavar=("MIN", "MAX"),
        help="Meters to search along the TCP local y/gripper-width axis.",
    )
    parser.add_argument(
        "--rl-roll-delta-range",
        type=float,
        nargs=2,
        default=(-0.35, 0.35),
        metavar=("MIN", "MAX"),
        help="Radians to search around the TCP local approach axis.",
    )
    parser.add_argument(
        "--rl-gripper-open-range",
        type=float,
        nargs=2,
        default=(0.5, 1.0),
        metavar=("MIN", "MAX"),
        help="Normalized ManiSkill gripper command range for open stages.",
    )
    parser.add_argument(
        "--rl-gripper-closed-range",
        type=float,
        nargs=2,
        default=(-1.0, -0.2),
        metavar=("MIN", "MAX"),
        help="Normalized ManiSkill gripper command range for close/lift stages.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
        help="ZeroGrasp checkpoint path.",
    )
    parser.add_argument("--config", default="configs/maniskill.yaml", help="ZeroGrasp config path.")
    parser.add_argument("--device", default=None, help="Optional ZeroGrasp torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--enable-collision-detection",
        action="store_true",
        help="Enable ZeroGrasp collision filtering. Default keeps it off to avoid OOM on this setup.",
    )
    parser.add_argument("--no-grasp-marker", action="store_true", help="Do not draw the 3D grasp marker in the video.")
    parser.add_argument("--position-only", action="store_true", help="Ignore ZeroGrasp rotation during execution.")
    parser.add_argument("--maniskill-env-name", default="maniskill", help="Conda env name for ManiSkill commands.")
    parser.add_argument("--zerograsp-env-name", default="graduate", help="Conda env name for ZeroGrasp commands.")
    parser.add_argument("--no-conda", action="store_true", help="Run all modules with the current Python environment.")
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
        rl_tuning_path=run_dir / "rl_tuning.json",
        manifest_path=run_dir / "run_manifest.json",
    )
    layout.input_dir.mkdir(parents=True, exist_ok=True)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    return layout


def build_pipeline_steps(args: argparse.Namespace, layout: PipelineLayout) -> list[PipelineStep]:
    steps: list[PipelineStep] = []
    if args.input_dir is None:
        steps.append(
            PipelineStep(
                name="export_input",
                conda_env=args.maniskill_env_name,
                module="maniskill_codex.export_zerograsp_input",
                module_args=[
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
                ],
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
        PipelineStep(
            name="zerograsp",
            conda_env=args.zerograsp_env_name,
            module="maniskill_codex.run_zerograsp_inference",
            module_args=zg_args,
        )
    )

    steps.append(
        PipelineStep(
            name="projection",
            conda_env=args.maniskill_env_name,
            module="maniskill_codex.grasp_projection",
            module_args=[
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
            ],
        )
    )

    execute_args = [
        "--env-id",
        args.env_id,
        "--zerograsp-output",
        str(layout.output_dir),
        "--episodes",
        "1",
        "--seed",
        str(args.seed),
        "--camera",
        args.camera,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        *camera_view_cli_args(args.camera_eye, args.camera_target),
        "--render-width",
        str(args.render_width),
        "--render-height",
        str(args.render_height),
        "--stage-steps",
        str(args.stage_steps),
        "--pregrasp-max-steps",
        str(args.pregrasp_max_steps),
        "--stage-max-steps",
        str(args.stage_max_steps),
        "--settle-pos-tolerance",
        str(args.settle_pos_tolerance),
        "--descend-settle-pos-tolerance",
        str(args.descend_settle_pos_tolerance),
        "--workspace-z-min",
        str(args.workspace_z_min),
        "--pregrasp-offset",
        str(args.pregrasp_offset),
        "--lift-offset",
        str(args.lift_offset),
        "--video-fps",
        str(args.video_fps),
        "--video-out",
        str(layout.video_path),
        "--approach-axis",
        args.approach_axis,
    ]
    if not args.no_grasp_marker:
        execute_args.append("--show-grasp-marker")
    if args.position_only:
        execute_args.append("--position-only")
    if args.rl_tune:
        execute_args.extend(
            [
                "--rl-tune",
                "--rl-iters",
                str(args.rl_iters),
                "--rl-population",
                str(args.rl_population),
                "--rl-elite-fraction",
                str(args.rl_elite_fraction),
                "--rl-output",
                str(layout.rl_tuning_path),
                "--rl-approach-offset-range",
                *[str(float(v)) for v in args.rl_approach_offset_range],
                "--rl-tcp-x-offset-range",
                *[str(float(v)) for v in args.rl_tcp_x_offset_range],
                "--rl-tcp-y-offset-range",
                *[str(float(v)) for v in args.rl_tcp_y_offset_range],
                "--rl-roll-delta-range",
                *[str(float(v)) for v in args.rl_roll_delta_range],
                "--rl-gripper-open-range",
                *[str(float(v)) for v in args.rl_gripper_open_range],
                "--rl-gripper-closed-range",
                *[str(float(v)) for v in args.rl_gripper_closed_range],
            ]
        )
        if args.rl_seed is not None:
            execute_args.extend(["--rl-seed", str(args.rl_seed)])
        if not args.rl_stop_on_success:
            execute_args.append("--no-rl-stop-on-success")
    steps.append(
        PipelineStep(
            name="execute",
            conda_env=args.maniskill_env_name,
            module="maniskill_codex.execute_zerograsp_pick",
            module_args=execute_args,
        )
    )
    return steps


def conda_python_command(
    conda_env: str,
    module: str,
    module_args: list[str],
    cwd: Path,
    use_conda: bool = True,
) -> list[str]:
    module_command = " ".join(["python", "-m", module, *[shlex.quote(arg) for arg in module_args]])
    if not use_conda:
        return ["bash", "-lc", f"PYTHONPATH=. {module_command}"]

    shell = (
        'source "$(conda info --base)/etc/profile.d/conda.sh" && '
        f"conda activate {shlex.quote(conda_env)} && "
        f"cd {shlex.quote(str(cwd))} && "
        f"PYTHONPATH=. {module_command}"
    )
    return ["bash", "-lc", shell]


def run_command(step: PipelineStep, layout: PipelineLayout, cwd: Path, use_conda: bool = True) -> dict[str, object]:
    command = conda_python_command(step.conda_env, step.module, step.module_args, cwd=cwd, use_conda=use_conda)
    started = time.time()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    runtime_sec = time.time() - started
    stdout_path = layout.logs_dir / f"{step.name}.stdout.log"
    stderr_path = layout.logs_dir / f"{step.name}.stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    result = {
        "name": step.name,
        "module": step.module,
        "conda_env": step.conda_env,
        "command": command,
        "exit_code": int(proc.returncode),
        "runtime_sec": float(runtime_sec),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Pipeline step {step.name!r} failed with exit code {proc.returncode}. See {stderr_path}.")
    return result


def copy_existing_input(input_dir: str | Path, layout: PipelineLayout) -> None:
    source = Path(input_dir).expanduser().resolve()
    required = ["rgb.png", "depth.png", "mask.png", "camera.json"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source} is missing required ZeroGrasp input files: {', '.join(missing)}")
    for name in required:
        shutil.copy2(source / name, layout.input_dir / name)
    rgbd = source / "rgbd.npz"
    if rgbd.is_file():
        shutil.copy2(rgbd, layout.input_dir / "rgbd.npz")


def write_manifest(layout: PipelineLayout, args: argparse.Namespace, steps: list[dict[str, object]]) -> None:
    manifest = {
        "run_dir": str(layout.run_dir),
        "env_id": args.env_id,
        "seed": args.seed,
        "camera": args.camera,
        "camera_eye": args.camera_eye,
        "camera_target": args.camera_target,
        "mask_mode": args.mask_mode,
        "approach_axis": args.approach_axis,
        "workspace_z_min": args.workspace_z_min,
        "pregrasp_max_steps": args.pregrasp_max_steps,
        "stage_max_steps": args.stage_max_steps,
        "settle_pos_tolerance": args.settle_pos_tolerance,
        "descend_settle_pos_tolerance": args.descend_settle_pos_tolerance,
        "input_dir": str(layout.input_dir),
        "zerograsp_output_dir": str(layout.output_dir),
        "recommended_grasp": str(layout.output_dir / "recommended_grasp_top1.json"),
        "projection": str(layout.projection_path),
        "video": str(layout.video_path),
        "rl_tuning": str(layout.rl_tuning_path),
        "rl_tune": bool(args.rl_tune),
        "steps": steps,
    }
    layout.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cwd = Path(__file__).resolve().parents[1]
    layout = prepare_run_layout(args.output_root, args.run_name)
    if args.input_dir is not None:
        copy_existing_input(args.input_dir, layout)

    steps = build_pipeline_steps(args, layout)
    step_results = []
    try:
        for step in steps:
            print(f"[{step.name}] {step.module}")
            step_results.append(run_command(step, layout, cwd=cwd, use_conda=not args.no_conda))
    finally:
        write_manifest(layout, args, step_results)

    print(f"Run folder: {layout.run_dir}")
    print(f"Recommended grasp: {layout.output_dir / 'recommended_grasp_top1.json'}")
    print(f"Projection: {layout.projection_path}")
    print(f"Video: {layout.video_path}")
    if args.rl_tune:
        print(f"RL tuning: {layout.rl_tuning_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
