"""Run the full ManiSkill/ZeroGrasp pipeline over a seed range and summarize results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from maniskill_codex.camera_views import add_camera_view_args, camera_view_cli_args

SUMMARY_FIELDS = [
    "seed",
    "run_name",
    "exit_code",
    "success",
    "stage",
    "object",
    "score",
    "width_m",
    "approach_z",
    "run_dir",
    "projection",
    "video",
    "error",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run maniskill_codex.run_full_pipeline for an inclusive seed range. "
            "Each seed gets its own run folder and the batch writes TSV/JSON summaries."
        )
    )
    parser.add_argument("--seed-range", required=True, help="Inclusive seed range, e.g. 1-20 or a single seed.")
    parser.add_argument("--output-root", required=True, help="Folder where all seed run folders and summaries are written.")
    parser.add_argument("--batch-name", default=None, help="Prefix for run folders and summary files. Defaults to a timestamp.")
    parser.add_argument("--env-id", default="PickSingleYCB-v1", help="ManiSkill environment id.")
    parser.add_argument("--camera", default="base_camera", help="ManiSkill sensor name.")
    parser.add_argument("--width", type=int, default=1280, help="ZeroGrasp sensor width.")
    parser.add_argument("--height", type=int, default=1024, help="ZeroGrasp sensor height.")
    add_camera_view_args(parser)
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
    parser.add_argument("--rl-tune", action="store_true", help="Enable residual RL/CEM tuning in each seed run.")
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
    parser.add_argument("--rl-approach-offset-range", type=float, nargs=2, default=(0.0, 0.05))
    parser.add_argument("--rl-tcp-x-offset-range", type=float, nargs=2, default=(-0.02, 0.02))
    parser.add_argument("--rl-tcp-y-offset-range", type=float, nargs=2, default=(-0.02, 0.02))
    parser.add_argument("--rl-roll-delta-range", type=float, nargs=2, default=(-0.35, 0.35))
    parser.add_argument("--rl-gripper-open-range", type=float, nargs=2, default=(0.5, 1.0))
    parser.add_argument("--rl-gripper-closed-range", type=float, nargs=2, default=(-1.0, -0.2))
    parser.add_argument(
        "--mask-mode",
        choices=("task-target", "all-objects", "visible-area"),
        default="task-target",
        help="Which ManiSkill segmentation ids to pass to ZeroGrasp.",
    )
    parser.add_argument(
        "--approach-axis",
        choices=("negative-x", "positive-x", "flip-world-z"),
        default="flip-world-z",
        help="Approach convention passed to projection and execution. Defaults to flip-world-z.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
        help="ZeroGrasp checkpoint path.",
    )
    parser.add_argument("--config", default="configs/maniskill.yaml", help="ZeroGrasp config path.")
    parser.add_argument("--device", default=None, help="Optional ZeroGrasp torch device, e.g. cuda or cpu.")
    parser.add_argument("--enable-collision-detection", action="store_true", help="Enable ZeroGrasp collision filtering.")
    parser.add_argument("--no-grasp-marker", action="store_true", help="Do not draw the 3D grasp marker in videos.")
    parser.add_argument("--position-only", action="store_true", help="Ignore ZeroGrasp rotation during execution.")
    parser.add_argument("--maniskill-env-name", default="maniskill", help="Conda env name for ManiSkill commands.")
    parser.add_argument("--zerograsp-env-name", default="graduate", help="Conda env name for ZeroGrasp commands.")
    parser.add_argument("--no-conda", action="store_true", help="Run full pipeline steps in the current Python environment.")
    return parser.parse_args(argv)


def parse_seed_range(seed_range: str) -> list[int]:
    text = str(seed_range).strip()
    if re.fullmatch(r"-?\d+", text):
        return [int(text)]
    match = re.fullmatch(r"(-?\d+)\s*[-:]\s*(-?\d+)", text)
    if match is None:
        raise ValueError(f"Invalid seed range {seed_range!r}. Use forms like 1-20 or 7.")
    start = int(match.group(1))
    end = int(match.group(2))
    if end < start:
        raise ValueError(f"Seed range end must be >= start, got {seed_range!r}.")
    return list(range(start, end + 1))


def batch_name_or_default(batch_name: str | None) -> str:
    return batch_name or datetime.now().strftime("seed_batch_%Y%m%d_%H%M%S")


def run_name_for_seed(batch_name: str, seed: int) -> str:
    return f"{batch_name}_seed{seed}"


def build_pipeline_command(args: argparse.Namespace, seed: int, run_name: str) -> list[str]:
    command = [
        "python",
        "-m",
        "maniskill_codex.run_full_pipeline",
        "--output-root",
        str(args.output_root),
        "--run-name",
        run_name,
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
        "--mask-mode",
        args.mask_mode,
        "--approach-axis",
        args.approach_axis,
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--maniskill-env-name",
        args.maniskill_env_name,
        "--zerograsp-env-name",
        args.zerograsp_env_name,
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.enable_collision_detection:
        command.append("--enable-collision-detection")
    if args.no_grasp_marker:
        command.append("--no-grasp-marker")
    if args.position_only:
        command.append("--position-only")
    if args.rl_tune:
        command.extend(
            [
                "--rl-tune",
                "--rl-iters",
                str(args.rl_iters),
                "--rl-population",
                str(args.rl_population),
                "--rl-elite-fraction",
                str(args.rl_elite_fraction),
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
            command.extend(["--rl-seed", str(args.rl_seed)])
        if not args.rl_stop_on_success:
            command.append("--no-rl-stop-on-success")
    if args.no_conda:
        command.append("--no-conda")
    return command


def run_one_seed(
    args: argparse.Namespace,
    seed: int,
    batch_name: str,
    cwd: Path,
    batch_logs_dir: Path,
) -> dict[str, object]:
    run_name = run_name_for_seed(batch_name, seed)
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / run_name
    command = build_pipeline_command(args, seed, run_name)
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    runtime_sec = time.time() - started
    batch_logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = batch_logs_dir / f"{run_name}.stdout.log"
    stderr_path = batch_logs_dir / f"{run_name}.stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    record = collect_run_record(seed, run_name, run_dir, proc.returncode)
    record.update(
        {
            "runtime_sec": round(runtime_sec, 3),
            "batch_stdout_log": str(stdout_path),
            "batch_stderr_log": str(stderr_path),
            "command": " ".join(command),
        }
    )
    if proc.returncode != 0 and not record.get("error"):
        record["error"] = f"pipeline exited {proc.returncode}; see {stderr_path}"
    return record


def collect_run_record(seed: int, run_name: str, run_dir: str | Path, exit_code: int) -> dict[str, object]:
    run_dir = Path(run_dir).expanduser().resolve()
    scene = _read_json(run_dir / "zg_input" / "scene.json")
    grasp = _read_json(run_dir / "zg_output" / "recommended_grasp_top1.json")
    execute_log = _read_text(run_dir / "logs" / "execute.stdout.log")
    final = _parse_final_result(execute_log)
    approach_z = _parse_approach_z(execute_log)
    obj = {}
    if isinstance(scene.get("objects"), list) and scene["objects"]:
        obj = scene["objects"][0]
    error = ""
    if exit_code != 0:
        error = "pipeline failed"
    elif not (run_dir / "execution.mp4").is_file():
        error = "missing execution.mp4"

    return {
        "seed": int(seed),
        "run_name": run_name,
        "exit_code": int(exit_code),
        "success": final["success"],
        "stage": final["stage"],
        "object": obj.get("display_name") or obj.get("actor_name") or "",
        "score": grasp.get("score", ""),
        "width_m": grasp.get("width_m", ""),
        "approach_z": approach_z,
        "run_dir": str(run_dir),
        "projection": str(run_dir / "grasp_projection.png"),
        "video": str(run_dir / "execution.mp4"),
        "error": error,
    }


def write_summary_files(
    records: list[dict[str, object]],
    output_root: str | Path,
    batch_name: str,
) -> tuple[Path, Path]:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tsv_path = output_root / f"{batch_name}_summary.tsv"
    json_path = output_root / f"{batch_name}_summary.json"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    payload = {
        "batch_name": batch_name,
        "total": len(records),
        "success_count": sum(1 for record in records if bool(record.get("success"))),
        "command_ok_count": sum(1 for record in records if int(record.get("exit_code", 1)) == 0),
        "records": records,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return tsv_path, json_path


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = parse_seed_range(args.seed_range)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_name = batch_name_or_default(args.batch_name)
    batch_logs_dir = output_root / f"{batch_name}_batch_logs"
    cwd = Path(__file__).resolve().parents[1]

    records: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, start=1):
        run_name = run_name_for_seed(batch_name, seed)
        print(f"[{index}/{len(seeds)}] seed={seed} run={run_name}", flush=True)
        record = run_one_seed(args, seed, batch_name, cwd, batch_logs_dir)
        records.append(record)
        tsv_path, json_path = write_summary_files(records, output_root, batch_name)
        print(
            f"  exit={record['exit_code']} success={record['success']} "
            f"stage={record['stage']} object={record['object']}",
            flush=True,
        )

    tsv_path, json_path = write_summary_files(records, output_root, batch_name)
    print(f"Summary TSV: {tsv_path}")
    print(f"Summary JSON: {json_path}")
    failures = [record for record in records if int(record.get("exit_code", 1)) != 0]
    return 1 if failures else 0


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _parse_final_result(log: str) -> dict[str, object]:
    match = re.search(r"final_result=\{'success': (True|False), 'stage': '([^']+)'", log)
    if match:
        return {"success": match.group(1) == "True", "stage": match.group(2)}
    summary = re.search(r"Summary:\s+(\d+)/(\d+)\s+success", log)
    if summary:
        return {"success": int(summary.group(1)) > 0, "stage": ""}
    return {"success": False, "stage": ""}


def _parse_approach_z(log: str) -> str:
    match = re.search(r"approach_axis_base=\[([^\]]+)\]", log)
    if not match:
        return ""
    parts = match.group(1).split()
    return parts[2] if len(parts) >= 3 else ""


if __name__ == "__main__":
    raise SystemExit(main())
