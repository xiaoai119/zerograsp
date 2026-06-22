#!/usr/bin/env python3
"""Run M5 instance-aware multi-view collision-world benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from maniskill_curobo_real.capture_multiview_rgbd import (
    capture_seed_views,
    default_three_view_specs,
)
from maniskill_curobo_real.multiview_instance_scene import (
    build_multiview_instance_obb_scene,
    build_multiview_instance_voxel_esdf_scene,
    load_multiview_instances,
    write_multiview_instance_voxel_scene_result,
)
from maniskill_curobo_real.run_world_collision_stages import (
    DEFAULT_CAMERA_EYE,
    DEFAULT_CAMERA_TARGET,
    DEFAULT_CANDIDATE_ROOT,
    DEFAULT_MANISKILL_PYTHON,
    PersistentRealExecutionRunner,
    find_candidate_output,
    summarize_run,
)
from maniskill_curobo_real.scene_builder import write_scene_result


STAGES = (
    "m5-instance-obb-no-table",
    "m5-instance-esdf-no-table",
    "m5-instance-obb-table",
    "m5-instance-esdf-table",
)
STAGE_LABELS = {
    "m5-instance-obb-no-table": "m5_multiview_instance_obb_no_table",
    "m5-instance-esdf-no-table": "m5_multiview_instance_voxel_esdf_no_table",
    "m5-instance-obb-table": "m5_multiview_instance_obb_table",
    "m5-instance-esdf-table": "m5_multiview_instance_voxel_esdf_table",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="PickClutterYCB-v1")
    parser.add_argument("--seed-start", type=int, default=2)
    parser.add_argument("--seed-end", type=int, default=21)
    parser.add_argument("--stage", choices=STAGES, default="m5-instance-obb-no-table")
    parser.add_argument(
        "--output-root",
        default="maniskill_curobo_real/runs/m5_multiview_instance_pickclutter_seed2_21",
    )
    parser.add_argument("--reuse-candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    parser.add_argument("--side-yaw-deg", type=float, default=32.0)
    parser.add_argument("--side-distance", type=float, default=0.25)
    parser.add_argument("--side-height-above-target", type=float, default=0.19)
    parser.add_argument("--mask-mode", default="all-objects")
    parser.add_argument("--settle-before-capture-steps", type=int, default=20)
    parser.add_argument("--settle-before-export-steps", type=int, default=20)
    parser.add_argument("--m5-depth-min", type=float, default=0.05)
    parser.add_argument("--m5-depth-max", type=float, default=2.5)
    parser.add_argument("--m5-min-instance-points", type=int, default=400)
    parser.add_argument("--m5-max-obstacles", type=int, default=50)
    parser.add_argument("--m5-instance-padding", type=float, default=0.0)
    parser.add_argument("--m5-min-cuboid-dimension", type=float, default=0.005)
    parser.add_argument("--m5-voxel-size", type=float, default=0.01)
    parser.add_argument("--m5-voxel-dilation", type=int, default=0)
    parser.add_argument("--m5-voxel-max-points", type=int, default=1500)
    parser.add_argument("--approach-axis", default="positive-x")
    parser.add_argument("--grasp-depth-scale", type=float, default=1.0)
    parser.add_argument("--grasp-depth-max-offset", type=float, default=0.04)
    parser.add_argument("--grasp-depth-auto-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--workspace-z-min", type=float, default=0.01)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-waypoints-per-stage", type=int, default=80)
    parser.add_argument("--robot-config", default="franka.yml")
    parser.add_argument("--scene-min-cuboid-dimension", type=float, default=0.005)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--maniskill-python", default=str(DEFAULT_MANISKILL_PYTHON))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_root = Path(args.reuse_candidate_root).expanduser().resolve()
    views = default_three_view_specs(
        main_eye=args.camera_eye,
        main_target=args.camera_target,
        side_yaw_deg=float(args.side_yaw_deg),
        side_distance=float(args.side_distance),
        side_height_above_target=float(args.side_height_above_target),
    )
    records = []
    execution_runner = PersistentRealExecutionRunner(repo_root=Path(__file__).resolve().parents[1])
    try:
        for seed in range(int(args.seed_start), int(args.seed_end) + 1):
            print(f"[{stage_label(args)}] seed {seed}", flush=True)
            try:
                record = run_seed(
                    args=args,
                    seed=seed,
                    output_root=output_root,
                    candidate_root=candidate_root,
                    views=views,
                    execution_runner=execution_runner,
                )
            except Exception as exc:
                record = {
                    "stage": stage_label(args),
                    "seed": int(seed),
                    "status": "exception",
                    "outcome": "exception",
                    "object_lift_success": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)
            write_summary(output_root, args, records)
            print(
                f"  status={record.get('status')} outcome={record.get('outcome')} "
                f"lift={record.get('object_lift_success')}",
                flush=True,
            )
    finally:
        execution_runner.close()
    write_summary(output_root, args, records)
    return 0


def run_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    output_root: Path,
    candidate_root: Path,
    views,
    execution_runner: PersistentRealExecutionRunner,
) -> dict:
    run_started = time.time()
    stage = stage_label(args)
    run_dir = output_root / stage / f"seed{seed:03d}"
    manifest_path = run_dir / "run_manifest.json"
    if args.reuse_existing and manifest_path.is_file():
        return summarize_run(stage=stage, seed=seed, run_dir=run_dir, reused=True)

    zg_output = find_candidate_output(candidate_root, seed)
    if zg_output is None:
        return {
            "stage": stage,
            "seed": int(seed),
            "status": "missing_zerograsp_candidate",
            "outcome": "missing_zerograsp_candidate",
            "object_lift_success": False,
            "candidate_root": str(candidate_root),
        }

    multiview_root = output_root / "multiview_inputs"
    capture_seed_views(args=args, seed=seed, views=views, output_root=multiview_root)
    scene = build_scene(args=args, seed=seed, multiview_root=multiview_root, run_dir=run_dir)
    command = execute_command(
        args=args,
        seed=seed,
        zg_output=zg_output,
        run_dir=run_dir,
        scene_model=Path(scene["scene_model"]),
    )
    command_result = execution_runner.run(command, logs_dir=run_dir / "logs", name="execute")
    record = summarize_run(stage=stage, seed=seed, run_dir=run_dir, command=command_result)
    record["instance_scene"] = scene
    record["total_runtime_sec"] = float(time.time() - run_started)
    return record


def build_scene(
    *,
    args: argparse.Namespace,
    seed: int,
    multiview_root: Path,
    run_dir: Path,
) -> dict:
    instances, input_metadata = load_multiview_instances(
        input_root=multiview_root,
        seed=seed,
        depth_min=float(args.m5_depth_min),
        depth_max=float(args.m5_depth_max),
    )
    scene_dir = run_dir / "real_scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = scene_dir / "curobo_scene_metadata.json"
    started = time.time()
    include_table = args.stage in {"m5-instance-obb-table", "m5-instance-esdf-table"}
    if args.stage in {"m5-instance-obb-no-table", "m5-instance-obb-table"}:
        result = build_multiview_instance_obb_scene(
            instances=instances,
            input_metadata=input_metadata,
            min_instance_points=int(args.m5_min_instance_points),
            instance_padding=float(args.m5_instance_padding),
            min_cuboid_dimension=float(args.m5_min_cuboid_dimension),
            max_obstacles=int(args.m5_max_obstacles),
            include_table=include_table,
        )
        scene_path = scene_dir / "curobo_scene.yml"
        write_scene_result(result, scene_path, metadata_path=metadata_path)
    elif args.stage in {"m5-instance-esdf-no-table", "m5-instance-esdf-table"}:
        result = build_multiview_instance_voxel_esdf_scene(
            instances=instances,
            input_metadata=input_metadata,
            min_instance_points=int(args.m5_min_instance_points),
            max_obstacles=int(args.m5_max_obstacles),
            max_points_per_instance=int(args.m5_voxel_max_points),
            voxel_size=float(args.m5_voxel_size),
            dilation_voxels=int(args.m5_voxel_dilation),
            include_table=include_table,
        )
        scene_path = scene_dir / "curobo_scene_voxel.npz"
        write_multiview_instance_voxel_scene_result(
            result,
            scene_path,
            metadata_path=metadata_path,
        )
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")
    return {
        "status": "ok",
        "stage": stage_label(args),
        "seed": int(seed),
        "scene_model": str(scene_path),
        "metadata": str(metadata_path),
        "runtime_sec": float(time.time() - started),
        "n_instances_seen": len(instances),
        "n_pointcloud_obstacles": int(result.metadata.get("n_pointcloud_obstacles", 0)),
        "table_included": bool(result.metadata.get("table_included", False)),
    }


def execute_command(
    *,
    args: argparse.Namespace,
    seed: int,
    zg_output: Path,
    run_dir: Path,
    scene_model: Path,
) -> list[str]:
    command = [
        str(Path(args.maniskill_python).expanduser()),
        "-m",
        "maniskill_curobo.scripts.execute_curobo_pick",
        "--env-id",
        args.env_id,
        "--zerograsp-output",
        str(zg_output),
        "--seed",
        str(seed),
        "--camera",
        args.camera,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--mask-mode",
        "task-target",
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
        str(args.grasp_depth_scale),
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
        "fixed",
        "--scene-model",
        str(scene_model),
        "--scene-min-cuboid-dimension",
        str(args.scene_min_cuboid_dimension),
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
    if args.grasp_depth_auto_fallback:
        command.append("--grasp-depth-auto-fallback")
    if not args.record_video:
        command.append("--no-video")
    return command


def write_summary(output_root: Path, args: argparse.Namespace, records: list[dict]) -> None:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("outcome") or record.get("status"))
        counts[key] = counts.get(key, 0) + 1
    payload = {
        "stage": stage_label(args),
        "env_id": args.env_id,
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_end),
        "counts": counts,
        "success_count": sum(1 for record in records if record.get("object_lift_success")),
        "records": records,
    }
    path = output_root / f"{stage_label(args)}_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_label(args: argparse.Namespace) -> str:
    return STAGE_LABELS[str(args.stage)]


if __name__ == "__main__":
    raise SystemExit(main())
