#!/usr/bin/env python3
"""Build M4C voxel-ESDF collision worlds in Lift2 ManiSkill scenes."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from maniskill_codex.zerograsp_inputs import (
    collect_mask_actor_records,
    extract_zerograsp_input,
    save_zerograsp_input_bundle,
)
from maniskill_curobo.scripts import execute_curobo_pick as execute
from maniskill_curobo_real.scene_builder import (
    build_oracle_instance_voxel_esdf_scene,
    write_voxel_scene_result,
)
from maniskill_curobo_real.run_world_collision_stages import write_m3_debug_views

from .export_lift2_zerograsp_input import (
    DEFAULT_CAMERA_EYE,
    DEFAULT_CAMERA_FRAME,
    DEFAULT_CAMERA_TARGET,
    build_env,
    resolve_camera_eye_target,
    settle_env,
)
from .render_lift2_seed import PickClutterYCBLift2Env  # noqa: F401


DEFAULT_LIFT2_WORKSPACE_BOUNDS = ((0.25, 1.15), (-0.75, 0.75), (0.68, 1.02))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="PickClutterYCBLift2-v1")
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_collision_spheres",
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument(
        "--camera-frame",
        choices=(DEFAULT_CAMERA_FRAME, "robot", "world"),
        default=DEFAULT_CAMERA_FRAME,
    )
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    parser.add_argument("--mask-mode", default="task-target")
    parser.add_argument("--m4-min-instance-points", type=int, default=400)
    parser.add_argument("--m4-max-obstacles", type=int, default=50)
    parser.add_argument("--m4-voxel-max-points", type=int, default=1500)
    parser.add_argument("--m4-voxel-size", type=float, default=0.01)
    parser.add_argument("--m4-voxel-dilation", type=int, default=0)
    parser.add_argument(
        "--workspace-x",
        type=float,
        nargs=2,
        default=list(DEFAULT_LIFT2_WORKSPACE_BOUNDS[0]),
    )
    parser.add_argument(
        "--workspace-y",
        type=float,
        nargs=2,
        default=list(DEFAULT_LIFT2_WORKSPACE_BOUNDS[1]),
    )
    parser.add_argument(
        "--workspace-z",
        type=float,
        nargs=2,
        default=list(DEFAULT_LIFT2_WORKSPACE_BOUNDS[2]),
    )
    parser.add_argument("--min-valid-points", type=int, default=100)
    parser.add_argument("--debug-views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-max-points", type=int, default=60000)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/m4c_lift2_collision_spheres_seed1_200"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "command.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    records_path = output_root / "records.jsonl"
    records_by_seed = load_existing_records(records_path) if args.reuse_existing else {}
    env = build_env(args)
    try:
        for index, seed in enumerate(range(args.seed_start, args.seed_end + 1), start=1):
            if args.reuse_existing and seed in records_by_seed:
                print(f"[{index}] seed {seed}: reused", flush=True)
                continue
            print(f"[{index}] seed {seed}: building Lift2 M4C scene", flush=True)
            record = run_seed(args=args, env=env, output_root=output_root, seed=seed)
            records_by_seed[seed] = record
            write_records(records_path, records_by_seed)
            write_summary(output_root / "summary.json", records_by_seed)
            print(
                f"  status={record['status']} "
                f"valid_points={record.get('valid_points')} "
                f"obstacles={record.get('n_obstacles')}",
                flush=True,
            )
    finally:
        env.close()
    return 0


def run_seed(*, args: argparse.Namespace, env: Any, output_root: Path, seed: int) -> dict[str, Any]:
    started = time.time()
    seed_dir = output_root / f"seed{seed:03d}"
    input_dir = seed_dir / "zerograsp_input"
    scene_dir = seed_dir / "real_scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    try:
        workspace_bounds = normalized_workspace_bounds(args)
        obs, _ = env.reset(seed=seed)
        obs = settle_env(env, obs, steps=args.settle_steps)
        bundle = extract_zerograsp_input(
            obs,
            env,
            args.camera,
            mask_mode=args.mask_mode,
        )
        save_zerograsp_input_bundle(bundle, input_dir)
        camera_model = execute.camera_model_matrix(env, args.camera)
        world_from_base = execute.robot_base_matrix(env)
        result = build_oracle_instance_voxel_esdf_scene(
            depth_m=bundle.depth,
            mask=bundle.mask,
            segmentation=raw_segmentation(obs, args.camera),
            object_records=collect_mask_actor_records(env, "all-objects"),
            target_records=collect_mask_actor_records(env, "task-target"),
            camera_matrix=bundle.camera_matrix,
            camera_model_matrix=camera_model,
            world_from_base_matrix=world_from_base,
            workspace_bounds=workspace_bounds,
            min_instance_points=int(args.m4_min_instance_points),
            max_obstacles=int(args.m4_max_obstacles),
            max_points_per_instance=int(args.m4_voxel_max_points),
            voxel_size=float(args.m4_voxel_size),
            dilation_voxels=int(args.m4_voxel_dilation),
        )
        valid_points = int(result.metadata.get("valid_points", 0))
        if valid_points < int(args.min_valid_points):
            raise RuntimeError(
                f"too few valid workspace points: {valid_points} < {int(args.min_valid_points)}"
            )
        scene_path = scene_dir / "curobo_scene_voxel.npz"
        metadata_path = scene_dir / "curobo_scene_metadata.json"
        write_voxel_scene_result(result, scene_path, metadata_path=metadata_path)
        debug_views_error = None
        if args.debug_views:
            try:
                write_m3_debug_views(
                    output_dir=seed_dir / "debug_views",
                    bundle=bundle,
                    segmentation=raw_segmentation(obs, args.camera),
                    object_records=collect_mask_actor_records(env, "all-objects"),
                    target_records=collect_mask_actor_records(env, "task-target"),
                    camera_model=camera_model,
                    world_from_base=world_from_base,
                    min_instance_points=int(args.m4_min_instance_points),
                    instance_padding=0.0,
                    min_cuboid_dimension=0.005,
                    max_points=int(args.debug_max_points),
                )
            except Exception:
                debug_path = seed_dir / "debug_views_error.txt"
                debug_path.write_text(traceback.format_exc(), encoding="utf-8")
                debug_views_error = str(debug_path)
        camera_eye_world, camera_target_world = resolve_camera_eye_target(args)
        return {
            "seed": int(seed),
            "status": "scene_built",
            "env_id": args.env_id,
            "robot_uid": args.robot_uid,
            "camera_frame": args.camera_frame,
            "camera_eye_world": camera_eye_world.tolist(),
            "camera_target_world": camera_target_world.tolist(),
            "workspace_bounds": [list(axis) for axis in workspace_bounds],
            "scene_model": str(scene_path),
            "metadata": str(metadata_path),
            "debug_views_error": debug_views_error,
            "runtime_sec": float(time.time() - started),
            "valid_points": valid_points,
            "target_points": int(result.metadata.get("target_points", 0)),
            "non_target_points": int(result.metadata.get("non_target_points", 0)),
            "n_obstacles": int(result.metadata.get("n_obstacles", 0)),
            "occupied_voxels": int(result.metadata.get("occupied_voxels", 0)),
            "voxel_shape": result.metadata.get("voxel_shape"),
        }
    except Exception as exc:
        error_path = seed_dir / "error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return {
            "seed": int(seed),
            "status": "failed",
            "error": repr(exc),
            "error_trace": str(error_path),
            "runtime_sec": float(time.time() - started),
        }


def normalized_workspace_bounds(
    args: argparse.Namespace,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    bounds = (
        tuple(float(v) for v in args.workspace_x),
        tuple(float(v) for v in args.workspace_y),
        tuple(float(v) for v in args.workspace_z),
    )
    for axis_name, axis_bounds in zip(("x", "y", "z"), bounds):
        if len(axis_bounds) != 2 or axis_bounds[0] >= axis_bounds[1]:
            raise ValueError(f"invalid workspace-{axis_name}: {axis_bounds}")
    return bounds


def raw_segmentation(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    sensor = obs["sensor_data"][camera_name]
    value = sensor["PositionSegmentation"]
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return arr[0, :, :, 3].astype("int32")


def load_existing_records(records_path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not records_path.is_file():
        return records
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[int(record["seed"])] = record
    return records


def write_records(records_path: Path, records_by_seed: dict[int, dict[str, Any]]) -> None:
    records_path.write_text(
        "".join(
            json.dumps(records_by_seed[seed], ensure_ascii=False, sort_keys=True) + "\n"
            for seed in sorted(records_by_seed)
        ),
        encoding="utf-8",
    )


def write_summary(summary_path: Path, records_by_seed: dict[int, dict[str, Any]]) -> None:
    records = [records_by_seed[seed] for seed in sorted(records_by_seed)]
    built = [record for record in records if record.get("status") == "scene_built"]
    failed = [record for record in records if record.get("status") != "scene_built"]
    summary = {
        "total": len(records),
        "scene_built": len(built),
        "failed": len(failed),
        "failed_seeds": [int(record["seed"]) for record in failed],
        "avg_runtime_sec": float(np.mean([record["runtime_sec"] for record in records]))
        if records
        else 0.0,
        "avg_valid_points": float(np.mean([record.get("valid_points", 0) for record in built]))
        if built
        else 0.0,
        "avg_obstacles": float(np.mean([record.get("n_obstacles", 0) for record in built]))
        if built
        else 0.0,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
