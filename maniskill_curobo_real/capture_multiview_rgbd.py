#!/usr/bin/env python3
"""Capture close-range three-view RGB-D bundles for nvblox experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from maniskill_codex.execute_zerograsp_pick import build_env
from maniskill_codex.export_zerograsp_input import settle_environment
from maniskill_codex.zerograsp_inputs import (
    MASK_MODES,
    extract_zerograsp_input,
    save_zerograsp_input_bundle,
)
from maniskill_curobo.scripts.execute_curobo_pick import (
    camera_model_matrix,
    robot_base_matrix,
)


DEFAULT_CAMERA_EYE = (-0.20, 0.0, 0.27)
DEFAULT_CAMERA_TARGET = (0.05, 0.0, 0.08)


@dataclass(frozen=True)
class CameraViewSpec:
    """World-frame camera eye/target for one RGB-D capture."""

    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    role: str


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="PickSingleYCB-v1")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=20)
    parser.add_argument(
        "--output-root",
        default="maniskill_curobo_real/runs/nvblox_multiview_rgbd",
    )
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=list(DEFAULT_CAMERA_TARGET),
    )
    parser.add_argument(
        "--side-yaw-deg",
        type=float,
        default=32.0,
        help="Yaw offset for the two auxiliary close-range views.",
    )
    parser.add_argument(
        "--side-distance",
        type=float,
        default=0.25,
        help="Horizontal distance from target for auxiliary views, in meters.",
    )
    parser.add_argument(
        "--side-height-above-target",
        type=float,
        default=0.19,
        help="Auxiliary camera height above target point, in meters.",
    )
    parser.add_argument(
        "--settle-before-capture-steps",
        type=int,
        default=20,
        help="Hold still after reset before each RGB-D capture.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=MASK_MODES,
        default="all-objects",
        help="Mask mode saved for ZeroGrasp and target-exclusion guidance.",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the generated camera view specs.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    views = default_three_view_specs(
        main_eye=args.camera_eye,
        main_target=args.camera_target,
        side_yaw_deg=float(args.side_yaw_deg),
        side_distance=float(args.side_distance),
        side_height_above_target=float(args.side_height_above_target),
    )
    if args.dry_run:
        print(json.dumps([asdict(view) for view in views], indent=2))
        return 0

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in range(int(args.seed_start), int(args.seed_end) + 1):
        record = capture_seed_views(args=args, seed=seed, views=views, output_root=output_root)
        records.append(record)
        write_json(
            output_root / "multiview_capture_summary.json",
            {
                "env_id": args.env_id,
                "seed_start": int(args.seed_start),
                "seed_end": int(args.seed_end),
                "camera": args.camera,
                "views": [asdict(view) for view in views],
                "records": records,
            },
        )
        print(
            f"[capture] seed={seed} status={record['status']} "
            f"views={len(record.get('views', []))}",
            flush=True,
        )
    return 0


def default_three_view_specs(
    *,
    main_eye: Iterable[float],
    main_target: Iterable[float],
    side_yaw_deg: float = 32.0,
    side_distance: float = 0.25,
    side_height_above_target: float = 0.19,
) -> list[CameraViewSpec]:
    """Return one existing main view plus two close auxiliary side views."""

    eye = np.asarray(tuple(main_eye), dtype=np.float64).reshape(3)
    target = np.asarray(tuple(main_target), dtype=np.float64).reshape(3)
    main_xy = eye[:2] - target[:2]
    norm = float(np.linalg.norm(main_xy))
    if norm < 1e-6:
        main_dir = np.asarray([-1.0, 0.0], dtype=np.float64)
    else:
        main_dir = main_xy / norm
    distance = max(float(side_distance), 1e-3)
    height = float(side_height_above_target)
    yaw = math.radians(float(side_yaw_deg))

    def side_view(name: str, angle: float, role: str) -> CameraViewSpec:
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ],
            dtype=np.float64,
        )
        side_dir = rotation @ main_dir
        side_eye = np.array(
            [
                target[0] + distance * side_dir[0],
                target[1] + distance * side_dir[1],
                target[2] + height,
            ],
            dtype=np.float64,
        )
        return CameraViewSpec(
            name=name,
            eye=tuple(float(v) for v in side_eye),
            target=tuple(float(v) for v in target),
            role=role,
        )

    return [
        CameraViewSpec(
            name="view_0_existing_rgbd",
            eye=tuple(float(v) for v in eye),
            target=tuple(float(v) for v in target),
            role="existing_baseline_camera",
        ),
        side_view("view_1_close_left", yaw, "close_auxiliary_left"),
        side_view("view_2_close_right", -yaw, "close_auxiliary_right"),
    ]


def capture_seed_views(
    *,
    args: argparse.Namespace,
    seed: int,
    views: list[CameraViewSpec],
    output_root: Path,
) -> dict:
    seed_dir = output_root / f"seed{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_record: dict = {
        "seed": int(seed),
        "status": "ok",
        "views": [],
        "started_at": time.time(),
    }
    for view in views:
        view_dir = seed_dir / view.name
        bundle_path = view_dir / "rgbd.npz"
        metadata_path = view_dir / "view_metadata.json"
        if args.reuse_existing and bundle_path.is_file() and metadata_path.is_file():
            seed_record["views"].append(
                {
                    "name": view.name,
                    "role": view.role,
                    "output_dir": str(view_dir),
                    "reused": True,
                }
            )
            continue
        view_record = capture_one_view(args=args, seed=seed, view=view, output_dir=view_dir)
        seed_record["views"].append(view_record)
    seed_record["runtime_sec"] = float(time.time() - seed_record["started_at"])
    write_json(seed_dir / "multiview_manifest.json", seed_record)
    return seed_record


def capture_one_view(
    *,
    args: argparse.Namespace,
    seed: int,
    view: CameraViewSpec,
    output_dir: Path,
) -> dict:
    started = time.time()
    env = build_env(
        width=int(args.width),
        height=int(args.height),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        env_id=str(args.env_id),
        camera_name=str(args.camera),
        camera_eye=view.eye,
        camera_target=view.target,
        control_mode=str(args.control_mode),
    )
    try:
        obs, _ = env.reset(seed=int(seed))
        if int(args.settle_before_capture_steps) > 0:
            obs, _ = settle_environment(
                env,
                int(args.settle_before_capture_steps),
                control_mode=str(args.control_mode),
            )
        bundle = extract_zerograsp_input(
            obs,
            env,
            str(args.camera),
            mask_mode=str(args.mask_mode),
        )
        out = save_zerograsp_input_bundle(bundle, output_dir)
        world_from_camera = camera_model_matrix(env, str(args.camera))
        world_from_base = robot_base_matrix(env)
        base_from_world = np.linalg.inv(world_from_base)
        base_from_camera = base_from_world @ world_from_camera
        metadata = {
            "seed": int(seed),
            "env_id": args.env_id,
            "camera": args.camera,
            "view": asdict(view),
            "width": int(args.width),
            "height": int(args.height),
            "mask_mode": args.mask_mode,
            "settle_before_capture_steps": int(args.settle_before_capture_steps),
            "camera_matrix": np.asarray(bundle.camera_matrix, dtype=np.float64).tolist(),
            "world_from_camera": np.asarray(world_from_camera, dtype=np.float64).tolist(),
            "world_from_base": np.asarray(world_from_base, dtype=np.float64).tolist(),
            "base_from_camera": np.asarray(base_from_camera, dtype=np.float64).tolist(),
            "n_objects": len(bundle.object_records),
            "objects": bundle.object_records,
            "runtime_sec": float(time.time() - started),
        }
        write_json(out / "view_metadata.json", metadata)
        return {
            "name": view.name,
            "role": view.role,
            "output_dir": str(out),
            "rgbd": str(out / "rgbd.npz"),
            "metadata": str(out / "view_metadata.json"),
            "runtime_sec": metadata["runtime_sec"],
        }
    finally:
        env.close()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

