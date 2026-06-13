#!/usr/bin/env python3
"""Export the RGB-D/mask bundle sent to ZeroGrasp in the Lift2 scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from transforms3d.quaternions import quat2mat

from maniskill_codex.zerograsp_inputs import (
    MASK_MODES,
    extract_zerograsp_input,
    save_zerograsp_input_bundle,
)

from .lift2_agent import LIFT2_REST_QPOS
from .render_lift2_seed import LIFT2_ROOT_POSE, PickClutterYCBLift2Env  # noqa: F401


# Lift2's head camera is fixed in the robot frame, not the world frame. The
# current viewpoint is 0.3 m in front of the root and 1.1716 m above the URDF
# root. Because the robot is yaw-rotated by pi in the scene, this resolves to
# world eye [0.406926, 0.0, 0.42] for the current seed1 placement.
DEFAULT_CAMERA_FRAME = "robot"
DEFAULT_CAMERA_EYE = (0.300000, 0.0, 1.171600)
DEFAULT_CAMERA_TARGET = (0.656926, 0.0, 0.831600)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-id", default="PickClutterYCBLift2-v1")
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_collision_spheres",
    )
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument(
        "--camera-frame",
        choices=("robot", "world"),
        default=DEFAULT_CAMERA_FRAME,
        help=(
            "Coordinate frame for --camera-eye/--camera-target. The default "
            "keeps the head camera fixed in Lift2's local robot frame."
        ),
    )
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=list(DEFAULT_CAMERA_TARGET),
    )
    parser.add_argument(
        "--mask-mode",
        choices=MASK_MODES,
        default="task-target",
        help="Which ManiSkill segmentation ids are passed to ZeroGrasp.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/m4c_lift2_seed001/zerograsp_input"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    env = build_env(args)
    out_dir = args.output_dir.expanduser().resolve()
    try:
        obs, _ = env.reset(seed=args.seed)
        obs = settle_env(env, obs, steps=args.settle_steps)
        camera_eye_world, camera_target_world = resolve_camera_eye_target(args)
        bundle = extract_zerograsp_input(
            obs,
            env,
            args.camera,
            mask_mode=args.mask_mode,
        )
        save_zerograsp_input_bundle(bundle, out_dir)
        overlay_path = out_dir / "rgb_mask_overlay.png"
        Image.fromarray(make_mask_overlay(bundle.rgb, bundle.mask)).save(overlay_path)
        scene = {
            "env_id": args.env_id,
            "seed": args.seed,
            "robot_uid": args.robot_uid,
            "camera": args.camera,
            "width": args.width,
            "height": args.height,
            "camera_frame": args.camera_frame,
            "camera_eye": list(args.camera_eye),
            "camera_target": list(args.camera_target),
            "camera_eye_world": camera_eye_world.tolist(),
            "camera_target_world": camera_target_world.tolist(),
            "lift2_root_pose": {
                "p": np.asarray(LIFT2_ROOT_POSE.p, dtype=float).reshape(3).tolist(),
                "q": np.asarray(LIFT2_ROOT_POSE.q, dtype=float).reshape(4).tolist(),
            },
            "settle_steps": args.settle_steps,
            "mask_mode": args.mask_mode,
            "n_objects": len(bundle.object_records),
            "objects": bundle.object_records,
            "rgb": str((out_dir / "rgb.png").resolve()),
            "mask_overlay": str(overlay_path.resolve()),
        }
        (out_dir / "scene.json").write_text(
            json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()

    print(json.dumps(scene, ensure_ascii=False, indent=2))
    return 0


def build_env(args: argparse.Namespace):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.render import PREBUILT_SHADER_CONFIGS, set_shader_pack
    from mani_skill.utils import sapien_utils

    camera_eye_world, camera_target_world = resolve_camera_eye_target(args)
    set_shader_pack(PREBUILT_SHADER_CONFIGS["minimal"])
    sensor_configs = {
        "width": args.width,
        "height": args.height,
        "shader_pack": "minimal",
        args.camera: {
            "pose": sapien_utils.look_at(camera_eye_world.tolist(), camera_target_world.tolist()),
            "shader_pack": "minimal",
        },
    }
    return gym.make(
        args.env_id,
        robot_uids=args.robot_uid,
        control_mode="pd_joint_pos",
        obs_mode="sensor_data",
        render_mode="rgb_array",
        max_episode_steps=1000,
        sensor_configs=sensor_configs,
        human_render_camera_configs={
            "width": args.width,
            "height": args.height,
            "shader_pack": "minimal",
        },
    )


def resolve_camera_eye_target(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    eye = np.asarray(args.camera_eye, dtype=np.float64).reshape(3)
    target = np.asarray(args.camera_target, dtype=np.float64).reshape(3)
    if args.camera_frame == "world":
        return eye, target
    root_p = np.asarray(LIFT2_ROOT_POSE.p, dtype=np.float64).reshape(3)
    root_q = np.asarray(LIFT2_ROOT_POSE.q, dtype=np.float64).reshape(4)
    root_R = quat2mat(root_q)
    return root_p + root_R @ eye, root_p + root_R @ target


def settle_env(env, obs: dict, *, steps: int) -> dict:
    action = np.asarray(LIFT2_REST_QPOS, dtype=np.float32)
    for _ in range(int(steps)):
        obs, _, _, _, _ = env.step(action)
    return obs


def make_mask_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.uint8).copy()
    foreground = np.asarray(mask) > 0
    if not np.any(foreground):
        return base
    tint = np.zeros_like(base)
    tint[..., 0] = 255
    tint[..., 1] = 64
    base[foreground] = (0.65 * base[foreground] + 0.35 * tint[foreground]).astype(np.uint8)
    return base


if __name__ == "__main__":
    raise SystemExit(main())
