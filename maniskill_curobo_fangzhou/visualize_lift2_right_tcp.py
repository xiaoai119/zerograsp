#!/usr/bin/env python3
"""Render a one-frame visualization of Lift2's estimated right-arm TCP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
from PIL import Image

from mani_skill.utils import sapien_utils

from .generate_lift2_curobo_config import DEFAULT_TCP_XYZ_RIGHT_LINK26
from .lift2_constants import LIFT2_REST_QPOS
from .render_lift2_seed import PickClutterYCBLift2Env  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_full_collision",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--tcp-xyz", type=float, nargs=3, default=list(DEFAULT_TCP_XYZ_RIGHT_LINK26))
    parser.add_argument("--marker-radius", type=float, default=0.008)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=[-0.10, 1.00, 0.72])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.30, 0.0, 0.03])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/lift2_right_tcp_marker_seed001"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"right_tcp_marker_seed{args.seed:03d}_{args.width}x{args.height}.png"
    metadata_path = out_dir / "metadata.json"

    env = gym.make(
        "PickClutterYCBLift2-v1",
        robot_uids=args.robot_uid,
        control_mode="pd_joint_pos",
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=80,
        human_render_camera_configs={
            "width": args.width,
            "height": args.height,
            "fov": 1.0,
            "near": 0.01,
            "far": 100,
        },
    )
    try:
        env.reset(seed=args.seed)
        raw_env = env.unwrapped
        raw_env.agent.robot.set_qpos(torch.as_tensor(LIFT2_REST_QPOS, device=raw_env.device).unsqueeze(0))
        set_render_camera(env, list(args.camera_eye), list(args.camera_target))

        link_map = {str(link.name): link for link in raw_env.agent.robot.get_links()}
        right_link26 = link_map["right_link26"]
        tcp_world = transform_link_point(right_link26, np.asarray(args.tcp_xyz, dtype=np.float64))
        add_tcp_marker(raw_env.scene, tcp_world, radius=float(args.marker_radius))
        for _ in range(5):
            env.step(LIFT2_REST_QPOS)
        frame = normalize_rgb(env.render())
        Image.fromarray(frame).save(image_path)
        metadata = {
            "seed": args.seed,
            "tcp_parent_link": "right_link26",
            "tcp_xyz_right_link26": [float(v) for v in args.tcp_xyz],
            "tcp_world": tcp_world.tolist(),
            "robot_uid": args.robot_uid,
            "marker_radius": float(args.marker_radius),
            "image": str(image_path),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()

    print(json.dumps({"image": str(image_path), "metadata": str(metadata_path)}, ensure_ascii=False, indent=2))
    return 0


def add_tcp_marker(scene: Any, tcp_world: np.ndarray, *, radius: float) -> None:
    material = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 1.0, 0.98])
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        radius=radius,
        material=material,
    )
    builder.set_initial_pose(sapien.Pose(p=tcp_world.tolist()))
    builder.build_kinematic(name="lift2_right_tcp_marker")


def transform_link_point(link: Any, point_link: np.ndarray) -> np.ndarray:
    world_from_link = pose_to_matrix(getattr(link, "pose", None), f"{link.name} pose")
    return (world_from_link @ np.array([*point_link, 1.0], dtype=np.float64))[:3]


def pose_to_matrix(pose: Any, name: str) -> np.ndarray:
    raw = getattr(pose, "raw_pose", None)
    if raw is None:
        raise ValueError(f"{name} has no raw_pose")
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    position = raw[:3]
    quat_wxyz = raw[3:7]
    return sapien.Pose(p=position, q=quat_wxyz).to_transformation_matrix()


def set_render_camera(env: Any, eye: list[float], target: list[float]) -> None:
    pose = sapien_utils.look_at(eye=eye, target=target).raw_pose.detach().cpu().numpy().reshape(-1)
    camera = env.unwrapped._human_render_cameras["render_camera"].camera
    camera.set_local_pose(sapien.Pose(p=pose[:3], q=pose[3:]))


def normalize_rgb(frame: Any) -> np.ndarray:
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255.0, 0, 255)
    return np.ascontiguousarray(frame[..., :3].astype(np.uint8))


if __name__ == "__main__":
    raise SystemExit(main())
