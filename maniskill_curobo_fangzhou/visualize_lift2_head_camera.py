#!/usr/bin/env python3
"""Render Lift2's robot-mounted ZeroGrasp RGB-D camera frame from a third view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
import sapien
import torch
from PIL import Image

from mani_skill.utils import sapien_utils

from maniskill_curobo.scripts.execute_curobo_pick import add_marker_bar_actor

from .export_lift2_zerograsp_input import (
    DEFAULT_CAMERA_EYE,
    DEFAULT_CAMERA_FRAME,
    DEFAULT_CAMERA_TARGET,
    build_env,
)
from .lift2_constants import LIFT2_HEAD_CAMERA_LINK, LIFT2_REST_QPOS
from .render_lift2_seed import PickClutterYCBLift2Env, PickSingleYCBLift2Env  # noqa: F401


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--env-id",
        choices=("PickSingleYCBLift2-v1", "PickClutterYCBLift2-v1"),
        default="PickSingleYCBLift2-v1",
    )
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_full_collision",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--render-camera-eye",
        type=float,
        nargs=3,
        default=[1.05, 0.82, 0.62],
        help="Third-person render camera eye in world frame.",
    )
    parser.add_argument(
        "--render-camera-target",
        type=float,
        nargs=3,
        default=[0.42, 0.0, 0.22],
        help="Third-person render camera target in world frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/lift2_head_camera_marker_seed001"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_args = argparse.Namespace(
        seed=args.seed,
        env_id=args.env_id,
        robot_uid=args.robot_uid,
        camera="base_camera",
        width=args.width,
        height=args.height,
        settle_steps=args.settle_steps,
        camera_frame=DEFAULT_CAMERA_FRAME,
        camera_eye=list(DEFAULT_CAMERA_EYE),
        camera_target=list(DEFAULT_CAMERA_TARGET),
        mask_mode="task-target",
        output_dir=output_dir / "_unused",
    )
    env = build_env(env_args)
    image_path = output_dir / "head_camera_marker_third_view.png"
    video_path = output_dir / "head_camera_marker_third_view.mp4"
    manifest_path = output_dir / "manifest.json"
    try:
        obs, _ = env.reset(seed=args.seed)
        raw_env = env.unwrapped
        action = np.asarray(LIFT2_REST_QPOS, dtype=np.float32)
        for _ in range(max(0, int(args.settle_steps))):
            obs, _, _, _, _ = env.step(action)
        head_link = find_link(raw_env.agent.robot, LIFT2_HEAD_CAMERA_LINK)
        camera_pose = pose_to_matrix(head_link.pose, "head camera link pose")
        marker_actors = add_camera_pose_marker(raw_env.scene, camera_pose)

        set_render_camera(env, args.render_camera_eye, args.render_camera_target)
        frame = normalize_rgb(env.render())
        Image.fromarray(frame).save(image_path)

        frames = []
        base_eye = np.asarray(args.render_camera_eye, dtype=np.float64).reshape(3)
        target = np.asarray(args.render_camera_target, dtype=np.float64).reshape(3)
        relative = base_eye - target
        for index in range(max(1, int(args.frames))):
            angle = 0.10 * np.sin(2.0 * np.pi * index / max(1, int(args.frames) - 1))
            eye = target + rotate_z(relative, angle)
            set_render_camera(env, eye.tolist(), target.tolist())
            frames.append(normalize_rgb(env.render()))
        imageio.mimsave(video_path, frames, fps=int(args.fps))

        manifest = {
            "seed": int(args.seed),
            "env_id": args.env_id,
            "robot_uid": args.robot_uid,
            "camera_link": LIFT2_HEAD_CAMERA_LINK,
            "camera_position_world": camera_pose[:3, 3].tolist(),
            "camera_forward_world": camera_pose[:3, 0].tolist(),
            "camera_left_world": camera_pose[:3, 1].tolist(),
            "camera_up_world": camera_pose[:3, 2].tolist(),
            "render_camera_eye": list(args.render_camera_eye),
            "render_camera_target": list(args.render_camera_target),
            "image": str(image_path),
            "video": str(video_path),
            "marker_actor_count": len(marker_actors),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()

    print(json.dumps({"image": str(image_path), "video": str(video_path), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


def find_link(robot: Any, name: str) -> Any:
    for link in robot.get_links():
        if str(link.name) == name:
            return link
    raise RuntimeError(f"Link {name!r} not found.")


def add_camera_pose_marker(scene: Any, camera_pose_world: np.ndarray) -> list[Any]:
    center = camera_pose_world[:3, 3]
    forward = unit(camera_pose_world[:3, 0])
    left = unit(camera_pose_world[:3, 1])
    up = unit(camera_pose_world[:3, 2])
    actors = []

    material = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 0.0, 0.95])
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(pose=sapien.Pose(), radius=0.025, material=material)
    builder.set_initial_pose(sapien.Pose(p=center.tolist()))
    actors.append(builder.build_kinematic(name="head_camera_marker_center"))

    actors.append(add_marker_bar_actor(scene, center, center + forward * 0.28, 0.008, [1.0, 0.9, 0.0, 0.95], "head_camera_forward"))
    actors.append(add_marker_bar_actor(scene, center, center + left * 0.16, 0.006, [0.0, 0.25, 1.0, 0.90], "head_camera_left"))
    actors.append(add_marker_bar_actor(scene, center, center + up * 0.16, 0.006, [0.0, 1.0, 0.1, 0.90], "head_camera_up"))

    far_center = center + forward * 0.24
    corners = [
        far_center + left * 0.11 + up * 0.07,
        far_center + left * 0.11 - up * 0.07,
        far_center - left * 0.11 - up * 0.07,
        far_center - left * 0.11 + up * 0.07,
    ]
    for idx, corner in enumerate(corners):
        actors.append(add_marker_bar_actor(scene, center, corner, 0.0035, [1.0, 1.0, 0.0, 0.65], f"head_camera_frustum_ray_{idx}"))
    for idx, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1])):
        actors.append(add_marker_bar_actor(scene, a, b, 0.0035, [1.0, 1.0, 0.0, 0.65], f"head_camera_frustum_edge_{idx}"))
    return actors


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
    return frame[..., :3].astype(np.uint8)


def pose_to_matrix(pose: Any, name: str) -> np.ndarray:
    raw = getattr(pose, "raw_pose", None)
    if raw is None:
        raise ValueError(f"{name} has no raw_pose")
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 2:
        raw = raw[0]
    p = raw[:3]
    q = raw[3:7]
    return sapien.Pose(p=p, q=q).to_transformation_matrix()


def rotate_z(vector: np.ndarray, angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot @ np.asarray(vector, dtype=np.float64).reshape(3)


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return vector / norm


if __name__ == "__main__":
    raise SystemExit(main())
