#!/usr/bin/env python3
"""Record a high-resolution Lift2 motion video with YAML-enabled debug spheres."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import sapien
import torch
from PIL import Image

from mani_skill.utils import sapien_utils

from .lift2_constants import LIFT2_JOINT_NAMES, LIFT2_REST_QPOS
from .render_lift2_seed import PickClutterYCBLift2Env  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_collision_spheres_debug",
    )
    parser.add_argument("--camera-eye", type=float, nargs=3, default=[-0.10, 1.00, 0.72])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.30, 0.0, 0.03])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/lift2_motion_seed001"),
    )
    return parser.parse_args()


def normalize_rgb(frame: Any) -> np.ndarray:
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255.0, 0, 255)
    return np.ascontiguousarray(frame[..., :3].astype(np.uint8))


def motion_qpos(phase: float) -> np.ndarray:
    """Smooth, conservative two-arm swing with grippers opening early."""

    q = LIFT2_REST_QPOS.astype(np.float32).copy()
    angle = 2.0 * np.pi * phase
    open_alpha = smoothstep(min(phase / 0.35, 1.0))
    gripper = 0.002 + 0.040 * open_alpha

    q[4] = 0.35 * np.sin(angle)
    q[5] = -0.35 * np.sin(angle)
    q[6] = 1.05 + 0.22 * np.sin(angle + 0.4)
    q[7] = 1.05 + 0.22 * np.sin(angle + 0.4)
    q[8] = 1.55 + 0.35 * np.sin(angle + 1.1)
    q[9] = 1.55 + 0.35 * np.sin(angle + 1.1)
    q[10] = 0.45 * np.sin(2.0 * angle + 0.2)
    q[11] = -0.45 * np.sin(2.0 * angle + 0.2)
    q[12] = 0.55 * np.sin(angle + 1.8)
    q[13] = -0.55 * np.sin(angle + 1.8)
    q[14] = 0.55 * np.sin(1.5 * angle)
    q[15] = -0.55 * np.sin(1.5 * angle)
    q[16] = gripper
    q[17] = gripper
    q[18] = gripper
    q[19] = gripper
    return q


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def set_render_camera(env, eye: list[float], target: list[float]) -> None:
    pose = sapien_utils.look_at(eye=eye, target=target).raw_pose.detach().cpu().numpy().reshape(-1)
    camera = env.unwrapped._human_render_cameras["render_camera"].camera
    camera.set_local_pose(sapien.Pose(p=pose[:3], q=pose[3:]))


def main() -> int:
    args = parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"lift2_collision_spheres_motion_seed{args.seed:03d}_{args.width}x{args.height}.mp4"
    poster_path = out_dir / "poster_frame.png"
    metadata_path = out_dir / "metadata.json"

    env = gym.make(
        "PickClutterYCBLift2-v1",
        robot_uids=args.robot_uid,
        control_mode="pd_joint_pos",
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=max(args.frames + 20, 200),
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
        set_render_camera(env, list(args.camera_eye), list(args.camera_target))
        active_joint_names = [joint.name for joint in raw_env.agent.robot.get_active_joints()]
        if tuple(active_joint_names) != LIFT2_JOINT_NAMES:
            raise RuntimeError(
                "Unexpected active-joint order: "
                f"expected={LIFT2_JOINT_NAMES}, actual={active_joint_names}"
            )

        with imageio.get_writer(
            video_path,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            poster_saved = False
            for frame_index in range(args.frames):
                phase = frame_index / max(args.frames - 1, 1)
                qpos = motion_qpos(phase)
                env.step(qpos)
                frame = normalize_rgb(env.render())
                if not poster_saved and frame_index >= args.frames // 3:
                    Image.fromarray(frame).save(poster_path)
                    poster_saved = True
                writer.append_data(frame)

        metadata = {
            "seed": args.seed,
            "env_id": "PickClutterYCBLift2-v1",
            "robot_uid": args.robot_uid,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "frames": args.frames,
            "camera_eye": list(args.camera_eye),
            "camera_target": list(args.camera_target),
            "active_joint_names": active_joint_names,
            "video": str(video_path),
            "poster_frame": str(poster_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finally:
        env.close()

    print(json.dumps({"video": str(video_path), "poster_frame": str(poster_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
