#!/usr/bin/env python3
"""Render Unitree H2 in one deterministic ManiSkill tabletop scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
from PIL import Image
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_clutter_ycb import PickClutterYCBEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from .h2_agent import UnitreeH2DAEVisual, UnitreeH2STL, UnitreeH2UpperGripper  # noqa: F401
from .h2_constants import H2_REST_QPOS_BY_JOINT, H2_UPPER_BODY_JOINT_DEFAULTS


# Place the fixed pelvis high enough for the feet to sit near the table-scene
# floor.  The yaw turns H2 toward the tabletop workspace for inspection.
H2_ROOT_POSE = sapien.Pose(
    p=[0.95, 0.0, 1.05],
    q=euler2quat(0.0, 0.0, np.pi),
)
H2_UPPER_ROOT_POSE = sapien.Pose(
    p=[0.78, 0.0, 0.0],
    q=euler2quat(0.0, 0.0, np.pi),
)


@register_env(
    "PickClutterYCBH2-v1",
    asset_download_ids=["ycb", "pick_clutter_ycb_configs"],
    max_episode_steps=100,
)
class PickClutterYCBH2Env(PickClutterYCBEnv):
    """PickClutter scene specialized enough to place and render Unitree H2."""

    SUPPORTED_ROBOTS = [
        "unitree_h2_stl",
        "unitree_h2_dae_visual",
        "unitree_h2_upper_gripper",
    ]

    def __init__(self, *args: Any, robot_uids: str = "unitree_h2_upper_gripper", **kwargs: Any):
        self._h2_robot_uid = robot_uids
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0.0, **kwargs)

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(
            eye=[1.85, 1.55, 1.35],
            target=[0.35, 0.0, 0.55],
        )
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=1280,
            height=720,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(
            self,
            options,
            root_pose_for_robot(self._h2_robot_uid),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        active_joint_names = [joint.name for joint in self.agent.robot.get_active_joints()]
        qpos = np.repeat(
            qpos_for_joints(active_joint_names, self._h2_robot_uid)[None, :],
            len(env_idx),
            axis=0,
        )
        self.agent.reset(qpos)
        self.agent.robot.set_pose(root_pose_for_robot(self._h2_robot_uid))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBH2Env.SUPPORTED_ROBOTS,
        default="unitree_h2_upper_gripper",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=None)
    parser.add_argument("--camera-target", type=float, nargs=3, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_unitree_h2/runs"),
    )
    return parser.parse_args()


def root_pose_for_robot(robot_uid: str) -> sapien.Pose:
    if robot_uid == "unitree_h2_upper_gripper":
        return H2_UPPER_ROOT_POSE
    return H2_ROOT_POSE


def qpos_for_joints(joint_names: list[str], robot_uid: str) -> np.ndarray:
    defaults = (
        H2_UPPER_BODY_JOINT_DEFAULTS
        if robot_uid == "unitree_h2_upper_gripper"
        else H2_REST_QPOS_BY_JOINT
    )
    return np.array([defaults.get(name, 0.0) for name in joint_names], dtype=np.float32)


def camera_defaults(robot_uid: str) -> tuple[list[float], list[float], list[float], list[float]]:
    if robot_uid == "unitree_h2_upper_gripper":
        return (
            [1.35, 1.05, 0.85],
            [0.34, 0.0, 0.34],
            [1.00, 0.78, 0.62],
            [0.48, 0.0, 0.35],
        )
    return (
        [1.85, 1.55, 1.35],
        [0.35, 0.0, 0.55],
        [1.35, 1.15, 1.05],
        [0.55, 0.0, 0.65],
    )


def normalize_rgb(frame: Any) -> np.ndarray:
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255.0, 0, 255)
    return np.ascontiguousarray(frame[..., :3].astype(np.uint8))


def set_render_camera(env, eye: list[float], target: list[float]) -> None:
    pose = sapien_utils.look_at(eye=eye, target=target).raw_pose.detach().cpu().numpy().reshape(-1)
    camera = env.unwrapped._human_render_cameras["render_camera"].camera
    camera.set_local_pose(sapien.Pose(p=pose[:3], q=pose[3:]))


def main() -> int:
    args = parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    run_dir = args.output_dir / f"pickclutter_{args.robot_uid}_seed{args.seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    camera_eye, camera_target, closeup_eye, closeup_target = camera_defaults(args.robot_uid)
    if args.camera_eye is not None:
        camera_eye = list(args.camera_eye)
    if args.camera_target is not None:
        camera_target = list(args.camera_target)

    env = gym.make(
        "PickClutterYCBH2-v1",
        robot_uids=args.robot_uid,
        control_mode="pd_joint_pos",
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=100,
        human_render_camera_configs={
            "width": args.width,
            "height": args.height,
            "fov": 1.0,
            "near": 0.01,
            "far": 100,
        },
    )

    image_path = run_dir / "scene.png"
    closeup_path = run_dir / "robot_closeup.png"
    metadata_path = run_dir / "metadata.json"
    try:
        env.reset(seed=args.seed)
        raw_env = env.unwrapped
        robot = raw_env.agent.robot
        active_joint_names = [joint.name for joint in robot.get_active_joints()]

        set_render_camera(env, camera_eye, camera_target)
        Image.fromarray(normalize_rgb(env.render())).save(image_path)

        set_render_camera(env, closeup_eye, closeup_target)
        Image.fromarray(normalize_rgb(env.render())).save(closeup_path)

        qpos = robot.get_qpos().detach().cpu().numpy().reshape(-1)
        runtime_collision_counts = {}
        for link in robot.get_links():
            count = sum(
                len(component.get_collision_shapes())
                for component in link._objs
            )
            if count:
                runtime_collision_counts[link.name] = count

        metadata = {
            "seed": args.seed,
            "env_id": "PickClutterYCBH2-v1",
            "robot_uid": args.robot_uid,
            "root_fixed": True,
            "robot_root_pose": robot.pose.raw_pose.detach().cpu().numpy().reshape(-1).tolist(),
            "active_joint_names": active_joint_names,
            "qpos": qpos.tolist(),
            "camera_eye": camera_eye,
            "camera_target": camera_target,
            "link_names": [link.name for link in robot.get_links()],
            "runtime_collision_shape_count": sum(runtime_collision_counts.values()),
            "runtime_collision_shape_link_counts": runtime_collision_counts,
            "target_object_name": raw_env.target_object.name,
            "image": str(image_path.resolve()),
            "robot_closeup_image": str(closeup_path.resolve()),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()

    print(f"scene_image={image_path}")
    print(f"robot_closeup_image={closeup_path}")
    print(f"metadata={metadata_path}")
    print("h2_maniskill_render_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
