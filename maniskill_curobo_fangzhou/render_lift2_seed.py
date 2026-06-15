#!/usr/bin/env python3
"""Render the Lift2 URDF inside one deterministic ManiSkill clutter seed."""

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
from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .lift2_agent import LIFT2_JOINT_NAMES, LIFT2_REST_QPOS
from .urdf_adapter import load_collision_spheres


# Lift2 faces the table from the positive-X side. After the 180-degree yaw,
# its base extends 0.2180664 m toward negative X. The first placement at
# x=0.7069260 left only a narrow visual clearance and the full mesh collision
# base/wheels penetrated the table workspace. Move the root 8 cm backward
# (positive world X) so full-collision execution starts without table contact.
LIFT2_ROOT_POSE = sapien.Pose(
    p=[0.7869260, 0.0, -0.7516],
    q=euler2quat(0.0, 0.0, np.pi),
)


@register_env(
    "PickClutterYCBLift2-v1",
    asset_download_ids=["ycb", "pick_clutter_ycb_configs"],
    max_episode_steps=100,
)
class PickClutterYCBLift2Env(PickClutterYCBEnv):
    """PickClutter scene specialized only enough to place and render Lift2."""

    SUPPORTED_ROBOTS = [
        "lift2_visual",
        "lift2_full_collision",
        "lift2_collision_spheres",
        "lift2_collision_spheres_debug",
    ]

    def __init__(self, *args: Any, robot_uids: str = "lift2_visual", **kwargs: Any):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0.0, **kwargs)

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(
            eye=[1.25, 1.35, 0.95],
            target=[-0.35, 0.0, -0.10],
        )
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=960,
            height=720,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(
            self,
            options,
            LIFT2_ROOT_POSE,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        qpos = np.repeat(LIFT2_REST_QPOS[None, :], len(env_idx), axis=0)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(LIFT2_ROOT_POSE)


@register_env(
    "PickSingleYCBLift2-v1",
    asset_download_ids=["ycb"],
    max_episode_steps=100,
)
class PickSingleYCBLift2Env(PickSingleYCBEnv):
    """PickSingle scene specialized only enough to place and render Lift2."""

    SUPPORTED_ROBOTS = PickClutterYCBLift2Env.SUPPORTED_ROBOTS

    def __init__(self, *args: Any, robot_uids: str = "lift2_visual", **kwargs: Any):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0.0, **kwargs)

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(
            eye=[1.25, 1.35, 0.95],
            target=[-0.35, 0.0, -0.10],
        )
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=960,
            height=720,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(
            self,
            options,
            LIFT2_ROOT_POSE,
        )

    def evaluate(self):
        """Lift2 does not implement ManiSkill's Panda-style is_grasping hook."""
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "is_grasped": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def _get_obs_extra(self, info: dict):
        tcp_pose = self._right_tcp_raw_pose()
        tcp_pos = tcp_pose[:, :3]
        obs = {
            "tcp_pose": tcp_pose,
            "goal_pos": self.goal_site.pose.p,
            "is_grasped": info["is_grasped"],
        }
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - tcp_pos,
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - tcp_pos,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs

    def _right_tcp_raw_pose(self) -> torch.Tensor:
        for link in self.agent.robot.get_links():
            if str(link.name) != "right_tcp":
                continue
            raw_pose = getattr(link.pose, "raw_pose", None)
            if raw_pose is not None:
                return raw_pose
        fallback = torch.zeros((self.num_envs, 7), device=self.device)
        fallback[:, 3] = 1.0
        return fallback

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

        qpos = np.repeat(LIFT2_REST_QPOS[None, :], len(env_idx), axis=0)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(LIFT2_ROOT_POSE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_collision_spheres_debug",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs"),
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
    return frame[..., :3].astype(np.uint8)


def main() -> int:
    args = parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    run_dir = args.output_dir / (
        f"pickclutter_{args.robot_uid}_seed{args.seed:03d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PickClutterYCBLift2-v1",
        robot_uids=args.robot_uid,
        control_mode="pd_joint_pos",
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=100,
    )
    try:
        env.reset(seed=args.seed)
        raw_env = env.unwrapped
        robot = raw_env.agent.robot
        active_joint_names = [joint.name for joint in robot.get_active_joints()]
        if tuple(active_joint_names) != LIFT2_JOINT_NAMES:
            raise RuntimeError(
                "Unexpected active-joint order: "
                f"expected={LIFT2_JOINT_NAMES}, actual={active_joint_names}"
            )

        image = normalize_rgb(env.render())
        image_path = run_dir / "scene.png"
        Image.fromarray(image).save(image_path)

        render_camera = raw_env._human_render_cameras["render_camera"].camera
        closeup_pose = sapien_utils.look_at(
            eye=[1.15, 1.50, 0.75],
            target=[0.15, 0.0, 0.02],
        ).raw_pose.detach().cpu().numpy().reshape(-1)
        render_camera.set_local_pose(
            sapien.Pose(p=closeup_pose[:3], q=closeup_pose[3:])
        )
        closeup = normalize_rgb(env.render())
        closeup_path = run_dir / "robot_closeup.png"
        Image.fromarray(closeup).save(closeup_path)

        qpos = robot.get_qpos().detach().cpu().numpy().reshape(-1)
        sphere_specs = (
            {}
            if args.robot_uid == "lift2_visual"
            else load_collision_spheres()
        )
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
            "env_id": "PickClutterYCBLift2-v1",
            "robot_uid": args.robot_uid,
            "visual_only": args.robot_uid == "lift2_visual",
            "collision_sphere_count": sum(
                len(specs) for specs in sphere_specs.values()
            ),
            "collision_sphere_link_counts": {
                link_name: len(specs)
                for link_name, specs in sphere_specs.items()
            },
            "runtime_collision_shape_count": sum(
                runtime_collision_counts.values()
            ),
            "runtime_collision_shape_link_counts": runtime_collision_counts,
            "root_fixed": True,
            "robot_root_pose": robot.pose.raw_pose.detach().cpu().numpy().reshape(-1).tolist(),
            "active_joint_names": active_joint_names,
            "qpos": qpos.tolist(),
            "link_names": [link.name for link in robot.get_links()],
            "target_object_name": raw_env.target_object.name,
            "image": str(image_path.resolve()),
            "robot_closeup_image": str(closeup_path.resolve()),
        }
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()

    print(f"scene_image={image_path}")
    print(f"robot_closeup_image={closeup_path}")
    print(f"metadata={metadata_path}")
    print("lift2_maniskill_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
