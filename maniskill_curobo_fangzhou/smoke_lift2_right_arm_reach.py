#!/usr/bin/env python3
"""Plan and execute a tiny right-arm Lift2 reaching motion with cuRobo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import sapien
import torch
import yaml
from PIL import Image

from mani_skill.utils import sapien_utils

from .generate_lift2_curobo_config import DEFAULT_OUTPUT
from .lift2_constants import LIFT2_CUROBO_SAFE_REST_QPOS
from .lift2_curobo_bridge import (
    make_lift2_action_from_right_arm_qpos,
    right_arm_qpos_from_maniskill,
)
from .render_lift2_seed import PickClutterYCBLift2Env  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--offset-base", type=float, nargs=3, default=[0.0, 0.0, -0.035])
    parser.add_argument(
        "--target-base",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Absolute right_tcp target position in the cuRobo/Lift2 base frame. "
            "When omitted, --offset-base is applied to the current TCP position."
        ),
    )
    parser.add_argument("--show-target-marker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-marker-radius", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=[-0.10, 1.00, 0.72])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.30, 0.0, 0.03])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/lift2_right_arm_reach_smoke_seed001"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    video_path = out_dir / f"right_arm_reach_seed{args.seed:03d}_{args.width}x{args.height}.mp4"
    poster_path = out_dir / "poster_frame.png"
    trajectory_path = out_dir / "planned_trajectory.npz"

    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    planner_cfg = MotionPlannerCfg.create(
        robot=config,
        scene_model=None,
        collision_cache={"cuboid": 8},
        use_cuda_graph=False,
        num_ik_seeds=8,
        num_trajopt_seeds=2,
    )
    planner = MotionPlanner(planner_cfg)
    planner.warmup(enable_graph=False, num_warmup_iterations=1)

    env = gym.make(
        "PickClutterYCBLift2-v1",
        robot_uids="lift2_collision_spheres",
        control_mode="pd_joint_pos",
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=240,
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
        raw_env.agent.robot.set_qpos(
            torch.as_tensor(LIFT2_CUROBO_SAFE_REST_QPOS, device=raw_env.device).unsqueeze(0)
        )
        for _ in range(5):
            env.step(LIFT2_CUROBO_SAFE_REST_QPOS)

        active_joint_names = [joint.name for joint in raw_env.agent.robot.get_active_joints()]
        qpos = raw_env.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)
        right_arm_qpos = right_arm_qpos_from_maniskill(active_joint_names, qpos)
        q_start = JointState.from_position(
            torch.as_tensor(right_arm_qpos, device="cuda", dtype=torch.float32).unsqueeze(0),
            joint_names=planner.joint_names,
        )
        start_state = planner.compute_kinematics(q_start)
        start_pose = start_state.tool_poses.get_link_pose("right_tcp")
        start_position = start_pose.position.detach().clone()
        start_quaternion = start_pose.quaternion.detach().clone()
        if args.target_base is None:
            target_position = start_position + torch.as_tensor(
                args.offset_base, device="cuda", dtype=torch.float32
            ).reshape(1, 3)
            target_mode = "offset_base"
        else:
            target_position = torch.as_tensor(
                args.target_base, device="cuda", dtype=torch.float32
            ).reshape(1, 3)
            target_mode = "target_base"
        if args.show_target_marker:
            target_world = transform_base_point(raw_env, target_position.detach().cpu().numpy().reshape(3))
            add_target_marker(raw_env.scene, target_world, radius=float(args.target_marker_radius))
        else:
            target_world = None

        goal_pose = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=target_position.reshape(1, 1, 1, 1, 3),
            quaternion=start_quaternion.reshape(1, 1, 1, 1, 4),
        )
        result = planner.plan_pose(goal_pose, q_start)
        success = result is not None and result.success is not None and bool(result.success.any())
        if not success:
            status = str(getattr(result, "status", "unknown"))
            raise RuntimeError(f"Lift2 right-arm cuRobo reach planning failed: {status}")

        plan = result.get_interpolated_plan()
        plan_position = plan.position.detach().cpu().numpy()
        if plan_position.ndim == 4:
            plan_position = plan_position[0, 0]
        elif plan_position.ndim == 3:
            plan_position = plan_position[0]
        plan_joint_names = list(plan.joint_names)
        right_arm_indices = [plan_joint_names.index(name) for name in planner.joint_names]
        trajectory = plan_position[:, right_arm_indices]
        np.savez(
            trajectory_path,
            trajectory=trajectory,
            joint_names=np.asarray(planner.joint_names),
            full_plan_trajectory=plan_position,
            full_plan_joint_names=np.asarray(plan_joint_names),
            start_qpos=right_arm_qpos,
            start_position=start_position.detach().cpu().numpy(),
            target_position=target_position.detach().cpu().numpy(),
            target_quaternion=start_quaternion.detach().cpu().numpy(),
        )
        with imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_index, q in enumerate(trajectory):
                action = make_lift2_action_from_right_arm_qpos(q, gripper_qpos=0.03)
                env.step(action)
                frame = normalize_rgb(env.render())
                if frame_index == max(len(trajectory) // 2, 0):
                    Image.fromarray(frame).save(poster_path)
                writer.append_data(frame)

        report = {
            "status": "ok",
            "config": str(args.config.expanduser().resolve()),
            "seed": args.seed,
            "planner_joint_names": planner.joint_names,
            "tool_frames": planner.tool_frames,
            "start_right_arm_qpos": right_arm_qpos.tolist(),
            "start_tcp_position": start_position.detach().cpu().numpy().reshape(-1).tolist(),
            "target_tcp_position": target_position.detach().cpu().numpy().reshape(-1).tolist(),
            "target_tcp_position_world": target_world.tolist() if target_world is not None else None,
            "target_tcp_quaternion_wxyz": start_quaternion.detach().cpu().numpy().reshape(-1).tolist(),
            "target_mode": target_mode,
            "target_base": None if args.target_base is None else [float(v) for v in args.target_base],
            "offset_base": [float(v) for v in args.offset_base],
            "trajectory_steps": int(trajectory.shape[0]),
            "trajectory": str(trajectory_path),
            "video": str(video_path),
            "poster_frame": str(poster_path),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()

    print(json.dumps({"report": str(report_path), "video": str(video_path)}, ensure_ascii=False, indent=2))
    return 0


def set_render_camera(env: Any, eye: list[float], target: list[float]) -> None:
    pose = sapien_utils.look_at(eye=eye, target=target).raw_pose.detach().cpu().numpy().reshape(-1)
    camera = env.unwrapped._human_render_cameras["render_camera"].camera
    camera.set_local_pose(sapien.Pose(p=pose[:3], q=pose[3:]))


def add_target_marker(scene: Any, target_world: np.ndarray, *, radius: float) -> None:
    material = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 1.0, 0.96])
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        radius=radius,
        material=material,
    )
    builder.set_initial_pose(sapien.Pose(p=target_world.tolist()))
    builder.build_kinematic(name="lift2_right_tcp_target_marker")


def transform_base_point(raw_env: Any, point_base: np.ndarray) -> np.ndarray:
    link_map = {str(link.name): link for link in raw_env.agent.robot.get_links()}
    base_link = link_map["base_link"]
    world_from_base = pose_to_matrix(getattr(base_link, "pose", None), "base_link pose")
    return (world_from_base @ np.array([*point_base, 1.0], dtype=np.float64))[:3]


def pose_to_matrix(pose: Any, name: str) -> np.ndarray:
    raw = getattr(pose, "raw_pose", None)
    if raw is None:
        raise ValueError(f"{name} has no raw_pose")
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    return sapien.Pose(p=raw[:3], q=raw[3:7]).to_transformation_matrix()


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
