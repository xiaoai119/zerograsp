#!/usr/bin/env python3
"""Plan a tiny Lift2 right-arm reach with a prebuilt M4C voxel ESDF scene."""

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

from maniskill_curobo_real.run_world_collision_stages import load_planner_scene_model

from .generate_lift2_curobo_config import DEFAULT_OUTPUT
from .lift2_constants import LIFT2_CUROBO_SAFE_REST_QPOS
from .lift2_curobo_bridge import (
    make_lift2_action_from_right_arm_qpos,
    right_arm_qpos_from_maniskill,
)
from .render_lift2_seed import PickClutterYCBLift2Env  # noqa: F401


DEFAULT_M4C_ROOT = Path(
    "maniskill_curobo_fangzhou/runs/m4c_lift2_collision_spheres_seed1_200"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--m4c-root", type=Path, default=DEFAULT_M4C_ROOT)
    parser.add_argument("--scene-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_full_collision",
    )
    parser.add_argument("--offset-base", type=float, nargs=3, default=[0.0, 0.0, -0.02])
    parser.add_argument("--target-base", type=float, nargs=3, default=None)
    parser.add_argument(
        "--target-world",
        type=float,
        nargs=3,
        default=None,
        help="Absolute right_tcp target position in the ManiSkill world frame.",
    )
    parser.add_argument(
        "--target-quaternion-wxyz",
        type=float,
        nargs=4,
        default=None,
        help="Optional target right_tcp orientation in the Lift2/cuRobo base frame.",
    )
    parser.add_argument("--show-target-marker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-marker-radius", type=float, default=0.02)
    parser.add_argument("--gripper-open", type=float, default=0.03)
    parser.add_argument("--gripper-closed", type=float, default=0.0)
    parser.add_argument("--target-hold-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=[-0.10, 1.00, 0.72])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.30, 0.0, 0.03])
    parser.add_argument("--num-ik-seeds", type=int, default=8)
    parser.add_argument("--num-trajopt-seeds", type=int, default=2)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/lift2_m4c_right_arm_reach_seed001"),
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
    video_path = out_dir / f"m4c_right_arm_reach_seed{args.seed:03d}_{args.width}x{args.height}.mp4"
    poster_path = out_dir / "poster_frame.png"
    trajectory_path = out_dir / "planned_trajectory.npz"

    scene_path = resolve_scene_model_path(args)
    scene_model = load_planner_scene_model(scene_path)
    collision_cache = collision_cache_for_scene(scene_model)

    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    planner_cfg = MotionPlannerCfg.create(
        robot=config,
        scene_model=scene_model,
        collision_cache=collision_cache,
        use_cuda_graph=False,
        num_ik_seeds=int(args.num_ik_seeds),
        num_trajopt_seeds=int(args.num_trajopt_seeds),
    )
    planner = MotionPlanner(planner_cfg)
    planner.warmup(enable_graph=False, num_warmup_iterations=int(args.warmup_iterations))

    env = gym.make(
        "PickClutterYCBLift2-v1",
        robot_uids=args.robot_uid,
        control_mode="pd_joint_pos",
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=1000,
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
        safe_rest = np.asarray(LIFT2_CUROBO_SAFE_REST_QPOS, dtype=np.float32)
        raw_env.agent.robot.set_qpos(torch.as_tensor(safe_rest, device=raw_env.device).unsqueeze(0))
        for _ in range(5):
            env.step(safe_rest)
        raw_env.agent.robot.set_qpos(torch.as_tensor(safe_rest, device=raw_env.device).unsqueeze(0))

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
        world_from_base = robot_base_matrix(raw_env)
        if args.target_world is not None:
            target_position_np = transform_point(
                np.linalg.inv(world_from_base),
                np.asarray(args.target_world, dtype=np.float64),
            )
            target_mode = "target_world"
        elif args.target_base is not None:
            target_position_np = np.asarray(args.target_base, dtype=np.float64)
            target_mode = "target_base"
        else:
            target_position_np = (
                start_position.detach().cpu().numpy().reshape(3)
                + np.asarray(args.offset_base, dtype=np.float64)
            )
            target_mode = "offset_base"
        target_position = torch.as_tensor(
            target_position_np,
            device="cuda",
            dtype=torch.float32,
        ).reshape(1, 3)
        if args.target_quaternion_wxyz is None:
            target_quaternion = start_quaternion
        else:
            quaternion = np.asarray(args.target_quaternion_wxyz, dtype=np.float64)
            quaternion /= np.linalg.norm(quaternion)
            target_quaternion = torch.as_tensor(
                quaternion,
                device="cuda",
                dtype=torch.float32,
            ).reshape(1, 4)
        target_world = transform_point(world_from_base, target_position_np)
        if args.show_target_marker:
            add_target_marker(
                raw_env.scene,
                target_world,
                radius=float(args.target_marker_radius),
            )

        goal_pose = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=target_position.reshape(1, 1, 1, 1, 3),
            quaternion=target_quaternion.reshape(1, 1, 1, 1, 4),
        )
        result = planner.plan_pose(goal_pose, q_start)
        success = result is not None and result.success is not None and bool(result.success.any())
        if not success:
            report = base_report(args, scene_path, scene_model, collision_cache)
            report.update(
                {
                    "status": "planning_failed",
                    "planner_status": str(getattr(result, "status", "unknown")),
                    "start_right_arm_qpos": right_arm_qpos.tolist(),
                    "start_tcp_position": start_position.detach().cpu().numpy().reshape(-1).tolist(),
                    "target_tcp_position": target_position.detach().cpu().numpy().reshape(-1).tolist(),
                }
            )
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"Lift2 M4C right-arm reach planning failed: {report['planner_status']}")

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
            target_quaternion=target_quaternion.detach().cpu().numpy(),
        )

        frames_written = 0
        with imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_index, q in enumerate(trajectory):
                action = make_lift2_action_from_right_arm_qpos(
                    q,
                    gripper_qpos=args.gripper_open,
                )
                for _ in range(max(1, int(args.action_repeat))):
                    env.step(action)
                    frame = normalize_rgb(env.render())
                    if frame_index == max(len(trajectory) // 2, 0):
                        Image.fromarray(frame).save(poster_path)
                    writer.append_data(frame)
                    frames_written += 1
            final_q = trajectory[-1]
            hold_action = make_lift2_action_from_right_arm_qpos(
                final_q,
                gripper_qpos=args.gripper_open,
            )
            for _ in range(max(0, int(args.target_hold_steps))):
                env.step(hold_action)
                writer.append_data(normalize_rgb(env.render()))
                frames_written += 1
            preclose_tcp_world = link_pose_matrix(raw_env, "right_tcp")[:3, 3]
            close_action = make_lift2_action_from_right_arm_qpos(
                final_q,
                gripper_qpos=args.gripper_closed,
            )
            for _ in range(max(0, int(args.close_steps))):
                env.step(close_action)
                writer.append_data(normalize_rgb(env.render()))
                frames_written += 1

        final_tcp_world = link_pose_matrix(raw_env, "right_tcp")[:3, 3]
        final_robot_qpos = raw_env.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)
        final_right_arm_qpos = right_arm_qpos_from_maniskill(
            active_joint_names,
            final_robot_qpos,
        )
        final_q_state = JointState.from_position(
            torch.as_tensor(
                final_right_arm_qpos,
                device="cuda",
                dtype=torch.float32,
            ).unsqueeze(0),
            joint_names=planner.joint_names,
        )
        final_fk_pose = planner.compute_kinematics(final_q_state).tool_poses.get_link_pose(
            "right_tcp"
        )
        final_fk_position_base = (
            final_fk_pose.position.detach().cpu().numpy().reshape(3)
        )
        final_fk_position_world = transform_point(
            world_from_base,
            final_fk_position_base,
        )

        report = base_report(args, scene_path, scene_model, collision_cache)
        report.update(
            {
                "status": "ok",
                "planner_joint_names": planner.joint_names,
                "tool_frames": planner.tool_frames,
                "start_right_arm_qpos": right_arm_qpos.tolist(),
                "start_tcp_position": start_position.detach().cpu().numpy().reshape(-1).tolist(),
                "target_tcp_position": target_position.detach().cpu().numpy().reshape(-1).tolist(),
                "target_tcp_position_world": target_world.tolist(),
                "target_tcp_quaternion_wxyz": target_quaternion.detach().cpu().numpy().reshape(-1).tolist(),
                "preclose_tcp_position_world": preclose_tcp_world.tolist(),
                "preclose_tcp_position_error_world_m": float(
                    np.linalg.norm(preclose_tcp_world - target_world)
                ),
                "final_tcp_position_world": final_tcp_world.tolist(),
                "final_tcp_position_error_world_m": float(np.linalg.norm(final_tcp_world - target_world)),
                "commanded_final_right_arm_qpos": final_q.tolist(),
                "actual_final_right_arm_qpos": final_right_arm_qpos.tolist(),
                "final_joint_tracking_max_abs_rad": float(
                    np.max(np.abs(final_right_arm_qpos - final_q))
                ),
                "final_curobo_fk_position_world": final_fk_position_world.tolist(),
                "final_curobo_fk_error_world_m": float(
                    np.linalg.norm(final_fk_position_world - target_world)
                ),
                "maniskill_vs_curobo_tcp_error_world_m": float(
                    np.linalg.norm(final_tcp_world - final_fk_position_world)
                ),
                "target_mode": target_mode,
                "robot_uid": args.robot_uid,
                "action_repeat": int(args.action_repeat),
                "target_hold_steps": int(args.target_hold_steps),
                "close_steps": int(args.close_steps),
                "frames_written": int(frames_written),
                "trajectory_steps": int(trajectory.shape[0]),
                "trajectory": str(trajectory_path),
                "video": str(video_path),
                "poster_frame": str(poster_path),
            }
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()
        destroy = getattr(planner, "destroy", None)
        if callable(destroy):
            destroy()

    print(json.dumps({"report": str(report_path), "video": str(video_path)}, ensure_ascii=False, indent=2))
    return 0


def resolve_scene_model_path(args: argparse.Namespace) -> Path:
    if args.scene_model is not None:
        return args.scene_model.expanduser().resolve()
    return (
        args.m4c_root.expanduser().resolve()
        / f"seed{int(args.seed):03d}"
        / "real_scene"
        / "curobo_scene_voxel.npz"
    )


def collision_cache_for_scene(scene_model: Any) -> dict[str, Any]:
    cache: dict[str, Any] = {"cuboid": 64}
    if getattr(scene_model, "mesh", None):
        cache["mesh"] = 64
    if getattr(scene_model, "voxel", None):
        voxel = scene_model.voxel[0]
        cache["voxel"] = {
            "layers": 1,
            "dims": list(voxel.dims),
            "voxel_size": float(voxel.voxel_size),
        }
    return cache


def base_report(
    args: argparse.Namespace,
    scene_path: Path,
    scene_model: Any,
    collision_cache: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "config": str(args.config.expanduser().resolve()),
        "scene_model": str(scene_path),
        "scene_has_cuboid": bool(getattr(scene_model, "cuboid", None)),
        "scene_has_voxel": bool(getattr(scene_model, "voxel", None)),
        "collision_cache": collision_cache,
        "offset_base": [float(v) for v in args.offset_base],
        "target_base": None if args.target_base is None else [float(v) for v in args.target_base],
        "target_world": None if args.target_world is None else [float(v) for v in args.target_world],
    }


def set_render_camera(env: Any, eye: list[float], target: list[float]) -> None:
    pose = sapien_utils.look_at(eye=eye, target=target).raw_pose.detach().cpu().numpy().reshape(-1)
    camera = env.unwrapped._human_render_cameras["render_camera"].camera
    camera.set_local_pose(sapien.Pose(p=pose[:3], q=pose[3:]))


def add_target_marker(scene: Any, target_world: np.ndarray, *, radius: float) -> None:
    material = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 1.0, 0.96])
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=radius, material=material)
    builder.set_initial_pose(sapien.Pose(p=target_world.tolist()))
    builder.build_kinematic(name="lift2_right_tcp_target_marker")


def robot_base_matrix(raw_env: Any) -> np.ndarray:
    return link_pose_matrix(raw_env, "base_link")


def link_pose_matrix(raw_env: Any, link_name: str) -> np.ndarray:
    for link in raw_env.agent.robot.get_links():
        if str(link.name) != link_name:
            continue
        raw = getattr(link.pose, "raw_pose", None)
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        return sapien.Pose(p=values[:3], q=values[3:7]).to_transformation_matrix()
    raise KeyError(f"Lift2 link not found: {link_name}")


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (
        np.asarray(transform, dtype=np.float64).reshape(4, 4)
        @ np.array([*np.asarray(point, dtype=np.float64).reshape(3), 1.0])
    )[:3]


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
