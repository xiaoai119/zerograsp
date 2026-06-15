#!/usr/bin/env python3
"""Execute ZeroGrasp candidates with Lift2's right arm and an M4C voxel world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
import sapien
import torch
import yaml

from mani_skill.utils import sapien_utils

from maniskill_curobo.joint_trajectory_utils import ordered_values
from maniskill_curobo.scripts import execute_curobo_pick as execute
from maniskill_curobo_real.run_world_collision_stages import load_planner_scene_model

from .export_lift2_zerograsp_input import (
    DEFAULT_CAMERA_EYE,
    DEFAULT_CAMERA_FRAME,
    DEFAULT_CAMERA_TARGET,
    build_env,
)
from .generate_lift2_curobo_config import DEFAULT_OUTPUT
from .lift2_constants import LIFT2_CUROBO_SAFE_REST_QPOS, LIFT2_JOINT_NAMES
from .lift2_curobo_bridge import (
    make_lift2_action_from_right_arm_qpos,
    right_arm_qpos_from_maniskill,
)
from .render_lift2_seed import PickClutterYCBLift2Env, PickSingleYCBLift2Env  # noqa: F401
from .smoke_lift2_m4c_right_arm_reach import collision_cache_for_scene


DEFAULT_M4C_ROOT = Path(
    "maniskill_curobo_fangzhou/runs/m4c_lift2_collision_spheres_seed1_200"
)
DEFAULT_ZEROGRASP_OUTPUT = Path(
    "maniskill_curobo_fangzhou/runs/lift2_m4c_grasp_seed001/zerograsp_output"
)
DEFAULT_OUTPUT_ROOT = Path("maniskill_curobo_fangzhou/runs/lift2_m4c_grasp_seed001")
DEFAULT_LIFT2_WORKSPACE_BOUNDS = ((0.25, 1.15), (-0.75, 0.75), (0.68, 1.35))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--env-id",
        choices=("PickClutterYCBLift2-v1", "PickSingleYCBLift2-v1"),
        default="PickClutterYCBLift2-v1",
    )
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_full_collision",
        help=(
            "ManiSkill Lift2 robot variant for physics execution. cuRobo still "
            "uses the separate sphere-based robot config."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--m4c-root", type=Path, default=DEFAULT_M4C_ROOT)
    parser.add_argument("--scene-model", type=Path, default=None)
    parser.add_argument("--zerograsp-output", type=Path, default=DEFAULT_ZEROGRASP_OUTPUT)
    parser.add_argument("--candidate-top-k", type=int, default=7)
    parser.add_argument(
        "--approach-axis",
        choices=execute.APPROACH_AXIS_CHOICES,
        default="positive-x",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("grasp", "lift2-gripper", "start"),
        default="lift2-gripper",
        help=(
            "Use Panda-style tool-z grasp orientation, Lift2's tool-x gripper "
            "orientation, or keep the current right_tcp orientation."
        ),
    )
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument(
        "--grasp-position-offset-base",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help=(
            "Debug-only Cartesian offset added to the final grasp target in "
            "Lift2/cuRobo base frame after ZeroGrasp depth compensation."
        ),
    )
    parser.add_argument("--grasp-depth-scale", type=float, default=1.0)
    parser.add_argument("--grasp-depth-max-offset", type=float, default=0.04)
    parser.add_argument(
        "--grasp-depth-fallback-fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
    )
    parser.add_argument("--gripper-open", type=float, default=0.03)
    parser.add_argument("--gripper-closed", type=float, default=0.0)
    parser.add_argument(
        "--initial-lift-joint-value",
        type=float,
        default=None,
        help=(
            "Optional initial joint4 value in meters. Use this with fixed-lift "
            "cuRobo configs so ManiSkill execution starts from the same locked "
            "lift height as the planner."
        ),
    )
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-waypoints-per-stage", type=int, default=120)
    parser.add_argument(
        "--camera-frame",
        choices=(DEFAULT_CAMERA_FRAME, "robot", "world"),
        default=DEFAULT_CAMERA_FRAME,
    )
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument(
        "--render-camera-eye",
        type=float,
        nargs=3,
        default=[-0.55, 0.48, 0.52],
        help="World-frame third-person camera eye for the execution video.",
    )
    parser.add_argument(
        "--render-camera-target",
        type=float,
        nargs=3,
        default=[0.18, 0.04, 0.08],
        help="World-frame third-person camera target for the execution video.",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--num-ik-seeds", type=int, default=16)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-grasp-marker", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "execution")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    import mani_skill.envs  # noqa: F401
    from curobo.types import JointState

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    diagnostics_path = output_dir / "planning_diagnostics.json"
    video_path = None if args.no_video else output_dir / "execution.mp4"

    scene_path = resolve_scene_model_path(args)
    scene_model = load_planner_scene_model(scene_path)
    collision_cache = collision_cache_for_scene(scene_model)
    planner = build_lift2_planner(args, scene_model)
    env = build_lift2_execution_env(args, output_dir)
    recorder = VideoRecorder(video_path, fps=args.video_fps)
    manifest: dict[str, Any] = {
        "args": sanitize(vars(args)),
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "zerograsp_output": str(args.zerograsp_output.expanduser().resolve()),
        "scene_model": str(scene_path),
        "collision_cache": sanitize(collision_cache),
        "planner_joint_names": planner.joint_names,
        "tool_frames": planner.tool_frames,
        "stages": [],
        "candidate_attempts": [],
    }
    object_trace: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    marker_actors: list[Any] = []
    try:
        env.reset(seed=args.seed)
        raw_env = env.unwrapped
        set_render_camera(
            env,
            eye=list(args.render_camera_eye),
            target=list(args.render_camera_target),
        )
        safe_rest = np.asarray(LIFT2_CUROBO_SAFE_REST_QPOS, dtype=np.float32)
        if args.initial_lift_joint_value is not None:
            safe_rest[LIFT2_JOINT_NAMES.index("joint4")] = float(args.initial_lift_joint_value)
        set_lift2_qpos(raw_env, safe_rest)
        for _ in range(max(0, int(args.settle_steps))):
            _, _, _, _, final_info = env.step(safe_rest)
        # Full mesh collisions can nudge the arm while the scene settles.  The
        # planned start state must match the physical execution start state.
        set_lift2_qpos(raw_env, safe_rest)
        recorder.capture(env)
        object_trace.append(lift2_trace_sample(env, "after_reset_settle"))

        active_joint_names = [joint.name for joint in raw_env.agent.robot.get_active_joints()]
        qpos = raw_env.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)
        start_right_arm_qpos = right_arm_qpos_from_maniskill(
            active_joint_names,
            qpos,
            curobo_joint_names=planner.joint_names,
        )
        q_start = start_right_arm_qpos.astype(np.float32)
        start_tcp_pose = current_right_tcp_pose(planner, q_start)
        manifest["start_right_arm_qpos"] = q_start.tolist()
        manifest["start_tcp_position_base"] = start_tcp_pose["position"].tolist()
        manifest["start_tcp_quaternion_wxyz"] = start_tcp_pose["quaternion"].tolist()

        camera_model = execute.camera_model_matrix(env, args.camera)
        world_from_base = execute.robot_base_matrix(env)
        candidates = execute.load_grasp_candidates(args.zerograsp_output, top_k=args.candidate_top_k)
        if not candidates:
            raise RuntimeError(f"No ZeroGrasp candidates found in {args.zerograsp_output}.")
        selected = select_candidate(
            args=args,
            candidates=candidates,
            camera_model=camera_model,
            world_from_base=world_from_base,
            planner=planner,
            q_start=q_start,
            start_quaternion=start_tcp_pose["quaternion"],
            output_dir=output_dir,
            diagnostics_path=diagnostics_path,
            manifest=manifest,
        )
        control_pose = selected["control_pose"]
        targets = selected["targets"]
        manifest["selected_candidate"] = sanitize(selected["manifest"])
        manifest["stage_targets"] = {
            name: {
                "position_base": value["position"].tolist(),
                "quaternion_wxyz": value["quaternion"].tolist(),
            }
            for name, value in targets.items()
        }
        if not args.no_grasp_marker:
            zerograsp_marker = execute.build_control_pose_marker_geometry(
                control_pose=selected["zero_pose"],
                world_from_base_matrix=world_from_base,
                width_m=float(selected["manifest"].get("width_m", 0.06) or 0.06),
            )
            marker_actors = execute.add_grasp_marker_to_scene(
                raw_env.scene,
                zerograsp_marker,
                name_prefix="zerograsp_raw_marker",
            )
            manifest["grasp_marker"] = {
                **execute.marker_manifest(zerograsp_marker, len(marker_actors)),
                "meaning": "raw_zerograsp_center_before_full_depth",
            }
            execution_target_marker = execute.build_control_pose_marker_geometry(
                control_pose=control_pose,
                world_from_base_matrix=world_from_base,
                width_m=float(selected["manifest"].get("width_m", 0.06) or 0.06),
            )
            manifest["execution_target_marker"] = {
                **execute.marker_manifest(execution_target_marker, 0),
                "enabled": False,
                "meaning": "full_depth_adjusted_execution_target",
                "reason": "diagnostic_only_to_keep_scene_marker_aligned_with_projection",
            }

        stages = [
            ("pre", args.gripper_open),
            ("grasp", args.gripper_open),
            ("lift", args.gripper_closed),
        ]
        cached_plans = selected.get("cached_plans", {})
        for stage_name, gripper_qpos in stages:
            target = targets[stage_name]
            if stage_name in cached_plans:
                planned = rename_cached_plan(cached_plans[stage_name], stage_name, output_dir)
                print(
                    f"stage={stage_name:<5} target={np.round(target['position'], 4).tolist()} "
                    f"waypoints={planned.trajectory_positions.shape[0]} plan=<cached_from_selection>"
                )
            else:
                planned = execute.plan_stage(
                    planner=planner,
                    q_start=q_start,
                    target_position=target["position"],
                    target_quaternion=target["quaternion"],
                    output_dir=output_dir,
                    stage_name=stage_name,
                    diagnostics_path=diagnostics_path,
                )
            manifest["stages"].append(execute.stage_manifest(planned))
            q_start = ordered_values(
                planned.trajectory_joint_names,
                planned.trajectory_positions[-1],
                planner.joint_names,
            ).astype(np.float32)
            final_info = execute_lift2_trajectory(
                env=env,
                planned=planned,
                planner_joint_names=planner.joint_names,
                recorder=recorder,
                gripper_qpos=gripper_qpos,
                action_repeat=args.action_repeat,
                max_waypoints=args.max_waypoints_per_stage,
            )
            object_trace.append(lift2_trace_sample(env, f"after_{stage_name}"))
            if stage_name == "grasp":
                final_action = make_lift2_action_from_right_arm_qpos(
                    q_start,
                    curobo_joint_names=planner.joint_names,
                    gripper_qpos=args.gripper_closed,
                )
                for _ in range(max(0, int(args.close_steps))):
                    _, _, _, _, final_info = env.step(final_action)
                    recorder.capture(env)
                object_trace.append(lift2_trace_sample(env, "after_gripper_close"))
        video_saved = recorder.save(allow_empty=True)
        manifest["status"] = "ok"
        manifest["video_saved"] = str(video_saved) if video_saved else None
        manifest["object_lift_trace"] = object_trace
        manifest["object_lift_metrics"] = compute_lift_metrics(object_trace)
        manifest["final_info"] = sanitize(final_info)
        write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            recorder.capture(env)
            video_saved = recorder.save(allow_empty=True)
            manifest["video_saved"] = str(video_saved) if video_saved else None
        except Exception as capture_exc:
            manifest["video_capture_error"] = f"{type(capture_exc).__name__}: {capture_exc}"
        try:
            object_trace.append(lift2_trace_sample(env, "exception"))
            manifest["object_lift_trace"] = object_trace
            manifest["object_lift_metrics"] = compute_lift_metrics(object_trace)
        except Exception as trace_exc:
            manifest["trace_error"] = f"{type(trace_exc).__name__}: {trace_exc}"
        write_json(manifest_path, manifest)
        raise
    finally:
        remove_marker_actors(marker_actors)
        env.close()
        destroy = getattr(planner, "destroy", None)
        if callable(destroy):
            destroy()

    print(json.dumps({"manifest": str(manifest_path), "video": str(video_path)}, indent=2))
    return 0


def select_candidate(
    *,
    args: argparse.Namespace,
    candidates: list[execute.GraspRecord],
    camera_model: np.ndarray,
    world_from_base: np.ndarray,
    planner: Any,
    q_start: np.ndarray,
    start_quaternion: np.ndarray,
    output_dir: Path,
    diagnostics_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    motion = execute.MotionConfig(
        pregrasp_offset_m=float(args.pregrasp_offset),
        lift_offset_m=float(args.lift_offset),
        workspace_z_min=DEFAULT_LIFT2_WORKSPACE_BOUNDS[2][0],
        gripper_open=float(args.gripper_open),
        gripper_closed=float(args.gripper_closed),
    )
    depth_scales = execute.grasp_depth_attempt_scales(
        float(args.grasp_depth_scale),
        args.grasp_depth_fallback_fractions,
    )
    for rank, grasp in enumerate(candidates):
        zero_pose = compute_lift2_control_pose(
            grasp=grasp,
            camera_model_matrix=camera_model,
            world_from_base_matrix=world_from_base,
            approach_axis=args.approach_axis,
            workspace_bounds=DEFAULT_LIFT2_WORKSPACE_BOUNDS,
            orientation_mode=args.orientation_mode,
            forced_quaternion=start_quaternion if args.orientation_mode == "start" else None,
        )
        pre_targets = build_stage_targets_lift2(
            zero_pose,
            motion,
            pregrasp_control_pose=zero_pose,
        )
        try:
            pre_plan = execute.plan_stage(
                planner=planner,
                q_start=q_start,
                target_position=pre_targets["pre"]["position"],
                target_quaternion=pre_targets["pre"]["quaternion"],
                output_dir=output_dir,
                stage_name="candidate_pre",
                diagnostics_path=diagnostics_path,
                diagnostic_metadata={
                    "candidate_rank": int(rank),
                    "candidate_score": float(grasp.score),
                    "candidate_depth_scale": None,
                },
                save_plan=False,
            )
        except RuntimeError as exc:
            manifest["candidate_attempts"].append(
                sanitize(
                    {
                        "rank": int(rank),
                        "score": float(grasp.score),
                        "source": grasp.source,
                        "object_id": grasp.object_id,
                        "depth_scale": None,
                        "zero_depth_position_base": zero_pose.position_base.tolist(),
                        "quaternion_wxyz": zero_pose.quaternion_wxyz.tolist(),
                        "approach_axis_base": zero_pose.approach_axis_base.tolist(),
                        "success": False,
                        "failure_reason": str(exc),
                    }
                )
            )
            continue
        q_after_pre = ordered_values(
            pre_plan.trajectory_joint_names,
            pre_plan.trajectory_positions[-1],
            planner.joint_names,
        ).astype(np.float32)
        for depth_scale in depth_scales:
            control_pose, depth_manifest = apply_depth_offset_lift2(
                zero_pose,
                depth_m=grasp.depth_m,
                scale=depth_scale,
                max_offset_m=args.grasp_depth_max_offset,
                workspace_bounds=DEFAULT_LIFT2_WORKSPACE_BOUNDS,
            )
            control_pose, position_offset_manifest = apply_position_offset_lift2(
                control_pose,
                offset_base=args.grasp_position_offset_base,
                workspace_bounds=DEFAULT_LIFT2_WORKSPACE_BOUNDS,
            )
            targets = build_stage_targets_lift2(control_pose, motion, pregrasp_control_pose=zero_pose)
            attempt = {
                "rank": int(rank),
                "score": float(grasp.score),
                "source": grasp.source,
                "object_id": grasp.object_id,
                "depth_scale": float(depth_scale),
                "depth_offset": depth_manifest,
                "position_offset": position_offset_manifest,
                "zero_depth_position_base": zero_pose.position_base.tolist(),
                "grasp_position_base": control_pose.position_base.tolist(),
                "quaternion_wxyz": control_pose.quaternion_wxyz.tolist(),
                "approach_axis_base": control_pose.approach_axis_base.tolist(),
                "success": False,
            }
            try:
                grasp_plan = execute.plan_stage(
                    planner=planner,
                    q_start=q_after_pre,
                    target_position=targets["grasp"]["position"],
                    target_quaternion=targets["grasp"]["quaternion"],
                    output_dir=output_dir,
                    stage_name="candidate_grasp",
                    diagnostics_path=diagnostics_path,
                    diagnostic_metadata={
                        "candidate_rank": int(rank),
                        "candidate_score": float(grasp.score),
                        "candidate_depth_scale": float(depth_scale),
                    },
                    save_plan=False,
                )
                q_after_grasp = ordered_values(
                    grasp_plan.trajectory_joint_names,
                    grasp_plan.trajectory_positions[-1],
                    planner.joint_names,
                ).astype(np.float32)
                lift_plan = execute.plan_stage(
                    planner=planner,
                    q_start=q_after_grasp,
                    target_position=targets["lift"]["position"],
                    target_quaternion=targets["lift"]["quaternion"],
                    output_dir=output_dir,
                    stage_name="candidate_lift",
                    diagnostics_path=diagnostics_path,
                    diagnostic_metadata={
                        "candidate_rank": int(rank),
                        "candidate_score": float(grasp.score),
                        "candidate_depth_scale": float(depth_scale),
                    },
                    save_plan=False,
                )
            except RuntimeError as exc:
                attempt["failure_reason"] = str(exc)
                manifest["candidate_attempts"].append(sanitize(attempt))
                continue
            attempt["success"] = True
            manifest["candidate_attempts"].append(sanitize(attempt))
            candidate_manifest = {
                **attempt,
                "width_m": float(grasp.width_m),
                "height_m": float(grasp.height_m),
                "depth_m": float(grasp.depth_m),
                "translation_m_camera": grasp.translation_m_camera.tolist(),
                "rotation_matrix_camera": grasp.rotation_matrix_camera.tolist(),
                "approach_axis_convention": args.approach_axis,
                "orientation_mode": args.orientation_mode,
            }
            print(
                "selected_grasp "
                f"rank={rank} score={grasp.score:.6f} "
                f"depth_scale={depth_scale:.3f} "
                f"pos={np.round(control_pose.position_base, 4).tolist()}"
            )
            return {
                "control_pose": control_pose,
                "zero_pose": zero_pose,
                "targets": targets,
                "manifest": candidate_manifest,
                "cached_plans": {
                    "pre": pre_plan,
                    "grasp": grasp_plan,
                    "lift": lift_plan,
                },
            }
    raise RuntimeError(
        f"No plannable Lift2 grasp candidate in top-{len(candidates)} "
        f"with depth scales {depth_scales}."
    )


def rename_cached_plan(
    plan: execute.PlannedStage,
    stage_name: str,
    output_dir: Path,
) -> execute.PlannedStage:
    output_path = output_dir / f"curobo_plan_{stage_name}_cached.npz"
    np.savez(
        output_path,
        stage=stage_name,
        target_position=np.asarray(plan.target_position, dtype=np.float32),
        target_quaternion=np.asarray(plan.target_quaternion, dtype=np.float32),
        trajectory_joint_names=np.asarray(plan.trajectory_joint_names),
        trajectory_positions=np.asarray(plan.trajectory_positions, dtype=np.float32),
    )
    return execute.PlannedStage(
        name=stage_name,
        target_position=plan.target_position,
        target_quaternion=plan.target_quaternion,
        trajectory_positions=plan.trajectory_positions,
        trajectory_joint_names=plan.trajectory_joint_names,
        output_path=output_path,
    )


def compute_lift2_control_pose(
    *,
    grasp: execute.GraspRecord,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    approach_axis: str,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    orientation_mode: str,
    forced_quaternion: np.ndarray | None = None,
) -> execute.ControlPose:
    position_base = clamp_lift2(
        execute.opencv_camera_to_base(
            grasp.translation_m_camera,
            camera_model_matrix,
            world_from_base_matrix,
        ),
        workspace_bounds,
    )
    rotation_world_grasp = execute.opencv_grasp_rotation_to_world_axes(
        grasp.rotation_matrix_camera,
        camera_model_matrix,
    )
    base_from_world = np.linalg.inv(execute.matrix4(world_from_base_matrix, "world_from_base_matrix"))
    rotation_base_grasp = execute.orthonormalize_rotation(
        base_from_world[:3, :3] @ rotation_world_grasp
    )
    approach = execute.zerograsp_approach_vector(rotation_base_grasp, approach_axis)
    width = execute.zerograsp_width_vector(rotation_base_grasp)
    if orientation_mode == "lift2-gripper":
        rotation_base_tool = lift2_tcp_rotation_from_approach_and_width(approach, width)
    else:
        rotation_base_tool = execute.tcp_rotation_from_approach_and_width(approach, width)
    quaternion = execute.quat_wxyz_from_matrix(rotation_base_tool)
    if forced_quaternion is not None:
        quaternion = execute.unit(np.asarray(forced_quaternion, dtype=np.float64).reshape(4))
        rotation_base_tool = execute.rotation_matrix_from_quat_wxyz(quaternion)
    return execute.ControlPose(
        position_base=position_base,
        rotation_base_tool=rotation_base_tool,
        quaternion_wxyz=quaternion,
        approach_axis_base=execute.unit(approach),
    )


def lift2_tcp_rotation_from_approach_and_width(
    approach_axis_base: np.ndarray,
    width_axis_base: np.ndarray,
) -> np.ndarray:
    """Map ZeroGrasp axes onto Lift2 right_tcp: local +X approaches, local +Y opens."""

    tool_x = execute.unit(approach_axis_base)
    width = execute.unit(width_axis_base)
    tool_y = width - tool_x * float(np.dot(width, tool_x))
    if np.linalg.norm(tool_y) < 1e-6:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper, tool_x))) > 0.95:
            helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tool_y = helper - tool_x * float(np.dot(helper, tool_x))
    tool_y = execute.unit(tool_y)
    tool_z = execute.unit(np.cross(tool_x, tool_y))
    tool_y = execute.unit(np.cross(tool_z, tool_x))
    return np.stack([tool_x, tool_y, tool_z], axis=1)


def apply_depth_offset_lift2(
    control_pose: execute.ControlPose,
    *,
    depth_m: float,
    scale: float,
    max_offset_m: float,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[execute.ControlPose, dict[str, Any]]:
    depth = max(0.0, float(depth_m))
    scale = float(scale)
    requested = min(depth * scale, float(max_offset_m))
    vector = execute.unit(control_pose.approach_axis_base) * requested
    before = np.asarray(control_pose.position_base, dtype=np.float64).reshape(3)
    after = clamp_lift2(before + vector, workspace_bounds)
    applied = after - before
    adjusted = execute.ControlPose(
        position_base=after,
        rotation_base_tool=control_pose.rotation_base_tool,
        quaternion_wxyz=control_pose.quaternion_wxyz,
        approach_axis_base=control_pose.approach_axis_base,
    )
    return adjusted, {
        "enabled": bool(scale > 0 and requested > 0),
        "depth_m": depth,
        "scale": scale,
        "max_offset_m": float(max_offset_m),
        "requested_offset_m": requested,
        "requested_vector_base": vector.tolist(),
        "position_base_before": before.tolist(),
        "position_base_after": after.tolist(),
        "applied_vector_base": applied.tolist(),
        "applied_distance_m": float(np.linalg.norm(applied)),
        "workspace_clamped": not np.allclose(vector, applied, atol=1e-9),
    }


def apply_position_offset_lift2(
    control_pose: execute.ControlPose,
    *,
    offset_base: Iterable[float],
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[execute.ControlPose, dict[str, Any]]:
    offset = np.asarray(list(offset_base), dtype=np.float64).reshape(3)
    before = np.asarray(control_pose.position_base, dtype=np.float64).reshape(3)
    after = clamp_lift2(before + offset, workspace_bounds)
    applied = after - before
    adjusted = execute.ControlPose(
        position_base=after,
        rotation_base_tool=control_pose.rotation_base_tool,
        quaternion_wxyz=control_pose.quaternion_wxyz,
        approach_axis_base=control_pose.approach_axis_base,
    )
    return adjusted, {
        "requested_offset_base": offset.tolist(),
        "position_base_before": before.tolist(),
        "position_base_after": after.tolist(),
        "applied_vector_base": applied.tolist(),
        "applied_distance_m": float(np.linalg.norm(applied)),
        "workspace_clamped": not np.allclose(offset, applied, atol=1e-9),
    }


def build_stage_targets_lift2(
    control_pose: execute.ControlPose,
    motion: execute.MotionConfig,
    *,
    pregrasp_control_pose: execute.ControlPose,
) -> dict[str, dict[str, np.ndarray]]:
    target = clamp_lift2(control_pose.position_base, DEFAULT_LIFT2_WORKSPACE_BOUNDS)
    quaternion = execute.unit(control_pose.quaternion_wxyz)
    approach = execute.unit(control_pose.approach_axis_base)
    pre_anchor = clamp_lift2(pregrasp_control_pose.position_base, DEFAULT_LIFT2_WORKSPACE_BOUNDS)
    pre = clamp_lift2(pre_anchor - approach * float(motion.pregrasp_offset_m), DEFAULT_LIFT2_WORKSPACE_BOUNDS)
    lift = clamp_lift2(target + np.array([0.0, 0.0, float(motion.lift_offset_m)]), DEFAULT_LIFT2_WORKSPACE_BOUNDS)
    return {
        "pre": {"position": pre, "quaternion": quaternion},
        "grasp": {"position": target, "quaternion": quaternion},
        "lift": {"position": lift, "quaternion": quaternion},
    }


def execute_lift2_trajectory(
    *,
    env: Any,
    planned: execute.PlannedStage,
    planner_joint_names: list[str],
    recorder: "VideoRecorder",
    gripper_qpos: float,
    action_repeat: int,
    max_waypoints: int,
) -> dict[str, Any]:
    trajectory = np.asarray(planned.trajectory_positions, dtype=np.float32)
    joint_names = list(planned.trajectory_joint_names)
    indices = [joint_names.index(name) for name in planner_joint_names]
    right_arm_trajectory = trajectory[:, indices]
    right_arm_trajectory = sample_rows(right_arm_trajectory, max_waypoints)
    final_info: dict[str, Any] = {}
    for right_q in right_arm_trajectory:
        action = make_lift2_action_from_right_arm_qpos(
            right_q,
            curobo_joint_names=planner_joint_names,
            gripper_qpos=gripper_qpos,
        )
        for _ in range(max(1, int(action_repeat))):
            _, _, _, _, info = env.step(action)
            final_info = info
            recorder.capture(env)
    return final_info


def sample_rows(arr: np.ndarray, max_rows: int) -> np.ndarray:
    if max_rows <= 0 or arr.shape[0] <= max_rows:
        return arr
    indices = np.linspace(0, arr.shape[0] - 1, int(max_rows)).round().astype(np.int64)
    return arr[indices]


def current_right_tcp_pose(planner: Any, q_start: np.ndarray) -> dict[str, np.ndarray]:
    from curobo.types import JointState

    q_start_tensor = torch.as_tensor(q_start, device="cuda", dtype=torch.float32).unsqueeze(0)
    state = JointState.from_position(q_start_tensor, joint_names=planner.joint_names)
    kin = planner.compute_kinematics(state)
    pose = kin.tool_poses.get_link_pose("right_tcp")
    return {
        "position": pose.position.detach().cpu().numpy().reshape(3),
        "quaternion": pose.quaternion.detach().cpu().numpy().reshape(4),
    }


def lift2_trace_sample(env: Any, label: str) -> dict[str, Any]:
    sample: dict[str, Any] = {"label": label}
    robot = env.unwrapped.agent.robot
    active_joint_names = [joint.name for joint in robot.get_active_joints()]
    qpos = robot.get_qpos().detach().cpu().numpy().reshape(-1)
    sample["active_joint_qpos"] = {
        name: float(value) for name, value in zip(active_joint_names, qpos)
    }
    target_pose = execute.target_object_pose(env)
    if target_pose is not None:
        sample["object_position_world"] = target_pose[:3, 3].tolist()
    tcp = find_link_pose_matrix(env, "right_tcp")
    if tcp is None:
        tcp = find_link_pose_matrix(env, "right_link26")
    if tcp is not None:
        sample["right_tcp_position_world"] = tcp[:3, 3].tolist()
    if target_pose is not None and tcp is not None:
        sample["object_tcp_distance_m"] = float(np.linalg.norm(target_pose[:3, 3] - tcp[:3, 3]))
    return sample


def compute_lift_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [
        np.asarray(row["object_position_world"], dtype=np.float64)
        for row in trace
        if "object_position_world" in row
    ]
    if not positions:
        return {"object_lift_success": False, "failure_reason": "target_object_pose_missing"}
    initial = positions[0]
    final = positions[-1]
    height_delta = float(final[2] - initial[2])
    max_height_delta = float(max(pos[2] for pos in positions) - initial[2])
    success = bool(height_delta >= 0.03)
    return {
        "object_lift_success": success,
        "initial_object_position_world": initial.tolist(),
        "final_object_position_world": final.tolist(),
        "height_delta_m": height_delta,
        "max_height_delta_m": max_height_delta,
        "min_required_lift_m": 0.03,
        "failure_reason": "" if success else "object_not_lifted",
    }


def find_link_pose_matrix(env: Any, link_name: str) -> np.ndarray | None:
    robot = env.unwrapped.agent.robot
    link = execute.find_robot_link(robot, link_name)
    if link is None or getattr(link, "pose", None) is None:
        return None
    try:
        return execute.pose_to_matrix(link.pose, f"{link_name}.pose")
    except Exception:
        return None


def remove_marker_actors(actors: list[Any]) -> None:
    """Remove visual-only markers when a persistent scene is reused."""

    for actor in reversed(actors):
        remove_from_scene = getattr(actor, "remove_from_scene", None)
        if callable(remove_from_scene):
            try:
                remove_from_scene()
            except Exception:
                pass


def resolve_scene_model_path(args: argparse.Namespace) -> Path:
    if args.scene_model is not None:
        return args.scene_model.expanduser().resolve()
    return (
        args.m4c_root.expanduser().resolve()
        / f"seed{int(args.seed):03d}"
        / "real_scene"
        / "curobo_scene_voxel.npz"
    )


def build_lift2_planner(args: argparse.Namespace, scene_model: Any) -> Any:
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    robot_config = yaml.safe_load(
        args.config.expanduser().resolve().read_text(encoding="utf-8")
    )
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_config,
        scene_model=scene_model,
        collision_cache=collision_cache_for_scene(scene_model),
        use_cuda_graph=False,
        num_ik_seeds=int(args.num_ik_seeds),
        num_trajopt_seeds=int(args.num_trajopt_seeds),
    )
    planner = MotionPlanner(planner_cfg)
    planner.warmup(
        enable_graph=False,
        num_warmup_iterations=int(args.warmup_iterations),
    )
    return planner


def build_lift2_execution_env(args: argparse.Namespace, output_dir: Path) -> Any:
    env_args = argparse.Namespace(
        seed=args.seed,
        env_id=args.env_id,
        robot_uid=args.robot_uid,
        camera=args.camera,
        width=args.render_width,
        height=args.render_height,
        settle_steps=args.settle_steps,
        camera_frame=args.camera_frame,
        camera_eye=args.camera_eye,
        camera_target=args.camera_target,
        mask_mode="task-target",
        output_dir=output_dir / "_unused_zg_input",
    )
    return build_env(env_args)


def set_lift2_qpos(raw_env: Any, qpos: np.ndarray) -> None:
    robot = raw_env.agent.robot
    q = torch.as_tensor(qpos, device=raw_env.device, dtype=torch.float32).unsqueeze(0)
    robot.set_qpos(q)
    set_qvel = getattr(robot, "set_qvel", None)
    if callable(set_qvel):
        set_qvel(torch.zeros_like(q))


def clamp_lift2(
    point: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    return np.array(
        [
            np.clip(point[0], bounds[0][0], bounds[0][1]),
            np.clip(point[1], bounds[1][0], bounds[1][1]),
            np.clip(point[2], bounds[2][0], bounds[2][1]),
        ],
        dtype=np.float64,
    )


def set_render_camera(env: Any, eye: list[float], target: list[float]) -> None:
    pose = sapien_utils.look_at(eye=eye, target=target).raw_pose.detach().cpu().numpy().reshape(-1)
    camera = env.unwrapped._human_render_cameras["render_camera"].camera
    camera.set_local_pose(sapien.Pose(p=pose[:3], q=pose[3:]))


class VideoRecorder:
    def __init__(self, output_path: Path | None, fps: int):
        self.output_path = output_path
        self.fps = int(fps)
        self.frame_count = 0
        self._writer: Any | None = None
        self._closed = False

    def capture(self, env: Any) -> None:
        if self.output_path is None:
            return
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                self.output_path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
        self._writer.append_data(normalize_rgb(env.render()))
        self.frame_count += 1

    def save(self, *, allow_empty: bool = False) -> Path | None:
        if self.output_path is None:
            return None
        if self._writer is not None and not self._closed:
            self._writer.close()
            self._closed = True
        if self.frame_count == 0 and not allow_empty:
            raise RuntimeError(f"No frames captured for {self.output_path}.")
        return self.output_path if self.frame_count > 0 else None


def normalize_rgb(frame: Any) -> np.ndarray:
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255.0, 0, 255)
    return frame[..., :3].astype(np.uint8)


def sanitize(value: Any) -> Any:
    return execute.sanitize_for_json(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
