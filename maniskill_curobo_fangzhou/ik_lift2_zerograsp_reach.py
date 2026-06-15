#!/usr/bin/env python3
"""Reach ZeroGrasp Lift2 targets with SAPIEN Pinocchio IK only.

This is a diagnostic path: it intentionally does not construct or call cuRobo.
The goal is to test whether the ZeroGrasp-derived right_tcp pose is kinematically
reachable for Lift2 before adding collision-aware trajectory optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sapien

from maniskill_curobo.scripts import execute_curobo_pick as execute

from .execute_lift2_m4c_grasp import (
    DEFAULT_LIFT2_WORKSPACE_BOUNDS,
    VideoRecorder,
    apply_depth_offset_lift2,
    build_lift2_execution_env,
    build_stage_targets_lift2,
    compute_lift2_control_pose,
    find_link_pose_matrix,
    lift2_trace_sample,
    remove_marker_actors,
    sanitize,
    set_lift2_qpos,
    set_render_camera,
    write_json,
)
from .export_lift2_zerograsp_input import DEFAULT_CAMERA_EYE, DEFAULT_CAMERA_FRAME, DEFAULT_CAMERA_TARGET
from .lift2_constants import LIFT2_CUROBO_SAFE_REST_QPOS, LIFT2_JOINT_NAMES
from .render_lift2_seed import PickClutterYCBLift2Env, PickSingleYCBLift2Env  # noqa: F401


RIGHT_ARM_REVOLUTE_JOINTS = (
    "right_joint21",
    "right_joint22",
    "right_joint23",
    "right_joint24",
    "right_joint25",
    "right_joint26",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--env-id",
        choices=("PickClutterYCBLift2-v1", "PickSingleYCBLift2-v1"),
        default="PickSingleYCBLift2-v1",
    )
    parser.add_argument(
        "--robot-uid",
        choices=PickClutterYCBLift2Env.SUPPORTED_ROBOTS,
        default="lift2_full_collision",
    )
    parser.add_argument(
        "--zerograsp-output",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/lift2_headcam_fixed_lift_seed001/zerograsp_output"),
    )
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument(
        "--approach-axis",
        choices=execute.APPROACH_AXIS_CHOICES,
        default="positive-x",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("grasp", "lift2-gripper", "start"),
        default="lift2-gripper",
    )
    parser.add_argument("--pregrasp-offset", type=float, default=0.16)
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--grasp-depth-scale", type=float, default=1.0)
    parser.add_argument("--grasp-depth-max-offset", type=float, default=0.04)
    parser.add_argument(
        "--grasp-depth-fallback-fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
    )
    parser.add_argument("--initial-lift-joint-value", type=float, default=0.46)
    parser.add_argument(
        "--unlock-lift",
        action="store_true",
        help="Allow joint4 to move in IK. Default keeps the lift fixed at --initial-lift-joint-value.",
    )
    parser.add_argument("--gripper-open", type=float, default=0.03)
    parser.add_argument("--gripper-closed", type=float, default=0.0)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--stage-steps", type=int, default=80)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--ik-max-iterations", type=int, default=2000)
    parser.add_argument("--ik-eps", type=float, default=1e-4)
    parser.add_argument("--ik-dt", type=float, default=0.1)
    parser.add_argument("--ik-damp", type=float, default=1e-6)
    parser.add_argument("--camera-frame", choices=(DEFAULT_CAMERA_FRAME, "robot", "world"), default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-camera-eye", type=float, nargs=3, default=[-0.05, 0.72, 0.40])
    parser.add_argument("--render-camera-target", type=float, nargs=3, default=[0.30, 0.35, 0.12])
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-marker", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_fangzhou/runs/lift2_headcam_fixed_lift_seed001/ik_reach"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    video_path = None if args.no_video else output_dir / "ik_reach.mp4"

    env = build_lift2_execution_env(args, output_dir)
    recorder = VideoRecorder(video_path, fps=args.video_fps)
    manifest: dict[str, Any] = {
        "args": sanitize(vars(args)),
        "mode": "sapien_pinocchio_ik_only_no_curobo",
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "zerograsp_output": str(args.zerograsp_output.expanduser().resolve()),
        "candidate_attempts": [],
    }
    marker_actors: list[Any] = []
    try:
        env.reset(seed=args.seed)
        raw_env = env.unwrapped
        set_render_camera(env, eye=list(args.render_camera_eye), target=list(args.render_camera_target))

        start_qpos = np.asarray(LIFT2_CUROBO_SAFE_REST_QPOS, dtype=np.float32)
        start_qpos[LIFT2_JOINT_NAMES.index("joint4")] = float(args.initial_lift_joint_value)
        for name in ("right_joint27", "right_joint28"):
            start_qpos[LIFT2_JOINT_NAMES.index(name)] = float(args.gripper_open)
        set_lift2_qpos(raw_env, start_qpos)
        for _ in range(max(0, int(args.settle_steps))):
            env.step(start_qpos)
        set_lift2_qpos(raw_env, start_qpos)
        recorder.capture(env)

        robot = raw_env.agent.robot
        link_names = [str(link.name) for link in robot.get_links()]
        right_tcp_index = link_names.index("right_tcp")
        active_qmask = build_active_qmask(unlock_lift=bool(args.unlock_lift))
        manifest["right_tcp_pinocchio_index"] = int(right_tcp_index)
        manifest["active_ik_joints"] = [
            name for name, active in zip(LIFT2_JOINT_NAMES, active_qmask) if int(active)
        ]
        manifest["start_qpos"] = start_qpos.tolist()
        manifest["start_trace"] = lift2_trace_sample(env, "start")

        camera_model = execute.camera_model_matrix(env, args.camera)
        world_from_base = execute.robot_base_matrix(env)
        candidates = execute.load_grasp_candidates(args.zerograsp_output, top_k=args.candidate_top_k)
        if not candidates:
            raise RuntimeError(f"No ZeroGrasp candidates found in {args.zerograsp_output}.")

        selected = search_ik_candidate(
            args=args,
            env=env,
            start_qpos=start_qpos,
            right_tcp_index=right_tcp_index,
            active_qmask=active_qmask,
            candidates=candidates,
            camera_model=camera_model,
            world_from_base=world_from_base,
            manifest=manifest,
        )
        manifest["selected_candidate"] = sanitize(selected)

        if selected is not None and not args.no_marker:
            marker_geometry = execute.build_control_pose_marker_geometry(
                control_pose=selected["control_pose"],
                world_from_base_matrix=world_from_base,
                width_m=float(selected["candidate"].get("width_m", 0.06) or 0.06),
            )
            marker_actors = execute.add_grasp_marker_to_scene(
                raw_env.scene,
                marker_geometry,
                name_prefix="ik_target_marker",
            )
            manifest["target_marker"] = {
                **execute.marker_manifest(marker_geometry, len(marker_actors)),
                "meaning": "IK-selected full-depth right_tcp target",
            }

        if selected is not None:
            execute_ik_waypoints(
                env=env,
                recorder=recorder,
                start_qpos=start_qpos,
                pre_qpos=np.asarray(selected["pre_ik"]["qpos"], dtype=np.float32),
                grasp_qpos=np.asarray(selected["grasp_ik"]["qpos"], dtype=np.float32),
                gripper_open=float(args.gripper_open),
                gripper_closed=float(args.gripper_closed),
                stage_steps=int(args.stage_steps),
                hold_steps=int(args.hold_steps),
                close_steps=int(args.close_steps),
                action_repeat=int(args.action_repeat),
            )
            manifest["final_trace"] = lift2_trace_sample(env, "final")
            manifest["final_errors"] = compute_stage_errors(
                env,
                world_from_base=world_from_base,
                targets=selected["targets"],
            )
            manifest["status"] = "ik_executed"
        else:
            manifest["status"] = "ik_failed_all_candidates"
            # Save one frame with whatever diagnostics are available.
            for _ in range(max(1, int(args.hold_steps))):
                env.step(start_qpos)
                recorder.capture(env)

        video_saved = recorder.save(allow_empty=True)
        manifest["video_saved"] = str(video_saved) if video_saved else None
        write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            video_saved = recorder.save(allow_empty=True)
            manifest["video_saved"] = str(video_saved) if video_saved else None
        except Exception as video_exc:
            manifest["video_error"] = f"{type(video_exc).__name__}: {video_exc}"
        write_json(manifest_path, manifest)
        raise
    finally:
        remove_marker_actors(marker_actors)
        env.close()

    print(json.dumps({"manifest": str(manifest_path), "video": str(video_path)}, ensure_ascii=False, indent=2))
    return 0


def build_active_qmask(*, unlock_lift: bool) -> np.ndarray:
    mask = np.zeros(len(LIFT2_JOINT_NAMES), dtype=np.int32)
    if unlock_lift:
        mask[LIFT2_JOINT_NAMES.index("joint4")] = 1
    for name in RIGHT_ARM_REVOLUTE_JOINTS:
        mask[LIFT2_JOINT_NAMES.index(name)] = 1
    return mask


def search_ik_candidate(
    *,
    args: argparse.Namespace,
    env: Any,
    start_qpos: np.ndarray,
    right_tcp_index: int,
    active_qmask: np.ndarray,
    candidates: list[execute.GraspRecord],
    camera_model: np.ndarray,
    world_from_base: np.ndarray,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
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
    start_quaternion = current_tcp_quaternion_base(env, world_from_base)
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
        pre_ik = solve_ik(
            env,
            link_index=right_tcp_index,
            target=pre_targets["pre"],
            initial_qpos=start_qpos,
            active_qmask=active_qmask,
            args=args,
        )
        pre_attempt = {
            "rank": int(rank),
            "score": float(grasp.score),
            "source": grasp.source,
            "object_id": grasp.object_id,
            "stage": "pre",
            "depth_scale": None,
            "zero_depth_position_base": zero_pose.position_base.tolist(),
            "pre_target_position_base": pre_targets["pre"]["position"].tolist(),
            "pre_target_quaternion_wxyz": pre_targets["pre"]["quaternion"].tolist(),
            "ik": summarize_ik(pre_ik),
        }
        manifest["candidate_attempts"].append(sanitize(pre_attempt))
        if not pre_ik["success"]:
            continue
        for depth_scale in depth_scales:
            control_pose, depth_manifest = apply_depth_offset_lift2(
                zero_pose,
                depth_m=grasp.depth_m,
                scale=depth_scale,
                max_offset_m=args.grasp_depth_max_offset,
                workspace_bounds=DEFAULT_LIFT2_WORKSPACE_BOUNDS,
            )
            targets = build_stage_targets_lift2(control_pose, motion, pregrasp_control_pose=zero_pose)
            grasp_ik = solve_ik(
                env,
                link_index=right_tcp_index,
                target=targets["grasp"],
                initial_qpos=np.asarray(pre_ik["qpos"], dtype=np.float64),
                active_qmask=active_qmask,
                args=args,
            )
            attempt = {
                "rank": int(rank),
                "score": float(grasp.score),
                "source": grasp.source,
                "object_id": grasp.object_id,
                "stage": "grasp",
                "depth_scale": float(depth_scale),
                "depth_offset": depth_manifest,
                "zero_depth_position_base": zero_pose.position_base.tolist(),
                "grasp_position_base": control_pose.position_base.tolist(),
                "grasp_quaternion_wxyz": control_pose.quaternion_wxyz.tolist(),
                "approach_axis_base": control_pose.approach_axis_base.tolist(),
                "ik": summarize_ik(grasp_ik),
            }
            manifest["candidate_attempts"].append(sanitize(attempt))
            if grasp_ik["success"]:
                return {
                    "rank": int(rank),
                    "score": float(grasp.score),
                    "source": grasp.source,
                    "object_id": grasp.object_id,
                    "depth_scale": float(depth_scale),
                    "candidate": {
                        "width_m": float(grasp.width_m),
                        "height_m": float(grasp.height_m),
                        "depth_m": float(grasp.depth_m),
                        "translation_m_camera": grasp.translation_m_camera.tolist(),
                        "rotation_matrix_camera": grasp.rotation_matrix_camera.tolist(),
                    },
                    "zero_pose": zero_pose,
                    "control_pose": control_pose,
                    "targets": targets,
                    "pre_ik": pre_ik,
                    "grasp_ik": grasp_ik,
                }
    return None


def solve_ik(
    env: Any,
    *,
    link_index: int,
    target: dict[str, np.ndarray],
    initial_qpos: np.ndarray,
    active_qmask: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pin = env.unwrapped.agent.robot.create_pinocchio_model()
    qpos, success, error = pin.compute_inverse_kinematics(
        int(link_index),
        sapien.Pose(
            p=np.asarray(target["position"], dtype=np.float64).reshape(3),
            q=np.asarray(target["quaternion"], dtype=np.float64).reshape(4),
        ),
        initial_qpos=np.asarray(initial_qpos, dtype=np.float64).reshape(-1),
        active_qmask=np.asarray(active_qmask, dtype=np.int32).reshape(-1),
        eps=float(args.ik_eps),
        max_iterations=int(args.ik_max_iterations),
        dt=float(args.ik_dt),
        damp=float(args.ik_damp),
    )
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    return {
        "success": bool(success),
        "qpos": qpos.tolist(),
        "error": error.tolist(),
        "error_norm": float(np.linalg.norm(error)),
        "active_joint_qpos": {
            name: float(qpos[LIFT2_JOINT_NAMES.index(name)])
            for name, active in zip(LIFT2_JOINT_NAMES, active_qmask)
            if int(active)
        },
    }


def summarize_ik(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(result["success"]),
        "error": result["error"],
        "error_norm": float(result["error_norm"]),
        "active_joint_qpos": result["active_joint_qpos"],
    }


def execute_ik_waypoints(
    *,
    env: Any,
    recorder: VideoRecorder,
    start_qpos: np.ndarray,
    pre_qpos: np.ndarray,
    grasp_qpos: np.ndarray,
    gripper_open: float,
    gripper_closed: float,
    stage_steps: int,
    hold_steps: int,
    close_steps: int,
    action_repeat: int,
) -> None:
    current = np.asarray(start_qpos, dtype=np.float32).copy()
    current = set_gripper(current, gripper_open)
    for goal in (pre_qpos, grasp_qpos):
        goal = set_gripper(np.asarray(goal, dtype=np.float32), gripper_open)
        for action in interpolate_qpos(current, goal, max(1, int(stage_steps))):
            step_action(env, recorder, action, action_repeat)
        current = goal
        for _ in range(max(0, int(hold_steps))):
            step_action(env, recorder, current, action_repeat)
    closed = set_gripper(current.copy(), gripper_closed)
    for action in interpolate_qpos(current, closed, max(1, int(close_steps))):
        step_action(env, recorder, action, action_repeat)
    for _ in range(max(0, int(hold_steps))):
        step_action(env, recorder, closed, action_repeat)


def interpolate_qpos(start: np.ndarray, goal: np.ndarray, steps: int) -> list[np.ndarray]:
    start = np.asarray(start, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    if steps <= 1:
        return [goal]
    return [
        (start * (1.0 - alpha) + goal * alpha).astype(np.float32)
        for alpha in np.linspace(0.0, 1.0, int(steps), endpoint=True)[1:]
    ]


def step_action(env: Any, recorder: VideoRecorder, action: np.ndarray, repeats: int) -> None:
    for _ in range(max(1, int(repeats))):
        env.step(np.asarray(action, dtype=np.float32))
        recorder.capture(env)


def set_gripper(qpos: np.ndarray, value: float) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float32).reshape(-1).copy()
    for name in ("right_joint27", "right_joint28"):
        q[LIFT2_JOINT_NAMES.index(name)] = float(value)
    return q


def current_tcp_quaternion_base(env: Any, world_from_base: np.ndarray) -> np.ndarray:
    tcp_world = find_link_pose_matrix(env, "right_tcp")
    if tcp_world is None:
        raise RuntimeError("right_tcp pose missing")
    base_from_world = np.linalg.inv(execute.matrix4(world_from_base, "world_from_base"))
    tcp_base = base_from_world @ tcp_world
    return execute.quat_wxyz_from_matrix(tcp_base[:3, :3])


def compute_stage_errors(
    env: Any,
    *,
    world_from_base: np.ndarray,
    targets: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    tcp_world = find_link_pose_matrix(env, "right_tcp")
    if tcp_world is None:
        return {"right_tcp_pose_missing": True}
    base_from_world = np.linalg.inv(execute.matrix4(world_from_base, "world_from_base"))
    tcp_base = base_from_world @ tcp_world
    tcp_position = tcp_base[:3, 3]
    grasp_target = np.asarray(targets["grasp"]["position"], dtype=np.float64).reshape(3)
    return {
        "right_tcp_position_base": tcp_position.tolist(),
        "grasp_target_position_base": grasp_target.tolist(),
        "position_error_to_grasp_m": float(np.linalg.norm(tcp_position - grasp_target)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
