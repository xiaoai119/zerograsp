#!/usr/bin/env python3
"""Execute a ZeroGrasp target with cuRobo joint-space planning in ManiSkill."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from maniskill_curobo.joint_trajectory_utils import (
    make_pd_joint_pos_actions,
    ordered_values,
    sample_waypoints,
    squeeze_trajectory_positions,
)
from maniskill_curobo.scene_export import (
    export_maniskill_scene_to_curobo,
    target_segmentation_ids_from_zerograsp_scene,
)


OPENCV_TO_SAPIEN_CAMERA = np.diag([1.0, -1.0, -1.0])
APPROACH_AXIS_CHOICES = ("negative-x", "positive-x", "flip-world-z")
DEFAULT_CAMERA_EYE = (-0.30, 0.0, 0.55)
DEFAULT_CAMERA_TARGET = (0.05, 0.0, 0.08)
DEFAULT_WORKSPACE_Z_MIN = 0.02
DEFAULT_WORKSPACE_BOUNDS = ((0.25, 0.85), (-0.45, 0.45), (0.02, 0.60))


@dataclass(frozen=True)
class GraspRecord:
    score: float
    width_m: float
    height_m: float
    depth_m: float
    rotation_matrix_camera: np.ndarray
    translation_m_camera: np.ndarray
    source: str
    object_index: int | None = None
    object_id: int | None = None


@dataclass(frozen=True)
class ControlPose:
    position_base: np.ndarray
    rotation_base_tool: np.ndarray
    quaternion_wxyz: np.ndarray
    approach_axis_base: np.ndarray
    planner_position_base: np.ndarray | None = None
    planner_rotation_base_tool: np.ndarray | None = None
    planner_quaternion_wxyz: np.ndarray | None = None


@dataclass(frozen=True)
class MotionConfig:
    pregrasp_offset_m: float = 0.10
    lift_offset_m: float = 0.15
    workspace_z_min: float = DEFAULT_WORKSPACE_Z_MIN
    gripper_open: float = 1.0
    gripper_closed: float = -1.0


@dataclass(frozen=True)
class PlannedStage:
    name: str
    target_position: np.ndarray
    target_quaternion: np.ndarray
    trajectory_positions: np.ndarray
    trajectory_joint_names: list[str]
    output_path: Path


@dataclass(frozen=True)
class GraspMarkerGeometry:
    center_world: np.ndarray
    approach_axis_world: np.ndarray
    width_axis_world: np.ndarray
    approach_end_world: np.ndarray
    width_endpoints_world: tuple[np.ndarray, np.ndarray]
    width_m: float


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
        if self._closed:
            raise RuntimeError(f"Cannot capture after video writer was closed: {self.output_path}")
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            import imageio.v2 as iio

            self._writer = iio.get_writer(str(self.output_path), fps=self.fps)
        self._writer.append_data(normalize_rgb_frame(env.render()))
        self.frame_count += 1

    def save(self, *, allow_empty: bool = False) -> Path | None:
        if self.output_path is None:
            return None
        if self.frame_count == 0:
            if not self._closed and self._writer is not None:
                self._writer.close()
                self._closed = True
            if allow_empty:
                return None
            raise RuntimeError(f"No frames captured for {self.output_path}.")
        if not self._closed and self._writer is not None:
            self._writer.close()
            self._closed = True
        return self.output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan pre-grasp, grasp, and lift stages with cuRobo, then execute the "
            "resulting joint trajectory in ManiSkill."
        )
    )
    parser.add_argument(
        "--zerograsp-output",
        default=None,
        help="Directory containing recommended_grasp_top1.json or raw_outputs/*.grasp.npy.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=1,
        help=(
            "Consider this many model-ranked grasp candidates and select the first "
            "candidate whose pre-grasp and grasp stages are both cuRobo-plannable. "
            "A value of 1 preserves the original top1-only behavior."
        ),
    )
    parser.add_argument(
        "--target-base",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Debug path: bypass ZeroGrasp and plan directly to this base-frame target.",
    )
    parser.add_argument(
        "--target-quat-wxyz",
        type=float,
        nargs=4,
        default=[1.0, 0.0, 0.0, 0.0],
        metavar=("W", "X", "Y", "Z"),
        help="Debug target orientation for --target-base.",
    )
    parser.add_argument(
        "--target-approach",
        type=float,
        nargs=3,
        default=[0.0, 0.0, -1.0],
        metavar=("X", "Y", "Z"),
        help="Debug approach direction for --target-base.",
    )
    parser.add_argument("--env-id", default="PickSingleYCB-v1", help="ManiSkill environment id.")
    parser.add_argument("--seed", type=int, default=1, help="ManiSkill reset seed.")
    parser.add_argument("--camera", default="base_camera", help="ManiSkill sensor name.")
    parser.add_argument("--width", type=int, default=1280, help="Sensor image width.")
    parser.add_argument("--height", type=int, default=1024, help="Sensor image height.")
    parser.add_argument(
        "--mask-mode",
        choices=("task-target", "all-objects", "visible-area"),
        default="task-target",
        help="Which ManiSkill segmentation ids to save as the ZeroGrasp input mask.",
    )
    parser.add_argument("--render-width", type=int, default=1280, help="MP4 render width.")
    parser.add_argument("--render-height", type=int, default=1024, help="MP4 render height.")
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=list(DEFAULT_CAMERA_EYE),
        metavar=("X", "Y", "Z"),
        help="World-frame camera position for the ZeroGrasp sensor.",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=list(DEFAULT_CAMERA_TARGET),
        metavar=("X", "Y", "Z"),
        help="World-frame point the ZeroGrasp sensor looks at.",
    )
    parser.add_argument(
        "--approach-axis",
        choices=APPROACH_AXIS_CHOICES,
        default="positive-x",
        help="ZeroGrasp local axis convention used when converting grasp orientation.",
    )
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--workspace-z-min", type=float, default=DEFAULT_WORKSPACE_Z_MIN)
    parser.add_argument(
        "--grasp-depth-scale",
        type=float,
        default=0.0,
        help=(
            "Move the commanded TCP target forward along the ZeroGrasp approach axis "
            "by depth_m times this scale. Use 1.0 to apply the full predicted depth; "
            "the default 0.0 preserves the legacy translation-only behavior."
        ),
    )
    parser.add_argument(
        "--grasp-depth-max-offset",
        type=float,
        default=0.04,
        help="Maximum TCP position offset produced by --grasp-depth-scale, in meters.",
    )
    parser.add_argument(
        "--grasp-depth-auto-fallback",
        action="store_true",
        help=(
            "If grasp planning fails, retry progressively shallower depth offsets. "
            "The pre-grasp target remains anchored to the zero-depth grasp."
        ),
    )
    parser.add_argument(
        "--grasp-depth-fallback-fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
        help=(
            "Fractions of --grasp-depth-scale attempted by automatic fallback. "
            "The requested scale is always tried first."
        ),
    )
    parser.add_argument(
        "--no-hand-tcp-calibration",
        dest="use_hand_tcp_calibration",
        action="store_false",
        help="Do not convert the ZeroGrasp TCP/grasp-center target to cuRobo's panda_hand frame.",
    )
    parser.set_defaults(use_hand_tcp_calibration=True)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-closed", type=float, default=-1.0)
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument(
        "--settle-before-export-steps",
        type=int,
        default=0,
        help="Hold the robot still for this many env steps after reset before saving ZeroGrasp input or planning.",
    )
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-waypoints-per-stage", type=int, default=120)
    parser.add_argument(
        "--stop-after-stage",
        choices=("pre", "grasp", "lift"),
        default=None,
        help="Stop after this motion stage and save the partial execution video.",
    )
    parser.add_argument(
        "--planning-only",
        action="store_true",
        help=(
            "Plan pre-grasp, grasp, and lift from chained planned joint states "
            "without stepping the ManiSkill environment."
        ),
    )
    parser.add_argument("--robot-config", default="franka.yml", help="cuRobo robot config.")
    parser.add_argument("--scene-model", default="collision_test.yml", help="cuRobo scene model.")
    parser.add_argument(
        "--scene-source",
        choices=("maniskill", "fixed"),
        default="maniskill",
        help=(
            "Use a per-run cuRobo scene exported from ManiSkill, or the fixed "
            "--scene-model config."
        ),
    )
    parser.add_argument(
        "--scene-include-target-object",
        action="store_true",
        help="Keep the ZeroGrasp target object as a cuRobo obstacle.",
    )
    parser.add_argument(
        "--scene-min-cuboid-dimension",
        type=float,
        default=0.005,
        help="Minimum cuboid dimension when approximating ManiSkill collision shapes.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-out", default=None, help="Optional explicit MP4 path.")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video frame capture and MP4 writing for faster rollouts.",
    )
    parser.add_argument(
        "--no-grasp-marker",
        dest="show_grasp_marker",
        action="store_false",
        help="Do not draw the ZeroGrasp center/approach/width marker in the scene video.",
    )
    parser.set_defaults(show_grasp_marker=True)
    parser.add_argument(
        "--marker-grasp-center-base",
        type=float,
        nargs=3,
        default=None,
        help=(
            "Optional extra grasp-center marker in robot base coordinates. "
            "Useful with --target-base, where the target is the panda_hand goal."
        ),
    )
    parser.add_argument(
        "--marker-width",
        type=float,
        default=None,
        help="Optional gripper width used for debug markers.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for all outputs. Defaults to maniskill_curobo/runs/<timestamp>.",
    )
    parser.add_argument(
        "--fail-on-task-success",
        action="store_true",
        help="Stop early if ManiSkill reports success during execution.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "planning_diagnostics.json"
    if diagnostics_path.exists():
        diagnostics_path.unlink()
    video_path = None if args.no_video else Path(args.video_out).expanduser().resolve() if args.video_out else output_dir / "execution.mp4"

    env = build_env(args)
    recorder = None if args.no_video else VideoRecorder(video_path, fps=args.video_fps)
    manifest: dict[str, Any] = {
        "args": vars(args),
        "output_dir": str(output_dir),
        "zg_input_dir": str(output_dir / "zg_input"),
        "zg_output_dir": str(output_dir / "zg_output") if args.zerograsp_output else None,
        "projection_path": str(output_dir / "grasp_projection.png") if args.zerograsp_output else None,
        "video_path": str(video_path) if video_path else None,
        "video_enabled": not bool(args.no_video),
        "planning_diagnostics_path": str(diagnostics_path),
        "planning_only": bool(args.planning_only),
        "stages": [],
    }
    object_trace: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    try:
        obs, _ = env.reset(seed=args.seed)
        robot = env.unwrapped.agent.robot
        active_joint_names = [joint.name for joint in robot.get_active_joints()]
        print(f"active_joint_names={active_joint_names}")
        object_trace.append(object_lift_trace_sample(env, "reset"))
        if args.settle_before_export_steps > 0:
            obs, settle_info = settle_environment(
                env,
                steps=args.settle_before_export_steps,
                recorder=recorder,
                gripper=args.gripper_open,
            )
            manifest["settle_before_export"] = {
                "steps": int(args.settle_before_export_steps),
                "final_info": sanitize_for_json(settle_info),
            }
            object_trace.append(object_lift_trace_sample(env, "after_settle_before_export"))

        zg_input_dir = save_current_zerograsp_input(obs, env, args, output_dir / "zg_input")
        print(f"zg_input={zg_input_dir}")
        effective_zerograsp_output = None
        if args.zerograsp_output:
            effective_zerograsp_output = copy_zerograsp_output(
                args.zerograsp_output,
                output_dir / "zg_output",
            )
            print(f"zg_output={effective_zerograsp_output}")
            projection_path = save_grasp_projection(
                input_dir=zg_input_dir,
                zerograsp_output_dir=effective_zerograsp_output,
                output_path=output_dir / "grasp_projection.png",
                approach_axis=args.approach_axis,
            )
            manifest["projection_path"] = str(projection_path)
            print(f"grasp_projection={projection_path}")

        control_pose, zero_depth_control_pose, grasp_manifest = load_control_pose(
            env,
            args,
            effective_zerograsp_output,
        )
        manifest["grasp"] = grasp_manifest
        scene_model = prepare_curobo_scene_model(
            env=env,
            args=args,
            output_dir=output_dir,
            zg_input_dir=zg_input_dir,
            grasp_manifest=grasp_manifest,
        )
        manifest["curobo_scene"] = scene_model["manifest"]
        marker_actors = []
        if args.show_grasp_marker and int(args.candidate_top_k) <= 1:
            marker_width = float(args.marker_width if args.marker_width is not None else grasp_manifest.get("width_m", 0.08) or 0.08)
            if args.marker_grasp_center_base is None:
                marker_geometry = build_control_pose_marker_geometry(
                    control_pose=control_pose,
                    world_from_base_matrix=robot_base_matrix(env),
                    width_m=marker_width,
                )
                marker_actors = add_grasp_marker_to_scene(env.unwrapped.scene, marker_geometry)
                manifest["grasp_marker"] = marker_manifest(marker_geometry, len(marker_actors))
                print(
                    "grasp_marker "
                    f"center={np.round(marker_geometry.center_world, 4).tolist()} "
                    f"approach={np.round(marker_geometry.approach_axis_world, 4).tolist()} "
                    f"width_axis={np.round(marker_geometry.width_axis_world, 4).tolist()}"
                )
            else:
                manifest["grasp_marker"] = {
                    "enabled": False,
                    "reason": "suppressed_when_grasp_center_marker_is_available",
                }
            if args.marker_grasp_center_base is not None:
                grasp_center_pose = ControlPose(
                    position_base=clamp_base_target(
                        np.asarray(args.marker_grasp_center_base, dtype=np.float64).reshape(3),
                        args.workspace_z_min,
                    ),
                    rotation_base_tool=control_pose.rotation_base_tool,
                    quaternion_wxyz=control_pose.quaternion_wxyz,
                    approach_axis_base=control_pose.approach_axis_base,
                )
                grasp_center_geometry = build_control_pose_marker_geometry(
                    control_pose=grasp_center_pose,
                    world_from_base_matrix=robot_base_matrix(env),
                    width_m=marker_width,
                    approach_length=0.05,
                )
                center_actors = add_grasp_marker_to_scene(
                    env.unwrapped.scene,
                    grasp_center_geometry,
                    name_prefix="zerograsp_grasp_center_marker",
                    center_color=[0.0, 1.0, 1.0, 0.9],
                    approach_color=[1.0, 0.4, 0.0, 0.9],
                    width_color=[0.75, 0.0, 1.0, 0.9],
                )
                marker_actors.extend(center_actors)
                manifest["grasp_center_marker"] = marker_manifest(grasp_center_geometry, len(center_actors))
                print(
                    "grasp_center_marker "
                    f"center={np.round(grasp_center_geometry.center_world, 4).tolist()} "
                    f"approach={np.round(grasp_center_geometry.approach_axis_world, 4).tolist()} "
                    f"width_axis={np.round(grasp_center_geometry.width_axis_world, 4).tolist()}"
                )
        else:
            manifest["grasp_marker"] = {
                "enabled": False,
                "reason": (
                    "suppressed_during_topk_candidate_selection"
                    if args.show_grasp_marker and int(args.candidate_top_k) > 1
                    else "disabled_by_args"
                ),
            }
        motion = MotionConfig(
            pregrasp_offset_m=args.pregrasp_offset,
            lift_offset_m=args.lift_offset,
            workspace_z_min=args.workspace_z_min,
            gripper_open=args.gripper_open,
            gripper_closed=args.gripper_closed,
        )
        targets = build_stage_targets(
            control_pose,
            motion,
            pregrasp_control_pose=zero_depth_control_pose,
        )
        manifest["stage_targets"] = {
            name: {
                "position_base": target["position"].tolist(),
                "quaternion_wxyz": target["quaternion"].tolist(),
            }
            for name, target in targets.items()
        }

        planner = build_planner(args, scene_model=scene_model["planner_scene_model"])
        arm_action_joint_names = [name for name in active_joint_names if name in planner.joint_names]
        if len(arm_action_joint_names) != len(planner.joint_names):
            raise RuntimeError(
                "ManiSkill arm action joints do not match cuRobo planner joints: "
                f"action={arm_action_joint_names}, curobo={planner.joint_names}"
            )

        qpos = first_vector(robot.get_qpos(), "robot qpos")
        q_start = ordered_values(active_joint_names, qpos, planner.joint_names).astype(np.float32)
        if (
            effective_zerograsp_output is not None
            and int(args.candidate_top_k) > 1
            and args.target_base is None
        ):
            control_pose, zero_depth_control_pose, grasp_manifest, targets = (
                select_plannable_grasp_candidate(
                    env=env,
                    args=args,
                    zerograsp_output_dir=effective_zerograsp_output,
                    planner=planner,
                    q_start=q_start,
                    motion=motion,
                    output_dir=output_dir,
                    diagnostics_path=diagnostics_path,
                )
            )
            manifest["grasp"] = grasp_manifest
            manifest["stage_targets"] = {
                name: {
                    "position_base": target["position"].tolist(),
                    "quaternion_wxyz": target["quaternion"].tolist(),
                }
                for name, target in targets.items()
            }
        if recorder is not None:
            recorder.capture(env)

        stages: list[tuple[str, float]] = [
            ("pre", motion.gripper_open),
            ("grasp", motion.gripper_open),
            ("lift", motion.gripper_closed),
        ]
        for stage_name, gripper in stages:
            selection_depth_scale = (
                (grasp_manifest.get("candidate_selection") or {}).get("selected_depth_scale")
            )
            if (
                stage_name == "grasp"
                and args.grasp_depth_auto_fallback
                and selection_depth_scale is None
            ):
                grasp_manifest.setdefault(
                    "grasp_depth_offset_requested",
                    dict(grasp_manifest.get("grasp_depth_offset") or {}),
                )
                attempts = []
                planned = None
                selected_pose = None
                selected_offset = None
                for depth_scale in grasp_depth_attempt_scales(
                    args.grasp_depth_scale,
                    args.grasp_depth_fallback_fractions,
                ):
                    candidate_pose, candidate_offset = apply_grasp_depth_offset(
                        zero_depth_control_pose,
                        depth_m=float(grasp_manifest.get("depth_m", 0.0) or 0.0),
                        scale=depth_scale,
                        max_offset_m=args.grasp_depth_max_offset,
                        workspace_z_min=args.workspace_z_min,
                    )
                    candidate_targets = build_stage_targets(
                        candidate_pose,
                        motion,
                        pregrasp_control_pose=zero_depth_control_pose,
                    )
                    candidate_target = candidate_targets["grasp"]
                    try:
                        planned = plan_stage(
                            planner=planner,
                            q_start=q_start,
                            target_position=candidate_target["position"],
                            target_quaternion=candidate_target["quaternion"],
                            output_dir=output_dir,
                            stage_name=stage_name,
                            diagnostics_path=diagnostics_path,
                            diagnostic_metadata={
                                "depth_scale": depth_scale,
                                "depth_offset_m": candidate_offset["applied_distance_m"],
                            },
                        )
                    except RuntimeError as exc:
                        attempts.append(
                            {
                                "scale": depth_scale,
                                "offset_m": candidate_offset["applied_distance_m"],
                                "success": False,
                                "failure_reason": str(exc),
                            }
                        )
                        continue
                    attempts.append(
                        {
                            "scale": depth_scale,
                            "offset_m": candidate_offset["applied_distance_m"],
                            "success": True,
                        }
                    )
                    selected_pose = candidate_pose
                    selected_offset = candidate_offset
                    targets = candidate_targets
                    break
                if planned is None or selected_pose is None or selected_offset is None:
                    grasp_manifest["grasp_depth_auto_fallback"] = {
                        "enabled": True,
                        "requested_scale": float(args.grasp_depth_scale),
                        "attempts": attempts,
                        "selected_scale": None,
                    }
                    raise RuntimeError(
                        "cuRobo planning failed for stage=grasp after depth fallback"
                    )
                update_grasp_manifest_pose(grasp_manifest, selected_pose, selected_offset)
                grasp_manifest["grasp_depth_auto_fallback"] = {
                    "enabled": True,
                    "requested_scale": float(args.grasp_depth_scale),
                    "attempts": attempts,
                    "selected_scale": float(selected_offset["scale"]),
                    "selected_offset_m": float(selected_offset["applied_distance_m"]),
                }
                manifest["stage_targets"] = {
                    name: {
                        "position_base": target["position"].tolist(),
                        "quaternion_wxyz": target["quaternion"].tolist(),
                    }
                    for name, target in targets.items()
                }
                print(
                    "grasp_depth_selected "
                    f"scale={selected_offset['scale']:.3f} "
                    f"offset={selected_offset['applied_distance_m']:.4f}"
                )
            else:
                target = targets[stage_name]
                planned = plan_stage(
                    planner=planner,
                    q_start=q_start,
                    target_position=target["position"],
                    target_quaternion=target["quaternion"],
                    output_dir=output_dir,
                    stage_name=stage_name,
                    diagnostics_path=diagnostics_path,
                )
            manifest["stages"].append(stage_manifest(planned))
            q_start = ordered_values(
                planned.trajectory_joint_names,
                planned.trajectory_positions[-1],
                planner.joint_names,
            ).astype(np.float32)
            if args.planning_only:
                continue

            actions = make_pd_joint_pos_actions(
                trajectory=planned.trajectory_positions,
                trajectory_joint_names=planned.trajectory_joint_names,
                arm_action_joint_names=arm_action_joint_names,
                gripper=gripper,
            )
            actions = sample_waypoints(actions, args.max_waypoints_per_stage)
            final_info = execute_action_waypoints(
                env=env,
                actions=actions,
                recorder=recorder,
                action_repeat=args.action_repeat,
                stop_on_success=args.fail_on_task_success,
            )
            object_trace.append(object_lift_trace_sample(env, f"after_{stage_name}"))
            last_executed_actions = actions
            if stage_name == "grasp":
                close_actions = repeat_last_action(actions, args.close_steps)
                if len(close_actions) > 0:
                    close_actions[:, -1] = motion.gripper_closed
                    final_info = execute_action_waypoints(
                        env=env,
                        actions=close_actions,
                        recorder=recorder,
                        action_repeat=1,
                        stop_on_success=args.fail_on_task_success,
                    )
                    object_trace.append(object_lift_trace_sample(env, "after_gripper_close"))
                    last_executed_actions = close_actions
            if args.settle_steps > 0:
                hold_actions = repeat_last_action(last_executed_actions, args.settle_steps)
                final_info = execute_action_waypoints(
                    env=env,
                    actions=hold_actions,
                    recorder=recorder,
                    action_repeat=1,
                    stop_on_success=args.fail_on_task_success,
                )
                object_trace.append(object_lift_trace_sample(env, f"after_{stage_name}_settle"))
            if args.stop_after_stage == stage_name:
                manifest["stopped_after_stage"] = stage_name
                print(f"stopped_after_stage={stage_name}")
                break

        video_saved = recorder.save() if recorder is not None else None
        manifest["object_lift_trace"] = object_trace
        if args.planning_only:
            manifest["planning_preflight_success"] = len(manifest["stages"]) == len(stages)
            manifest["object_lift_metrics"] = None
        else:
            manifest["object_lift_metrics"] = compute_object_lift_metrics(object_trace)
        manifest["final_info"] = sanitize_for_json(final_info)
        manifest["video_saved"] = str(video_saved) if video_saved else None
        write_json(output_dir / "run_manifest.json", manifest)
        print(f"run_manifest={output_dir / 'run_manifest.json'}")
        print(f"video={video_saved}")
    except Exception as exc:
        manifest["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["failure_reason"] = str(exc)
        if args.planning_only:
            manifest["planning_preflight_success"] = False
        try:
            if recorder is not None:
                recorder.capture(env)
        except Exception as capture_exc:
            manifest["failure_capture_error"] = f"{type(capture_exc).__name__}: {capture_exc}"
        try:
            if object_trace:
                object_trace.append(object_lift_trace_sample(env, "exception"))
        except Exception as trace_exc:
            manifest["failure_trace_error"] = f"{type(trace_exc).__name__}: {trace_exc}"
        video_saved = recorder.save(allow_empty=True) if recorder is not None else None
        manifest["object_lift_trace"] = object_trace
        manifest["object_lift_metrics"] = compute_object_lift_metrics(object_trace) if object_trace else {}
        manifest["final_info"] = sanitize_for_json(final_info)
        manifest["video_saved"] = str(video_saved) if video_saved else None
        manifest["partial_video_saved"] = bool(video_saved)
        write_json(output_dir / "run_manifest.json", manifest)
        print(f"partial_run_manifest={output_dir / 'run_manifest.json'}")
        print(f"partial_video={video_saved}")
        raise
    finally:
        env.close()
    return 0


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.env_id.replace("-", "_").replace("/", "_").lower()
    return Path(__file__).resolve().parents[1] / "runs" / f"{label}_seed{args.seed}_curobo_{timestamp}"


def build_env(args: argparse.Namespace) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils import sapien_utils

    sensor_configs: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        args.camera: {
            "pose": sapien_utils.look_at(list(args.camera_eye), list(args.camera_target)),
        },
    }
    return gym.make(
        args.env_id,
        render_mode="rgb_array",
        control_mode="pd_joint_pos",
        robot_uids="panda",
        obs_mode="sensor_data",
        max_episode_steps=1000,
        sensor_configs=sensor_configs,
        human_render_camera_configs={"width": args.render_width, "height": args.render_height},
    )


def settle_environment(
    env: Any,
    *,
    steps: int,
    recorder: VideoRecorder | None = None,
    gripper: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance the simulator with a hold action so initially unstable objects can settle."""

    obs: dict[str, Any] = {}
    info: dict[str, Any] = {}
    if steps <= 0:
        return obs, info
    action = hold_action(env, gripper=gripper)
    for _ in range(int(steps)):
        obs, _, _, _, info = env.step(action[None, :])
        if recorder is not None:
            recorder.capture(env)
    return obs, info


def hold_action(env: Any, *, gripper: float = 1.0) -> np.ndarray:
    dim = int(np.prod(getattr(env.action_space, "shape", (0,))))
    if dim <= 0:
        raise ValueError("Cannot infer action dimension for settle hold action.")
    action = np.zeros(dim, dtype=np.float32)
    if dim >= 8:
        try:
            robot = env.unwrapped.agent.robot
            active_joint_names = [joint.name for joint in robot.get_active_joints()]
            qpos = first_vector(robot.get_qpos(), "robot qpos")
            arm_names = [f"panda_joint{i}" for i in range(1, 8)]
            action[:7] = ordered_values(active_joint_names, qpos, arm_names).astype(np.float32)
            action[7] = float(gripper)
        except Exception:
            action[-1] = float(gripper)
    elif dim >= 1:
        action[-1] = float(gripper)
    return action


def save_current_zerograsp_input(
    obs: dict[str, Any],
    env: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    """Save the exact RGB-D, mask, camera, and scene metadata for this run."""

    from maniskill_codex.zerograsp_inputs import (
        extract_zerograsp_input,
        save_zerograsp_input_bundle,
    )

    bundle = extract_zerograsp_input(obs, env, args.camera, mask_mode=args.mask_mode)
    out = save_zerograsp_input_bundle(bundle, output_dir)
    scene = {
        "env_id": args.env_id,
        "seed": args.seed,
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "camera_eye": list(args.camera_eye),
        "camera_target": list(args.camera_target),
        "mask_mode": args.mask_mode,
        "settle_before_export_steps": int(args.settle_before_export_steps),
        "n_objects": len(bundle.object_records),
        "objects": bundle.object_records,
    }
    write_json(out / "scene.json", scene)
    return out


def copy_zerograsp_output(source_dir: str | Path, destination_dir: str | Path) -> Path:
    """Copy the ZeroGrasp output used by this run into the run artifact folder."""

    source = Path(source_dir).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"ZeroGrasp output directory not found: {source}")
    if source == destination:
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    copied_any = False
    for name in (
        "recommended_grasp_top1.json",
        "recommended_grasps_topk.json",
        "report.json",
        "run_report.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
            copied_any = True
    raw_source = source / "raw_outputs"
    if raw_source.is_dir():
        shutil.copytree(raw_source, destination / "raw_outputs")
        copied_any = True
    for grasp_path in source.glob("*.grasp.npy"):
        shutil.copy2(grasp_path, destination / grasp_path.name)
        copied_any = True
    if not copied_any:
        raise FileNotFoundError(
            f"{source} does not contain recommended_grasp_top1.json, recommended_grasps_topk.json, report.json, "
            "raw_outputs/, or *.grasp.npy files."
        )
    return destination


def save_grasp_projection(
    input_dir: str | Path,
    zerograsp_output_dir: str | Path,
    output_path: str | Path,
    approach_axis: str,
) -> Path:
    """Save the recommended ZeroGrasp grasp projected onto the saved RGB input."""

    from maniskill_codex.grasp_projection import draw_grasp_projection

    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(zerograsp_output_dir).expanduser().resolve()
    projection_path = Path(output_path).expanduser().resolve()
    return draw_grasp_projection(
        rgb_path=input_root / "rgb.png",
        camera_path=input_root / "camera.json",
        grasp_path=output_root / "recommended_grasp_top1.json",
        output_path=projection_path,
        approach_axis=approach_axis,
    )


def load_control_pose(
    env: Any,
    args: argparse.Namespace,
    zerograsp_output_dir: Path | None,
) -> tuple[ControlPose, ControlPose, dict[str, Any]]:
    if args.target_base is not None:
        position = clamp_base_target(np.asarray(args.target_base, dtype=np.float64), args.workspace_z_min)
        quaternion = unit(np.asarray(args.target_quat_wxyz, dtype=np.float64).reshape(4))
        approach = unit(np.asarray(args.target_approach, dtype=np.float64).reshape(3))
        control_pose = ControlPose(
            position_base=position,
            rotation_base_tool=rotation_matrix_from_quat_wxyz(quaternion),
            quaternion_wxyz=quaternion,
            approach_axis_base=approach,
        )
        return control_pose, control_pose, {
            "source": "target-base-debug",
            "position_base": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "approach_axis_base": approach.tolist(),
        }

    if zerograsp_output_dir is None:
        raise ValueError("Provide --zerograsp-output or --target-base.")

    grasp = load_grasp_candidates(zerograsp_output_dir, top_k=1)[0]
    return control_pose_from_grasp(env, args, grasp, depth_scale=args.grasp_depth_scale)


def control_pose_from_grasp(
    env: Any,
    args: argparse.Namespace,
    grasp: GraspRecord,
    *,
    depth_scale: float,
    log: bool = True,
) -> tuple[ControlPose, ControlPose, dict[str, Any]]:
    camera_model = camera_model_matrix(env, args.camera)
    world_from_base = robot_base_matrix(env)
    zero_depth_control_pose = compute_grasp_control_pose(
        grasp=grasp,
        camera_model_matrix=camera_model,
        world_from_base_matrix=world_from_base,
        approach_axis=args.approach_axis,
        workspace_z_min=args.workspace_z_min,
    )
    hand_tcp_manifest = {"enabled": False}
    if args.use_hand_tcp_calibration:
        hand_to_tcp = hand_tcp_transform_in_hand_frame(env)
        zero_depth_control_pose = apply_hand_tcp_calibration(
            zero_depth_control_pose,
            hand_to_tcp_translation=hand_to_tcp[:3, 3],
            hand_to_tcp_rotation=hand_to_tcp[:3, :3],
            workspace_z_min=args.workspace_z_min,
        )
        hand_tcp_manifest = {
            "enabled": True,
            "hand_frame": "panda_hand",
            "tcp_frame": "panda_hand_tcp",
            "translation_hand_tcp": hand_to_tcp[:3, 3].tolist(),
            "rotation_hand_tcp": hand_to_tcp[:3, :3].tolist(),
        }
    control_pose, depth_offset_manifest = apply_grasp_depth_offset(
        zero_depth_control_pose,
        depth_m=grasp.depth_m,
        scale=depth_scale,
        max_offset_m=args.grasp_depth_max_offset,
        workspace_z_min=args.workspace_z_min,
    )
    manifest = {
        "source": grasp.source,
        "score": grasp.score,
        "width_m": grasp.width_m,
        "height_m": grasp.height_m,
        "depth_m": grasp.depth_m,
        "object_index": grasp.object_index,
        "object_id": grasp.object_id,
        "translation_m_camera": grasp.translation_m_camera.tolist(),
        "rotation_matrix_camera": grasp.rotation_matrix_camera.tolist(),
        "position_base": control_pose.position_base.tolist(),
        "quaternion_wxyz": control_pose.quaternion_wxyz.tolist(),
        "planner_position_base": planner_position_base(control_pose).tolist(),
        "zero_depth_position_base": zero_depth_control_pose.position_base.tolist(),
        "zero_depth_planner_position_base": planner_position_base(
            zero_depth_control_pose
        ).tolist(),
        "planner_quaternion_wxyz": planner_quaternion_wxyz(control_pose).tolist(),
        "approach_axis_base": control_pose.approach_axis_base.tolist(),
        "approach_axis_convention": args.approach_axis,
        "grasp_depth_offset": depth_offset_manifest,
        "hand_tcp_calibration": hand_tcp_manifest,
        "camera_eye": list(args.camera_eye),
        "camera_target": list(args.camera_target),
    }
    if log:
        print(f"grasp_source={grasp.source} score={grasp.score:.6f}")
        print(f"target_grasp_center_base={np.round(control_pose.position_base, 4).tolist()}")
        print(f"target_panda_hand_base={np.round(planner_position_base(control_pose), 4).tolist()}")
        print(f"target_quat_wxyz={np.round(planner_quaternion_wxyz(control_pose), 4).tolist()}")
        print(f"approach_axis_base={np.round(control_pose.approach_axis_base, 4).tolist()}")
        print(
            "grasp_depth_offset "
            f"depth={grasp.depth_m:.4f} scale={depth_scale:.3f} "
            f"requested={depth_offset_manifest['requested_offset_m']:.4f} "
            f"applied={depth_offset_manifest['applied_distance_m']:.4f}"
        )
    return control_pose, zero_depth_control_pose, manifest


def prepare_curobo_scene_model(
    env: Any,
    args: argparse.Namespace,
    output_dir: Path,
    zg_input_dir: Path,
    grasp_manifest: dict[str, Any],
) -> dict[str, Any]:
    if args.scene_source == "fixed":
        return {
            "planner_scene_model": args.scene_model,
            "manifest": {
                "source": "fixed",
                "scene_model": args.scene_model,
            },
        }

    excluded_target_ids: set[int] = set()
    if not args.scene_include_target_object:
        excluded_target_ids = target_segmentation_ids_from_zerograsp_scene(
            zg_input_dir / "scene.json",
            grasp_manifest.get("object_id"),
        )
    export = export_maniskill_scene_to_curobo(
        env,
        output_dir / "curobo_scene.yml",
        metadata_path=output_dir / "curobo_scene_metadata.json",
        world_from_base_matrix=robot_base_matrix(env),
        exclude_segmentation_ids=excluded_target_ids,
        min_cuboid_dimension=args.scene_min_cuboid_dimension,
    )
    print(
        f"curobo_scene={export.scene_path} "
        f"obstacles={len(export.records)} excluded_target_ids={sorted(excluded_target_ids)}"
    )
    return {
        "planner_scene_model": str(export.scene_path),
        "manifest": {
            "source": "maniskill",
            "scene_model": str(export.scene_path),
            "metadata": str(export.metadata_path),
            "n_obstacles": len(export.records),
            "excluded_target_segmentation_ids": sorted(excluded_target_ids),
            "include_target_object": bool(args.scene_include_target_object),
        },
    }


def build_planner(args: argparse.Namespace, scene_model: str | dict[str, Any] | None = None) -> Any:
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    planner_scene_model = scene_model if scene_model is not None else args.scene_model
    config = MotionPlannerCfg.create(robot=args.robot_config, scene_model=planner_scene_model)
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=args.warmup_iterations)
    print(f"curobo_scene_model={planner_scene_model}")
    print(f"curobo_joint_names={planner.joint_names}")
    print(f"curobo_tool_frames={planner.tool_frames}")
    return planner


def planning_result_success(result: Any) -> bool:
    if result is None:
        return False
    success_value = getattr(result, "success", None)
    if success_value is None:
        return False
    try:
        success_value = success_value.any()
    except (AttributeError, TypeError):
        pass
    success_json = sanitize_for_json(success_value)
    try:
        return bool(np.asarray(success_json).any())
    except (TypeError, ValueError):
        return bool(success_json)


def finite_numeric_values(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float64)
    try:
        arr = np.asarray(sanitize_for_json(value), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=np.float64)
    return arr[np.isfinite(arr)]


def bool_values(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=bool)
    try:
        return np.asarray(sanitize_for_json(value), dtype=bool).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=bool)


def build_planning_diagnostic(
    stage_name: str,
    q_start: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    result: Any,
) -> dict[str, Any]:
    success = planning_result_success(result)
    raw_status = "result_none" if result is None else getattr(result, "status", "unknown")
    status = sanitize_for_json(raw_status)
    if not isinstance(status, (str, int, float, bool)) and status is not None:
        status = str(status)

    diagnostic: dict[str, Any] = {
        "stage": stage_name,
        "q_start": np.asarray(q_start, dtype=np.float32).tolist(),
        "target_position": np.asarray(target_position, dtype=np.float32).tolist(),
        "target_quaternion": np.asarray(target_quaternion, dtype=np.float32).tolist(),
        "success": success,
        "status": status,
    }
    if not success:
        diagnostic["failure_reason"] = "curobo_planning_failed"

    if result is None:
        return diagnostic

    for attr in (
        "planning_time",
        "solve_time",
        "position_error",
        "rotation_error",
        "feasible",
        "seed_rank",
        "debug_info",
    ):
        if hasattr(result, attr):
            diagnostic[attr] = sanitize_for_json(getattr(result, attr))

    feasible = bool_values(getattr(result, "feasible", None))
    if feasible.size:
        diagnostic["feasible_count"] = int(feasible.sum())
        diagnostic["feasible_total"] = int(feasible.size)

    seed_cost = getattr(result, "seed_cost", None)
    if seed_cost is not None:
        seed_cost_json = sanitize_for_json(seed_cost)
        diagnostic["seed_cost"] = seed_cost_json
        costs = finite_numeric_values(seed_cost)
        if costs.size:
            diagnostic["seed_cost_min"] = float(costs.min())
            diagnostic["seed_cost_mean"] = float(costs.mean())
            diagnostic["seed_cost_count"] = int(costs.size)

    return diagnostic


def append_planning_diagnostic(path: Path, diagnostic: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"stages": []}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
        elif isinstance(loaded, list):
            data = {"stages": loaded}
    data.setdefault("stages", []).append(sanitize_for_json(diagnostic))
    write_json(path, data)


def plan_stage(
    planner: Any,
    q_start: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    output_dir: Path,
    stage_name: str,
    diagnostics_path: Path | None = None,
    diagnostic_metadata: dict[str, Any] | None = None,
    save_plan: bool = True,
) -> PlannedStage:
    import torch
    from curobo.types import GoalToolPose, JointState

    q_start_tensor = torch.as_tensor(q_start, device="cuda", dtype=torch.float32).unsqueeze(0)
    start_state = JointState.from_position(q_start_tensor, joint_names=planner.joint_names)
    goal_pose = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.as_tensor(
            [[[[np.asarray(target_position, dtype=np.float32).tolist()]]]],
            device="cuda",
            dtype=torch.float32,
        ),
        quaternion=torch.as_tensor(
            [[[[np.asarray(target_quaternion, dtype=np.float32).tolist()]]]],
            device="cuda",
            dtype=torch.float32,
        ),
    )
    result = planner.plan_pose(goal_pose, start_state)
    diagnostic = build_planning_diagnostic(
        stage_name=stage_name,
        q_start=q_start,
        target_position=target_position,
        target_quaternion=target_quaternion,
        result=result,
    )
    if diagnostic_metadata:
        diagnostic.update(sanitize_for_json(diagnostic_metadata))
    if diagnostics_path is not None:
        append_planning_diagnostic(diagnostics_path, diagnostic)
    success = bool(diagnostic["success"])
    if not success:
        status = diagnostic["status"]
        raise RuntimeError(f"cuRobo planning failed for stage={stage_name}: {status}")

    interpolated = result.get_interpolated_plan()
    positions = squeeze_trajectory_positions(interpolated.position.detach().cpu().numpy())
    joint_names = list(interpolated.joint_names)
    output_path = output_dir / f"curobo_plan_{stage_name}.npz"
    if save_plan:
        np.savez(
            output_path,
            stage=stage_name,
            target_position=np.asarray(target_position, dtype=np.float32),
            target_quaternion=np.asarray(target_quaternion, dtype=np.float32),
            planner_joint_names=np.array(planner.joint_names),
            trajectory_joint_names=np.array(joint_names),
            trajectory_positions=positions,
            q_start=np.asarray(q_start, dtype=np.float32),
        )
    print(
        f"stage={stage_name:<5} target={np.round(target_position, 4).tolist()} "
        f"waypoints={positions.shape[0]} plan={output_path if save_plan else '<preflight>'}"
    )
    return PlannedStage(
        name=stage_name,
        target_position=np.asarray(target_position, dtype=np.float64),
        target_quaternion=np.asarray(target_quaternion, dtype=np.float64),
        trajectory_positions=positions,
        trajectory_joint_names=joint_names,
        output_path=output_path,
    )


def select_plannable_grasp_candidate(
    *,
    env: Any,
    args: argparse.Namespace,
    zerograsp_output_dir: Path,
    planner: Any,
    q_start: np.ndarray,
    motion: MotionConfig,
    output_dir: Path,
    diagnostics_path: Path,
) -> tuple[ControlPose, ControlPose, dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Select the highest-scored top-K candidate with plannable pre and grasp stages."""

    top_k = max(1, int(args.candidate_top_k))
    candidates = load_grasp_candidates(zerograsp_output_dir, top_k=top_k)
    if not candidates:
        raise ValueError(f"No grasp candidates found in {zerograsp_output_dir}.")

    attempts: list[dict[str, Any]] = []
    selected: tuple[ControlPose, ControlPose, dict[str, Any], dict[str, dict[str, np.ndarray]]] | None = None
    selected_attempt: dict[str, Any] | None = None
    for rank, grasp in enumerate(candidates):
        depth_scales = (
            grasp_depth_attempt_scales(
                args.grasp_depth_scale,
                args.grasp_depth_fallback_fractions,
            )
            if args.grasp_depth_auto_fallback
            else [float(args.grasp_depth_scale)]
        )
        for depth_scale in depth_scales:
            control_pose, zero_depth_control_pose, grasp_manifest = control_pose_from_grasp(
                env,
                args,
                grasp,
                depth_scale=depth_scale,
                log=False,
            )
            targets = build_stage_targets(
                control_pose,
                motion,
                pregrasp_control_pose=zero_depth_control_pose,
            )
            attempt = {
                "rank": int(rank),
                "score": float(grasp.score),
                "source": grasp.source,
                "object_id": grasp.object_id,
                "depth_m": float(grasp.depth_m),
                "depth_scale": float(depth_scale),
                "grasp_position_base": grasp_manifest["position_base"],
                "planner_position_base": grasp_manifest["planner_position_base"],
                "success": False,
            }
            try:
                pre_plan = plan_stage(
                    planner=planner,
                    q_start=q_start,
                    target_position=targets["pre"]["position"],
                    target_quaternion=targets["pre"]["quaternion"],
                    output_dir=output_dir,
                    stage_name="candidate_pre",
                    diagnostics_path=diagnostics_path,
                    diagnostic_metadata={
                        "candidate_selection": True,
                        "candidate_rank": int(rank),
                        "candidate_score": float(grasp.score),
                        "candidate_depth_scale": float(depth_scale),
                    },
                    save_plan=False,
                )
                q_after_pre = ordered_values(
                    pre_plan.trajectory_joint_names,
                    pre_plan.trajectory_positions[-1],
                    planner.joint_names,
                ).astype(np.float32)
                _grasp_plan = plan_stage(
                    planner=planner,
                    q_start=q_after_pre,
                    target_position=targets["grasp"]["position"],
                    target_quaternion=targets["grasp"]["quaternion"],
                    output_dir=output_dir,
                    stage_name="candidate_grasp",
                    diagnostics_path=diagnostics_path,
                    diagnostic_metadata={
                        "candidate_selection": True,
                        "candidate_rank": int(rank),
                        "candidate_score": float(grasp.score),
                        "candidate_depth_scale": float(depth_scale),
                    },
                    save_plan=False,
                )
            except RuntimeError as exc:
                attempt["failure_reason"] = str(exc)
                attempts.append(attempt)
                continue

            attempt["success"] = True
            attempts.append(attempt)
            selected = (control_pose, zero_depth_control_pose, grasp_manifest, targets)
            selected_attempt = attempt
            break
        if selected is not None:
            break

    if selected is None or selected_attempt is None:
        selection_manifest = {
            "enabled": True,
            "top_k_requested": top_k,
            "n_candidates_loaded": len(candidates),
            "requested_depth_scale": float(args.grasp_depth_scale),
            "selected_rank": None,
            "attempts": attempts,
        }
        append_planning_diagnostic(
            diagnostics_path,
            {
                "stage": "candidate_selection",
                "success": False,
                "failure_reason": "no_plannable_candidate",
                "candidate_selection": selection_manifest,
            },
        )
        raise RuntimeError(
            f"cuRobo candidate selection failed: no plannable candidate in top-{top_k}"
        )

    control_pose, zero_depth_control_pose, grasp_manifest, targets = selected
    grasp_manifest["candidate_selection"] = {
        "enabled": True,
        "top_k_requested": top_k,
        "n_candidates_loaded": len(candidates),
        "requested_depth_scale": float(args.grasp_depth_scale),
        "selected_rank": selected_attempt["rank"],
        "selected_score": selected_attempt["score"],
        "selected_depth_scale": selected_attempt["depth_scale"],
        "attempts": attempts,
    }
    print(
        "candidate_selected "
        f"rank={selected_attempt['rank']} "
        f"score={selected_attempt['score']:.6f} "
        f"depth_scale={selected_attempt['depth_scale']:.3f} "
        f"source={selected_attempt['source']}"
    )
    print(f"target_grasp_center_base={np.round(control_pose.position_base, 4).tolist()}")
    print(f"target_panda_hand_base={np.round(planner_position_base(control_pose), 4).tolist()}")
    print(f"target_quat_wxyz={np.round(planner_quaternion_wxyz(control_pose), 4).tolist()}")
    print(f"approach_axis_base={np.round(control_pose.approach_axis_base, 4).tolist()}")
    return control_pose, zero_depth_control_pose, grasp_manifest, targets


def execute_action_waypoints(
    env: Any,
    actions: np.ndarray,
    recorder: VideoRecorder | None,
    action_repeat: int,
    stop_on_success: bool,
) -> dict[str, Any]:
    final_info: dict[str, Any] = {}
    repeat = max(1, int(action_repeat))
    for action in np.asarray(actions, dtype=np.float32):
        for _ in range(repeat):
            _, _, _, truncated, info = env.step(action[None, :])
            if recorder is not None:
                recorder.capture(env)
            final_info = info
            if as_bool(truncated):
                return final_info
            if stop_on_success and info_success(info):
                return final_info
    return final_info


def object_lift_trace_sample(env: Any, label: str) -> dict[str, Any]:
    target_pose = target_object_pose(env)
    tcp_pose = tcp_pose_matrix(env)
    sample: dict[str, Any] = {"label": label}
    if target_pose is not None:
        sample["object_position_world"] = target_pose[:3, 3].tolist()
        sample["object_quat_wxyz"] = quat_wxyz_from_matrix(target_pose[:3, :3]).tolist()
    if tcp_pose is not None:
        sample["tcp_position_world"] = tcp_pose[:3, 3].tolist()
    if target_pose is not None and tcp_pose is not None:
        sample["object_tcp_distance_m"] = float(np.linalg.norm(target_pose[:3, 3] - tcp_pose[:3, 3]))
    return sample


def compute_object_lift_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [
        np.asarray(sample["object_position_world"], dtype=np.float64).reshape(3)
        for sample in trace
        if "object_position_world" in sample
    ]
    if not positions:
        return {
            "object_lift_success": False,
            "failure_reason": "target_object_pose_missing",
        }

    initial = positions[0]
    final = positions[-1]
    heights = np.asarray([position[2] for position in positions], dtype=np.float64)
    distances = [
        float(sample["object_tcp_distance_m"])
        for sample in trace
        if "object_tcp_distance_m" in sample
    ]
    final_distance = distances[-1] if distances else None
    height_delta = float(final[2] - initial[2])
    max_height_delta = float(heights.max() - initial[2])
    min_required_lift = 0.03
    max_final_tcp_distance = 0.16
    distance_ok = final_distance is None or final_distance <= max_final_tcp_distance
    object_lift_success = bool(height_delta >= min_required_lift and distance_ok)
    return {
        "object_lift_success": object_lift_success,
        "initial_object_position_world": initial.tolist(),
        "final_object_position_world": final.tolist(),
        "height_delta_m": height_delta,
        "max_height_delta_m": max_height_delta,
        "final_object_tcp_distance_m": final_distance,
        "min_required_lift_m": min_required_lift,
        "max_final_tcp_distance_m": max_final_tcp_distance,
        "failure_reason": "" if object_lift_success else "object_not_lifted",
    }


def target_object_pose(env: Any) -> np.ndarray | None:
    root = getattr(env, "unwrapped", env)
    for source_name in ("target_object", "obj", "_objs"):
        for actor in iter_actor_like(getattr(root, source_name, None)):
            pose = getattr(actor, "pose", None)
            if pose is None:
                get_pose = getattr(actor, "get_pose", None)
                pose = get_pose() if callable(get_pose) else None
            if pose is None:
                continue
            try:
                return pose_to_matrix(pose, f"{source_name}.pose")
            except Exception:
                continue
    return None


def tcp_pose_matrix(env: Any) -> np.ndarray | None:
    agent = env.unwrapped.agent
    tcp = getattr(agent, "tcp", None)
    if tcp is not None and getattr(tcp, "pose", None) is not None:
        try:
            return pose_to_matrix(tcp.pose, "agent.tcp.pose")
        except Exception:
            pass
    robot = getattr(agent, "robot", None)
    if robot is None:
        return None
    for link_name in ("panda_hand_tcp", "panda_hand", "tcp"):
        link = find_robot_link(robot, link_name)
        if link is not None and getattr(link, "pose", None) is not None:
            try:
                return pose_to_matrix(link.pose, f"{link_name}.pose")
            except Exception:
                continue
    return None


def iter_actor_like(value: Any):
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_actor_like(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from iter_actor_like(item)
        return
    yield value


def repeat_last_action(actions: np.ndarray, steps: int) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Actions must be a non-empty 2D array, got shape {arr.shape}.")
    if steps <= 0:
        return np.empty((0, arr.shape[1]), dtype=np.float32)
    return np.repeat(arr[-1:, :], int(steps), axis=0)


def build_stage_targets(
    control_pose: ControlPose,
    motion: MotionConfig,
    *,
    pregrasp_control_pose: ControlPose | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    target = clamp_base_target(planner_position_base(control_pose), z_min=motion.workspace_z_min)
    quaternion = planner_quaternion_wxyz(control_pose)
    approach = unit(control_pose.approach_axis_base)
    pregrasp_pose = pregrasp_control_pose or control_pose
    pregrasp_anchor = clamp_base_target(
        planner_position_base(pregrasp_pose),
        z_min=motion.workspace_z_min,
    )
    pre = clamp_base_target(
        pregrasp_anchor - approach * motion.pregrasp_offset_m,
        z_min=motion.workspace_z_min,
    )
    lift = clamp_base_target(target + np.array([0.0, 0.0, motion.lift_offset_m]), z_min=motion.workspace_z_min)
    return {
        "pre": {"position": pre, "quaternion": quaternion},
        "grasp": {"position": target, "quaternion": quaternion},
        "lift": {"position": lift, "quaternion": quaternion},
    }


def planner_position_base(control_pose: ControlPose) -> np.ndarray:
    if control_pose.planner_position_base is not None:
        return np.asarray(control_pose.planner_position_base, dtype=np.float64).reshape(3)
    return np.asarray(control_pose.position_base, dtype=np.float64).reshape(3)


def planner_rotation_base_tool(control_pose: ControlPose) -> np.ndarray:
    if control_pose.planner_rotation_base_tool is not None:
        return np.asarray(control_pose.planner_rotation_base_tool, dtype=np.float64).reshape(3, 3)
    return np.asarray(control_pose.rotation_base_tool, dtype=np.float64).reshape(3, 3)


def planner_quaternion_wxyz(control_pose: ControlPose) -> np.ndarray:
    if control_pose.planner_quaternion_wxyz is not None:
        return unit(np.asarray(control_pose.planner_quaternion_wxyz, dtype=np.float64).reshape(4))
    return unit(np.asarray(control_pose.quaternion_wxyz, dtype=np.float64).reshape(4))


def apply_hand_tcp_calibration(
    control_pose: ControlPose,
    hand_to_tcp_translation: np.ndarray,
    workspace_z_min: float,
    hand_to_tcp_rotation: np.ndarray | None = None,
) -> ControlPose:
    """Convert a desired TCP/grasp-center pose into cuRobo's panda_hand goal pose."""

    hand_tcp = np.eye(4, dtype=np.float64)
    hand_tcp[:3, :3] = (
        np.eye(3, dtype=np.float64)
        if hand_to_tcp_rotation is None
        else orthonormalize_rotation(np.asarray(hand_to_tcp_rotation, dtype=np.float64).reshape(3, 3))
    )
    hand_tcp[:3, 3] = np.asarray(hand_to_tcp_translation, dtype=np.float64).reshape(3)
    base_tcp = np.eye(4, dtype=np.float64)
    base_tcp[:3, :3] = np.asarray(control_pose.rotation_base_tool, dtype=np.float64).reshape(3, 3)
    base_tcp[:3, 3] = np.asarray(control_pose.position_base, dtype=np.float64).reshape(3)
    base_hand = base_tcp @ np.linalg.inv(hand_tcp)
    rotation_base_hand = orthonormalize_rotation(base_hand[:3, :3])
    return ControlPose(
        position_base=control_pose.position_base,
        rotation_base_tool=control_pose.rotation_base_tool,
        quaternion_wxyz=control_pose.quaternion_wxyz,
        approach_axis_base=control_pose.approach_axis_base,
        planner_position_base=clamp_base_target(base_hand[:3, 3], z_min=workspace_z_min),
        planner_rotation_base_tool=rotation_base_hand,
        planner_quaternion_wxyz=quat_wxyz_from_matrix(rotation_base_hand),
    )


def build_control_pose_marker_geometry(
    control_pose: ControlPose,
    world_from_base_matrix: np.ndarray,
    width_m: float,
    approach_length: float = 0.08,
) -> GraspMarkerGeometry:
    """Build the same center/approach/width marker semantics used by maniskill_codex."""

    world_from_base = matrix4(world_from_base_matrix, "world_from_base_matrix")
    rotation_world_tool = world_from_base[:3, :3] @ control_pose.rotation_base_tool
    center_world = base_point_to_world(control_pose.position_base, world_from_base)
    approach_axis_world = unit(world_from_base[:3, :3] @ control_pose.approach_axis_base)
    width_axis_world = unit(rotation_world_tool[:, 1])
    half_width = float(width_m) / 2.0
    return GraspMarkerGeometry(
        center_world=center_world,
        approach_axis_world=approach_axis_world,
        width_axis_world=width_axis_world,
        approach_end_world=center_world + approach_axis_world * float(approach_length),
        width_endpoints_world=(
            center_world + width_axis_world * half_width,
            center_world - width_axis_world * half_width,
        ),
        width_m=float(width_m),
    )


def add_grasp_marker_to_scene(
    scene: Any,
    geometry: GraspMarkerGeometry,
    *,
    name_prefix: str = "zerograsp_marker",
    center_color: list[float] | None = None,
    approach_color: list[float] | None = None,
    width_color: list[float] | None = None,
) -> list[Any]:
    """Draw a center sphere, approach arrow, and gripper-width bar."""

    import sapien

    center_rgba = [0.0, 1.0, 0.0, 0.85] if center_color is None else center_color
    approach_rgba = [1.0, 0.0, 0.0, 0.85] if approach_color is None else approach_color
    width_rgba = [0.0, 0.25, 1.0, 0.85] if width_color is None else width_color
    actors = []
    sphere_builder = scene.create_actor_builder()
    sphere_builder.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        radius=0.018,
        material=sapien.render.RenderMaterial(base_color=center_rgba),
    )
    sphere_builder.set_initial_pose(sapien.Pose(p=geometry.center_world.tolist()))
    actors.append(sphere_builder.build_kinematic(name=f"{name_prefix}_center"))

    actors.append(
        add_marker_bar_actor(
            scene=scene,
            start=geometry.center_world,
            end=geometry.approach_end_world,
            thickness=0.006,
            color=approach_rgba,
            name=f"{name_prefix}_approach",
        )
    )
    actors.append(
        add_marker_bar_actor(
            scene=scene,
            start=geometry.width_endpoints_world[0],
            end=geometry.width_endpoints_world[1],
            thickness=0.005,
            color=width_rgba,
            name=f"{name_prefix}_width",
        )
    )

    tip_builder = scene.create_actor_builder()
    tip_builder.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        radius=0.012,
        material=sapien.render.RenderMaterial(base_color=approach_rgba),
    )
    tip_builder.set_initial_pose(sapien.Pose(p=geometry.approach_end_world.tolist()))
    actors.append(tip_builder.build_kinematic(name=f"{name_prefix}_approach_tip"))
    return actors


def add_marker_bar_actor(
    scene: Any,
    start: np.ndarray,
    end: np.ndarray,
    thickness: float,
    color: list[float],
    name: str,
) -> Any:
    import sapien

    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        length = 1e-6
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    center = (start + end) / 2.0
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        pose=sapien.Pose(),
        half_size=[length / 2.0, thickness, thickness],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    builder.set_initial_pose(
        sapien.Pose(
            p=center.tolist(),
            q=quat_from_x_axis(direction).tolist(),
        )
    )
    return builder.build_kinematic(name=name)


def marker_manifest(geometry: GraspMarkerGeometry, actor_count: int) -> dict[str, Any]:
    return {
        "enabled": True,
        "actor_count": int(actor_count),
        "center_world": geometry.center_world.tolist(),
        "approach_axis_world": geometry.approach_axis_world.tolist(),
        "width_axis_world": geometry.width_axis_world.tolist(),
        "approach_end_world": geometry.approach_end_world.tolist(),
        "width_endpoints_world": [
            geometry.width_endpoints_world[0].tolist(),
            geometry.width_endpoints_world[1].tolist(),
        ],
        "width_m": geometry.width_m,
    }


def load_best_grasp(output_dir: str | Path) -> GraspRecord:
    return load_grasp_candidates(output_dir, top_k=1)[0]


def load_grasp_candidates(output_dir: str | Path, top_k: int | None = None) -> list[GraspRecord]:
    root = Path(output_dir).expanduser().resolve()
    limit = None if top_k is None else max(1, int(top_k))
    topk_path = root / "recommended_grasps_topk.json"
    if topk_path.is_file():
        payload = json.loads(topk_path.read_text(encoding="utf-8"))
        items = payload.get("grasps", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(f"{topk_path} must contain a list or a dict with a 'grasps' list.")
        records = [
            grasp_record_from_json(item, source=f"{topk_path}#{index}")
            for index, item in enumerate(items)
        ]
        records.sort(key=lambda record: record.score, reverse=True)
        if limit is not None:
            records = records[:limit]
        if records:
            return records

    grasp_files = sorted((root / "raw_outputs").glob("*.grasp.npy")) or sorted(root.glob("*.grasp.npy"))
    if limit is None or limit > 1:
        raw_records = load_raw_grasp_candidates(grasp_files)
        if raw_records:
            return raw_records[:limit] if limit is not None else raw_records

    json_path = root / "recommended_grasp_top1.json"
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return [grasp_record_from_json(data, source=str(json_path))]

    if not grasp_files:
        raise FileNotFoundError(
            f"No ZeroGrasp grasp output found under {root}. Expected recommended_grasp_top1.json "
            "or raw_outputs/*.grasp.npy."
        )
    raw_records = load_raw_grasp_candidates(grasp_files)
    if not raw_records:
        raise ValueError(f"All grasp arrays under {root} are empty.")
    return raw_records[:limit] if limit is not None else raw_records


def load_raw_grasp_candidates(grasp_files: list[Path]) -> list[GraspRecord]:
    records: list[GraspRecord] = []
    for path in grasp_files:
        arr = np.asarray(np.load(path), dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for row_index, row in enumerate(arr):
            records.append(
                GraspRecord(
                    score=float(row[0]),
                    width_m=float(row[1]),
                    height_m=float(row[2]),
                    depth_m=float(row[3]),
                    rotation_matrix_camera=row[4:13].reshape(3, 3),
                    translation_m_camera=row[13:16].reshape(3),
                    source=f"{path}#{row_index}",
                    object_index=None,
                    object_id=optional_int(row[16]) if row.shape[0] > 16 else None,
                )
            )
    records.sort(key=lambda record: record.score, reverse=True)
    return records


def grasp_record_from_json(data: dict[str, Any], *, source: str) -> GraspRecord:
    return GraspRecord(
        score=float(required(data, "score", source)),
        width_m=float(required(data, "width_m", source)),
        height_m=float(required(data, "height_m", source)),
        depth_m=float(required(data, "depth_m", source)),
        rotation_matrix_camera=np.asarray(
            required(data, "rotation_matrix_camera", source),
            dtype=np.float64,
        ).reshape(3, 3),
        translation_m_camera=np.asarray(
            required(data, "translation_m_camera", source),
            dtype=np.float64,
        ).reshape(3),
        source=str(data.get("source_file") or source),
        object_index=optional_int(data.get("object_index")),
        object_id=optional_int(data.get("object_id")),
    )


def compute_grasp_control_pose(
    grasp: GraspRecord,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    approach_axis: str,
    workspace_z_min: float,
) -> ControlPose:
    position_base = clamp_base_target(
        opencv_camera_to_base(grasp.translation_m_camera, camera_model_matrix, world_from_base_matrix),
        z_min=workspace_z_min,
    )
    rotation_world_grasp = opencv_grasp_rotation_to_world_axes(
        grasp.rotation_matrix_camera,
        camera_model_matrix,
    )
    base_from_world = np.linalg.inv(matrix4(world_from_base_matrix, "world_from_base_matrix"))
    rotation_base_grasp = orthonormalize_rotation(base_from_world[:3, :3] @ rotation_world_grasp)
    if approach_axis == "flip-world-z":
        rotation_base_tool = tcp_rotation_from_approach_and_width(
            zerograsp_approach_vector(rotation_base_grasp, approach_axis),
            zerograsp_width_vector(rotation_base_grasp),
        )
    else:
        rotation_base_tool = orthonormalize_rotation(
            rotation_base_grasp @ panda_tcp_axes_in_zerograsp_frame(approach_axis)
        )
    quaternion = quat_wxyz_from_matrix(rotation_base_tool)
    return ControlPose(
        position_base=position_base,
        rotation_base_tool=rotation_base_tool,
        quaternion_wxyz=quaternion,
        approach_axis_base=unit(rotation_base_tool[:, 2]),
    )


def apply_grasp_depth_offset(
    control_pose: ControlPose,
    *,
    depth_m: float,
    scale: float,
    max_offset_m: float,
    workspace_z_min: float,
) -> tuple[ControlPose, dict[str, Any]]:
    """Move the TCP target into the grasp along the predicted approach depth."""

    depth = max(0.0, float(depth_m))
    depth_scale = float(scale)
    max_offset = float(max_offset_m)
    if depth_scale < 0.0:
        raise ValueError("--grasp-depth-scale must be non-negative.")
    if max_offset < 0.0:
        raise ValueError("--grasp-depth-max-offset must be non-negative.")

    original_position = np.asarray(control_pose.position_base, dtype=np.float64).reshape(3)
    requested_offset = min(depth * depth_scale, max_offset)
    requested_vector = unit(control_pose.approach_axis_base) * requested_offset
    adjusted_position = clamp_base_target(
        original_position + requested_vector,
        z_min=workspace_z_min,
    )
    applied_vector = adjusted_position - original_position

    planner_position = control_pose.planner_position_base
    if planner_position is not None:
        planner_position = np.asarray(planner_position, dtype=np.float64).reshape(3) + applied_vector

    adjusted_pose = ControlPose(
        position_base=adjusted_position,
        rotation_base_tool=control_pose.rotation_base_tool,
        quaternion_wxyz=control_pose.quaternion_wxyz,
        approach_axis_base=control_pose.approach_axis_base,
        planner_position_base=planner_position,
        planner_rotation_base_tool=control_pose.planner_rotation_base_tool,
        planner_quaternion_wxyz=control_pose.planner_quaternion_wxyz,
    )
    manifest = {
        "enabled": bool(depth_scale > 0.0 and requested_offset > 0.0),
        "depth_m": depth,
        "scale": depth_scale,
        "max_offset_m": max_offset,
        "position_base_before": original_position.tolist(),
        "requested_offset_m": requested_offset,
        "requested_vector_base": requested_vector.tolist(),
        "position_base_after": adjusted_position.tolist(),
        "applied_vector_base": applied_vector.tolist(),
        "applied_distance_m": float(np.linalg.norm(applied_vector)),
        "workspace_clamped": not np.allclose(applied_vector, requested_vector, atol=1e-9),
    }
    return adjusted_pose, manifest


def grasp_depth_attempt_scales(
    requested_scale: float,
    fallback_fractions: Iterable[float],
) -> list[float]:
    requested = float(requested_scale)
    if requested < 0.0:
        raise ValueError("--grasp-depth-scale must be non-negative.")
    scales = [requested]
    for fraction in fallback_fractions:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "--grasp-depth-fallback-fractions values must be within [0, 1]."
            )
        candidate = requested * value
        if not any(abs(candidate - existing) < 1e-9 for existing in scales):
            scales.append(candidate)
    return scales


def update_grasp_manifest_pose(
    manifest: dict[str, Any],
    control_pose: ControlPose,
    depth_offset_manifest: dict[str, Any],
) -> None:
    manifest["position_base"] = control_pose.position_base.tolist()
    manifest["quaternion_wxyz"] = control_pose.quaternion_wxyz.tolist()
    manifest["planner_position_base"] = planner_position_base(control_pose).tolist()
    manifest["planner_quaternion_wxyz"] = planner_quaternion_wxyz(control_pose).tolist()
    manifest["approach_axis_base"] = control_pose.approach_axis_base.tolist()
    manifest["grasp_depth_offset"] = depth_offset_manifest


def opencv_camera_to_base(
    position_m: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
) -> np.ndarray:
    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    camera_from_opencv = OPENCV_TO_SAPIEN_CAMERA @ position
    world_from_camera = matrix4(camera_model_matrix, "camera_model_matrix")
    world = (world_from_camera @ np.array([*camera_from_opencv, 1.0], dtype=np.float64))[:3]
    base_from_world = np.linalg.inv(matrix4(world_from_base_matrix, "world_from_base_matrix"))
    return (base_from_world @ np.array([*world, 1.0], dtype=np.float64))[:3]


def opencv_grasp_rotation_to_world_axes(
    rotation_matrix_camera: np.ndarray,
    camera_model_matrix: np.ndarray,
) -> np.ndarray:
    rotation_cv = np.asarray(rotation_matrix_camera, dtype=np.float64).reshape(3, 3)
    world_from_camera = matrix4(camera_model_matrix, "camera_model_matrix")
    rotation_sapien = OPENCV_TO_SAPIEN_CAMERA @ rotation_cv
    return normalize_columns(world_from_camera[:3, :3] @ rotation_sapien)


def zerograsp_approach_vector(rotation_matrix: np.ndarray, approach_axis: str) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    if approach_axis == "flip-world-z":
        approach = -rotation[:, 0].copy()
        approach[2] *= -1.0
        return approach
    sign = 1.0 if approach_axis == "positive-x" else -1.0
    return sign * rotation[:, 0]


def zerograsp_width_vector(rotation_matrix: np.ndarray) -> np.ndarray:
    return -np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)[:, 1]


def panda_tcp_axes_in_zerograsp_frame(approach_axis: str) -> np.ndarray:
    if approach_axis == "positive-x":
        return np.array(
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    return np.array(
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )


def tcp_rotation_from_approach_and_width(approach_axis_base: np.ndarray, width_axis_base: np.ndarray) -> np.ndarray:
    tool_z = unit(approach_axis_base)
    width = unit(width_axis_base)
    tool_y = width - tool_z * float(np.dot(width, tool_z))
    if np.linalg.norm(tool_y) < 1e-6:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper, tool_z))) > 0.95:
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tool_y = helper - tool_z * float(np.dot(helper, tool_z))
    tool_y = unit(tool_y)
    tool_x = unit(np.cross(tool_y, tool_z))
    tool_y = unit(np.cross(tool_z, tool_x))
    return np.stack([tool_x, tool_y, tool_z], axis=1)


def clamp_base_target(target: np.ndarray, z_min: float | None = None) -> np.ndarray:
    bounds = DEFAULT_WORKSPACE_BOUNDS
    if z_min is not None:
        bounds = (bounds[0], bounds[1], (float(z_min), bounds[2][1]))
    point = np.asarray(target, dtype=np.float64).reshape(3)
    return np.array(
        [
            np.clip(point[0], bounds[0][0], bounds[0][1]),
            np.clip(point[1], bounds[1][0], bounds[1][1]),
            np.clip(point[2], bounds[2][0], bounds[2][1]),
        ],
        dtype=np.float64,
    )


def camera_model_matrix(env: Any, camera_name: str) -> np.ndarray:
    sensors = env.unwrapped.scene.sensors
    if camera_name not in sensors:
        raise KeyError(f"Camera {camera_name!r} not found. Available: {sorted(sensors)}")
    return first_matrix(sensors[camera_name].camera.get_model_matrix(), "camera model matrix")


def robot_base_matrix(env: Any) -> np.ndarray:
    return first_matrix(env.unwrapped.agent.robot.get_pose().to_transformation_matrix(), "robot base matrix")


def hand_tcp_transform_in_hand_frame(env: Any) -> np.ndarray:
    """Return T_panda_hand_panda_hand_tcp from the active ManiSkill Panda model."""

    root = getattr(env, "unwrapped", env)
    agent = getattr(root, "agent", None)
    robot = getattr(agent, "robot", None)
    if robot is None:
        raise ValueError("Cannot find ManiSkill robot for TCP calibration.")
    hand = find_robot_link(robot, "panda_hand")
    tcp = getattr(agent, "tcp", None)
    if tcp is None:
        tcp = find_robot_link(robot, "panda_hand_tcp")
    if hand is None or tcp is None:
        raise ValueError("Cannot find panda_hand and panda_hand_tcp links for TCP calibration.")
    world_from_hand = pose_to_matrix(getattr(hand, "pose", None), "panda_hand pose")
    world_from_tcp = pose_to_matrix(getattr(tcp, "pose", None), "panda_hand_tcp pose")
    return np.linalg.inv(world_from_hand) @ world_from_tcp


def find_robot_link(robot: Any, name: str) -> Any | None:
    get_links = getattr(robot, "get_links", None)
    if callable(get_links):
        for link in get_links():
            if getattr(link, "name", None) == name:
                return link
    links = getattr(robot, "links", None)
    if isinstance(links, dict):
        return links.get(name)
    if links is not None:
        for link in links:
            if getattr(link, "name", None) == name:
                return link
    return None


def pose_to_matrix(pose: Any, name: str) -> np.ndarray:
    if pose is None:
        raise ValueError(f"{name} is missing.")
    to_matrix = getattr(pose, "to_transformation_matrix", None)
    if callable(to_matrix):
        return first_matrix(to_matrix(), name)
    raw_pose = getattr(pose, "raw_pose", None)
    if raw_pose is not None:
        raw = first_vector(raw_pose, f"{name}.raw_pose")
        position = raw[:3]
        quaternion = raw[3:7]
    else:
        position = first_vector(getattr(pose, "p", None), f"{name}.p")[:3]
        quaternion = first_vector(getattr(pose, "q", None), f"{name}.q")[:4]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix_from_quat_wxyz(quaternion)
    matrix[:3, 3] = position
    return matrix


def base_point_to_world(position_base: np.ndarray, world_from_base_matrix: np.ndarray) -> np.ndarray:
    point = np.asarray(position_base, dtype=np.float64).reshape(3)
    world_from_base = matrix4(world_from_base_matrix, "world_from_base_matrix")
    return (world_from_base @ np.array([*point, 1.0], dtype=np.float64))[:3]


def stage_manifest(stage: PlannedStage) -> dict[str, Any]:
    return {
        "name": stage.name,
        "target_position": stage.target_position.tolist(),
        "target_quaternion": stage.target_quaternion.tolist(),
        "trajectory_shape": list(stage.trajectory_positions.shape),
        "trajectory_joint_names": stage.trajectory_joint_names,
        "output_path": str(stage.output_path),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize_for_json(data), indent=2, ensure_ascii=False), encoding="utf-8")


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return sanitize_for_json(value.detach().cpu().numpy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def normalize_rgb_frame(frame: Any) -> np.ndarray:
    arr = to_numpy(frame)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Rendered frame must have shape (H, W, 3/4), got {arr.shape}.")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def required(data: dict[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise ValueError(f"{path} is missing required field {key!r}.")
    return data[key]


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    value_int = int(value)
    return value_int if value_int >= 0 else None


def first_matrix(value: Any, name: str) -> np.ndarray:
    arr = to_numpy(value)
    if arr.shape == (4, 4):
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[0] >= 1 and arr.shape[1:] == (4, 4):
        return arr[0].astype(np.float64)
    raise ValueError(f"{name} must have shape (4,4) or (N,4,4), got {arr.shape}.")


def first_vector(value: Any, name: str) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim == 1:
        return arr.astype(np.float64)
    if arr.ndim == 2 and arr.shape[0] >= 1:
        return arr[0].astype(np.float64)
    raise ValueError(f"{name} must have shape (D,) or (N,D), got {arr.shape}.")


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def matrix4(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {arr.shape}.")
    return arr


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64).reshape(3, 3).copy()
    for index in range(3):
        arr[:, index] = unit(arr[:, index])
    return arr


def orthonormalize_rotation(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    u, _, vh = np.linalg.svd(rotation)
    out = u @ vh
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vh
    return out


def quat_wxyz_from_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = (1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) ** 0.5 * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = (1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) ** 0.5 * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = (1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) ** 0.5 * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    return unit(quat)


def rotation_matrix_from_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = unit(np.asarray(quaternion, dtype=np.float64).reshape(4))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_from_x_axis(direction: np.ndarray) -> np.ndarray:
    x_axis = unit(direction)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x_axis, helper))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = unit(np.cross(helper, x_axis))
    z_axis = unit(np.cross(x_axis, y_axis))
    return quat_wxyz_from_matrix(np.stack([x_axis, y_axis, z_axis], axis=1))


def unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return arr / norm


def info_success(info: dict[str, Any]) -> bool:
    return isinstance(info, dict) and "success" in info and as_bool(info["success"])


def as_bool(value: Any) -> bool:
    arr = to_numpy(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(np.any(arr))


if __name__ == "__main__":
    raise SystemExit(main())
