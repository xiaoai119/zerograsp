"""Execute an offline ZeroGrasp grasp inside a ManiSkill scene."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from maniskill_codex.camera_views import add_camera_view_args
from maniskill_codex.grasp_axes import (
    APPROACH_AXIS_CHOICES,
    panda_tcp_axes_in_zerograsp_frame,
    zerograsp_approach_vector,
    zerograsp_width_vector,
)
from maniskill_codex.grasp_markers import (
    add_grasp_marker_to_scene,
    build_grasp_marker_geometry,
    opencv_grasp_rotation_to_world_axes,
)
from maniskill_codex.grasp_rl_tuning import (
    RLRolloutEvaluation,
    RLTuningConfig,
    ResidualActionBounds,
    ResidualGraspAction,
    apply_residual_to_pose,
    reward_from_rollout_result,
    tune_residual_policy,
)
from maniskill_codex.transforms import opencv_camera_to_base
from maniskill_codex.zerograsp_inputs import (
    MASK_MODES,
    extract_zerograsp_input,
    save_zerograsp_input_bundle,
)
from maniskill_codex.zerograsp_outputs import GraspRecord, load_best_grasp


DEFAULT_WORKSPACE_Z_MIN = 0.02
DEFAULT_WORKSPACE_BOUNDS = (
    (0.25, 0.85),
    (-0.45, 0.45),
    (DEFAULT_WORKSPACE_Z_MIN, 0.60),
)

@dataclass(frozen=True)
class MotionConfig:
    """Parameters for a simple top-down pick attempt."""

    pregrasp_offset_m: float = 0.10
    lift_offset_m: float = 0.15
    stage_steps: int = 20
    pregrasp_max_steps: int = 200
    max_stage_steps: int = 80
    settle_pos_tolerance_m: float = 0.01
    descend_settle_pos_tolerance_m: float = 0.02
    workspace_z_min: float = DEFAULT_WORKSPACE_Z_MIN
    gripper_open: float = 1.0
    gripper_closed: float = -1.0


@dataclass(frozen=True)
class GraspControlPose:
    """A ZeroGrasp pose converted to the Panda controller frame."""

    position_base: np.ndarray
    rotation_base_tcp: np.ndarray
    euler_xyz_base: np.ndarray
    approach_axis_base: np.ndarray


class VideoRecorder:
    """Collect RGB frames from an environment and write them as an MP4."""

    def __init__(self, output_path: str | Path | None, fps: int = 20):
        self.output_path = Path(output_path).expanduser().resolve() if output_path else None
        self.fps = int(fps)
        self.frames: list[np.ndarray] = []

    @property
    def enabled(self) -> bool:
        return self.output_path is not None

    def capture(self, env: Any) -> None:
        if not self.enabled:
            return
        self.frames.append(_normalize_rgb_frame(env.render()))

    def save(self) -> Path | None:
        if not self.enabled:
            return None
        if not self.frames:
            raise RuntimeError(f"No video frames captured for {self.output_path}.")

        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio

        iio.imwrite(self.output_path, np.stack(self.frames, axis=0), fps=self.fps)
        return self.output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a ZeroGrasp offline grasp output in ManiSkill PickSingleYCB-v1."
    )
    parser.add_argument(
        "--zerograsp-output",
        required=True,
        help="Directory containing recommended_grasp_top1.json or raw_outputs/*.grasp.npy.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of reset seeds to run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--env-id", default="PickSingleYCB-v1", help="ManiSkill environment id.")
    parser.add_argument("--camera", default="base_camera", help="ManiSkill sensor name.")
    parser.add_argument("--width", type=int, default=1280, help="Sensor image width.")
    parser.add_argument("--height", type=int, default=1024, help="Sensor image height.")
    add_camera_view_args(parser)
    parser.add_argument("--render-width", type=int, default=1280, help="MP4 render image width.")
    parser.add_argument("--render-height", type=int, default=1024, help="MP4 render image height.")
    parser.add_argument("--stage-steps", type=int, default=20, help="Simulation steps per stage.")
    parser.add_argument("--pregrasp-max-steps", type=int, default=200, help="Maximum simulation steps for pre-grasp settling.")
    parser.add_argument("--stage-max-steps", type=int, default=80, help="Maximum simulation steps per motion stage.")
    parser.add_argument("--settle-pos-tolerance", type=float, default=0.01, help="TCP position tolerance before advancing stages.")
    parser.add_argument("--descend-settle-pos-tolerance", type=float, default=0.02, help="TCP position tolerance before closing after descend.")
    parser.add_argument("--workspace-z-min", type=float, default=DEFAULT_WORKSPACE_Z_MIN, help="Minimum base-frame z target.")
    parser.add_argument("--pregrasp-offset", type=float, default=0.10, help="Meters above grasp target.")
    parser.add_argument("--lift-offset", type=float, default=0.15, help="Meters above grasp target after close.")
    parser.add_argument("--video-out", default=None, help="Optional MP4 path to record the execution.")
    parser.add_argument("--video-fps", type=int, default=20, help="Frames per second for --video-out.")
    parser.add_argument(
        "--save-zg-input-dir",
        default=None,
        help="Optional directory for saving ZeroGrasp RGB-D, mask, and camera inputs.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=MASK_MODES,
        default="task-target",
        help=(
            "Which ManiSkill segmentation ids to save for ZeroGrasp inputs when "
            "--save-zg-input-dir is used."
        ),
    )
    parser.add_argument(
        "--show-grasp-marker",
        action="store_true",
        help="Add visual-only sphere/arrow marker for the ZeroGrasp output in the scene.",
    )
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Ignore ZeroGrasp rotation and keep the current TCP orientation.",
    )
    parser.add_argument(
        "--approach-axis",
        choices=APPROACH_AXIS_CHOICES,
        default="negative-x",
        help=(
            "Which ZeroGrasp local X direction should be treated as the approach direction. "
            "negative-x preserves the previous convention; positive-x flips the full X axis; "
            "flip-world-z keeps XY and flips only the world/base Z component."
        ),
    )
    parser.add_argument(
        "--rl-tune",
        action="store_true",
        help=(
            "Tune a small residual grasp policy before the final execution. "
            "This searches pose offsets and gripper open/close commands with ManiSkill rollouts."
        ),
    )
    parser.add_argument("--rl-iters", type=int, default=3, help="CEM iterations for --rl-tune.")
    parser.add_argument("--rl-population", type=int, default=8, help="Rollouts per CEM iteration for --rl-tune.")
    parser.add_argument("--rl-elite-fraction", type=float, default=0.25, help="Elite fraction used by CEM.")
    parser.add_argument("--rl-seed", type=int, default=None, help="Random seed for residual policy search.")
    parser.add_argument(
        "--rl-output",
        default=None,
        help="Optional JSON path for residual policy tuning traces.",
    )
    parser.add_argument(
        "--rl-stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop residual policy search after the first successful rollout.",
    )
    parser.add_argument(
        "--rl-approach-offset-range",
        type=float,
        nargs=2,
        default=(0.0, 0.05),
        metavar=("MIN", "MAX"),
        help="Meters to search along the grasp approach direction; positive drives deeper into the grasp.",
    )
    parser.add_argument(
        "--rl-tcp-x-offset-range",
        type=float,
        nargs=2,
        default=(-0.02, 0.02),
        metavar=("MIN", "MAX"),
        help="Meters to search along the TCP local x axis.",
    )
    parser.add_argument(
        "--rl-tcp-y-offset-range",
        type=float,
        nargs=2,
        default=(-0.02, 0.02),
        metavar=("MIN", "MAX"),
        help="Meters to search along the TCP local y/gripper-width axis.",
    )
    parser.add_argument(
        "--rl-roll-delta-range",
        type=float,
        nargs=2,
        default=(-0.35, 0.35),
        metavar=("MIN", "MAX"),
        help="Radians to search around the TCP local approach axis.",
    )
    parser.add_argument(
        "--rl-gripper-open-range",
        type=float,
        nargs=2,
        default=(0.5, 1.0),
        metavar=("MIN", "MAX"),
        help="Normalized ManiSkill gripper command range for pre/descend open stages.",
    )
    parser.add_argument(
        "--rl-gripper-closed-range",
        type=float,
        nargs=2,
        default=(-1.0, -0.2),
        metavar=("MIN", "MAX"),
        help="Normalized ManiSkill gripper command range for close/lift stages.",
    )
    return parser.parse_args(argv)


def clamp_base_target(
    target: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
    z_min: float | None = None,
) -> np.ndarray:
    """Clamp a base-frame target into a conservative Panda workspace."""

    point = np.asarray(target, dtype=np.float64).reshape(3)
    if z_min is not None:
        bounds = (bounds[0], bounds[1], (float(z_min), bounds[2][1]))
    return np.array(
        [
            np.clip(point[0], bounds[0][0], bounds[0][1]),
            np.clip(point[1], bounds[1][0], bounds[1][1]),
            np.clip(point[2], bounds[2][0], bounds[2][1]),
        ],
        dtype=np.float64,
    )


def build_env(
    width: int,
    height: int,
    render_width: int = 1280,
    render_height: int = 1024,
    env_id: str = "PickSingleYCB-v1",
    camera_name: str = "base_camera",
    camera_eye: Iterable[float] | None = None,
    camera_target: Iterable[float] | None = None,
    control_mode: str = "pd_ee_pose",
):
    """Create the ManiSkill environment. Imports stay local for lightweight tests."""

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ManiSkill is not importable. Run this command inside the maniskill conda env, "
            "for example: source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
            "conda activate maniskill"
        ) from exc

    sensor_configs: dict[str, Any] = {"width": width, "height": height}
    if camera_eye is not None or camera_target is not None:
        if camera_eye is None or camera_target is None:
            raise ValueError("camera_eye and camera_target must be provided together.")
        from mani_skill.utils import sapien_utils

        sensor_configs[camera_name] = {
            "pose": sapien_utils.look_at(list(camera_eye), list(camera_target))
        }

    return gym.make(
        env_id,
        render_mode="rgb_array",
        control_mode=control_mode,
        robot_uids="panda",
        obs_mode="sensor_data",
        max_episode_steps=300,
        sensor_configs=sensor_configs,
        human_render_camera_configs={"width": render_width, "height": render_height},
    )


def grasp_to_base_target(env: Any, grasp: GraspRecord, camera_name: str) -> np.ndarray:
    """Convert a ZeroGrasp camera-frame grasp center to robot base frame."""

    return grasp_to_base_control_pose(env, grasp, camera_name).position_base


def grasp_to_base_control_pose(env: Any, grasp: GraspRecord, camera_name: str) -> GraspControlPose:
    """Convert a ZeroGrasp 6D grasp to a Panda TCP target pose in base frame."""

    return compute_grasp_control_pose(
        grasp,
        _camera_model_matrix(env, camera_name),
        _robot_base_matrix(env),
    )


def compute_grasp_control_pose(
    grasp: GraspRecord,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    approach_axis: str = "negative-x",
) -> GraspControlPose:
    """Convert a ZeroGrasp camera-frame pose to a base-frame Panda TCP pose."""

    position_base = opencv_camera_to_base(
        grasp.translation_m_camera,
        camera_model_matrix,
        world_from_base_matrix,
    )
    rotation_world_grasp = opencv_grasp_rotation_to_world_axes(
        grasp.rotation_matrix_camera,
        camera_model_matrix,
    )
    base_from_world = np.linalg.inv(_first_matrix(world_from_base_matrix, "world_from_base_matrix"))
    rotation_base_grasp = _orthonormalize_rotation(base_from_world[:3, :3] @ rotation_world_grasp)
    if approach_axis == "flip-world-z":
        rotation_base_tcp = _tcp_rotation_from_approach_and_width(
            zerograsp_approach_vector(rotation_base_grasp, approach_axis),
            zerograsp_width_vector(rotation_base_grasp),
        )
    else:
        rotation_base_tcp = _orthonormalize_rotation(
            rotation_base_grasp @ panda_tcp_axes_in_zerograsp_frame(approach_axis)
        )
    return GraspControlPose(
        position_base=position_base,
        rotation_base_tcp=rotation_base_tcp,
        euler_xyz_base=matrix_to_euler_xyz(rotation_base_tcp),
        approach_axis_base=_unit(rotation_base_tcp[:, 2]),
    )


def matrix_to_euler_xyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to ManiSkill's XYZ Euler convention."""

    R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    y = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    cy = float(np.cos(y))
    if abs(cy) > 1e-8:
        x = np.arctan2(-R[1, 2], R[2, 2])
        z = np.arctan2(-R[0, 1], R[0, 0])
    else:
        x = np.arctan2(R[2, 1], R[1, 1])
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)


def build_stage_targets(
    target_base: np.ndarray,
    approach_axis_base: np.ndarray,
    motion: MotionConfig,
) -> dict[str, np.ndarray]:
    """Build pre-grasp/grasp/lift targets using the grasp approach direction."""

    target = clamp_base_target(target_base, z_min=motion.workspace_z_min)
    approach = _unit(approach_axis_base)
    pre = clamp_base_target(target - approach * motion.pregrasp_offset_m, z_min=motion.workspace_z_min)
    lift = clamp_base_target(
        target + np.array([0.0, 0.0, motion.lift_offset_m]),
        z_min=motion.workspace_z_min,
    )
    return {"pre": pre, "grasp": target, "lift": lift}


def execute_pick(
    env: Any,
    target_base: np.ndarray,
    motion: MotionConfig,
    recorder: VideoRecorder | None = None,
    target_euler_xyz: np.ndarray | None = None,
    approach_axis_base: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute pre-grasp, descend, close, and lift stages."""

    if approach_axis_base is None:
        approach_axis_base = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    targets = build_stage_targets(target_base, approach_axis_base, motion)
    target_euler = (
        _tcp_base_euler_xyz(env)
        if target_euler_xyz is None
        else _first_vector3(target_euler_xyz, "target_euler_xyz")
    )
    _log(verbose, f"  target_tcp_euler_xyz={np.round(target_euler, 4)}")
    stages = [
        ("pre", targets["pre"], motion.gripper_open, True),
        ("descend", targets["grasp"], motion.gripper_open, True),
        ("close", targets["grasp"], motion.gripper_closed, False),
        ("lift", targets["lift"], motion.gripper_closed, True),
    ]

    final_info: dict[str, Any] = {}
    for name, stage_target, gripper, wait_for_settle in stages:
        action = make_action(stage_target, target_euler, gripper)
        _log(verbose, f"  stage={name:<7} target_base={np.round(stage_target, 4)} gripper={gripper:.3f}")
        settle_tolerance = (
            motion.descend_settle_pos_tolerance_m
            if name == "descend"
            else motion.settle_pos_tolerance_m
        )
        max_steps = (
            motion.pregrasp_max_steps
            if name == "pre"
            else motion.max_stage_steps
            if wait_for_settle
            else motion.stage_steps
        )
        reached = False
        tcp_base = None
        err = None
        for step_index in range(max_steps):
            _, _, _, truncated, info = env.step(action)
            if recorder is not None:
                recorder.capture(env)
            final_info = info
            tcp_base = _tcp_base_position(env)
            if tcp_base is not None:
                err = float(np.linalg.norm(stage_target - tcp_base))
                reached = err <= settle_tolerance
            if _info_success(info):
                _log(verbose, f"    success signaled during {name}")
                return {"success": True, "stage": name, "info": info}
            if _as_bool(truncated):
                _log(verbose, f"    truncated during {name}: {info}")
                return {"success": False, "stage": name, "truncated": True, "info": info}
            if wait_for_settle and tcp_base is None and step_index + 1 >= motion.stage_steps:
                break
            if wait_for_settle and step_index + 1 >= motion.stage_steps and reached:
                break

        if tcp_base is not None:
            _log(verbose, f"    tcp_base={np.round(tcp_base, 4)} err={err:.4f} steps={step_index + 1}")
        if wait_for_settle and not reached and name in {"pre", "descend"}:
            _log(
                verbose,
                f"    {name} did not converge within {max_steps} steps "
                f"(tol={settle_tolerance:.4f})",
            )
            return {
                "success": False,
                "stage": name,
                "not_converged": True,
                "info": final_info,
            }

    return {"success": _info_success(final_info), "stage": "lift", "info": final_info}


def run_episode(
    env: Any,
    grasp: GraspRecord,
    args: argparse.Namespace,
    episode_index: int,
    recorder: VideoRecorder | None = None,
) -> dict[str, Any]:
    seed = args.seed + episode_index
    print(f"\n--- Episode {episode_index + 1}/{args.episodes} seed={seed} ---")

    if args.rl_tune:
        return run_rl_tuned_episode(env, grasp, args, episode_index, seed, recorder=recorder)

    return run_grasp_episode_once(
        env,
        grasp,
        args,
        episode_index,
        seed,
        recorder=recorder,
        residual_action=None,
        save_zg_input=bool(args.save_zg_input_dir),
        show_grasp_marker=bool(args.show_grasp_marker),
        verbose=True,
    )


def run_grasp_episode_once(
    env: Any,
    grasp: GraspRecord,
    args: argparse.Namespace,
    episode_index: int,
    seed: int,
    recorder: VideoRecorder | None = None,
    residual_action: ResidualGraspAction | None = None,
    save_zg_input: bool = False,
    show_grasp_marker: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Reset and execute one grasp rollout, optionally with a residual action."""

    obs, _ = env.reset(seed=seed)
    initial_object_z = _target_object_world_z(env)
    if save_zg_input and args.save_zg_input_dir:
        episode_dir = Path(args.save_zg_input_dir) / f"episode_{episode_index:03d}"
        bundle = extract_zerograsp_input(obs, env, args.camera, mask_mode=args.mask_mode)
        saved_dir = save_zerograsp_input_bundle(bundle, episode_dir)
        _log(
            verbose,
            f"  ZeroGrasp input saved: {saved_dir} "
            f"(mask_mode={bundle.mask_mode}, objects={len(bundle.object_records)})",
        )

    world_from_base = _robot_base_matrix(env)
    control_pose = compute_grasp_control_pose(
        grasp,
        _camera_model_matrix(env, args.camera),
        world_from_base,
        approach_axis=args.approach_axis,
    )
    raw_target = control_pose.position_base.copy()
    if residual_action is not None:
        control_pose = apply_residual_grasp_action(control_pose, residual_action)
    motion = MotionConfig(
        pregrasp_offset_m=args.pregrasp_offset,
        lift_offset_m=args.lift_offset,
        stage_steps=args.stage_steps,
        pregrasp_max_steps=args.pregrasp_max_steps,
        max_stage_steps=args.stage_max_steps,
        settle_pos_tolerance_m=args.settle_pos_tolerance,
        descend_settle_pos_tolerance_m=args.descend_settle_pos_tolerance,
        workspace_z_min=args.workspace_z_min,
        gripper_open=(
            1.0 if residual_action is None else float(residual_action.gripper_open)
        ),
        gripper_closed=(
            -1.0 if residual_action is None else float(residual_action.gripper_closed)
        ),
    )
    target = clamp_base_target(control_pose.position_base, z_min=motion.workspace_z_min)
    if show_grasp_marker:
        marker_actors = add_current_grasp_marker(
            env,
            grasp,
            args.camera,
            args.approach_axis,
            center_world=base_point_to_world(target, world_from_base),
        )
        _log(verbose, f"  grasp marker actors added: {len(marker_actors)}")
    if recorder is not None:
        recorder.capture(env)
    _log(verbose, f"  grasp source={grasp.source} score={grasp.score:.6f}")
    _log(verbose, f"  translation_camera={np.round(grasp.translation_m_camera, 4)}")
    _log(verbose, f"  target_base_raw={np.round(raw_target, 4)}")
    if residual_action is not None:
        _log(verbose, f"  residual_action={_round_dict(residual_action.to_dict())}")
        _log(verbose, f"  target_base_adjusted={np.round(control_pose.position_base, 4)}")
    _log(verbose, f"  target_base_clamped={np.round(target, 4)}")
    _log(
        verbose,
        f"  gripper_open={motion.gripper_open:.3f} gripper_closed={motion.gripper_closed:.3f}",
    )
    if args.position_only:
        target_euler = None
        approach_axis = None
        _log(verbose, "  grasp rotation ignored: keeping current TCP orientation")
    else:
        target_euler = control_pose.euler_xyz_base
        approach_axis = control_pose.approach_axis_base
        _log(verbose, f"  approach_axis_convention={args.approach_axis}")
        _log(verbose, f"  approach_axis_base={np.round(approach_axis, 4)}")
        _log(verbose, f"  grasp_tcp_euler_xyz={np.round(target_euler, 4)}")

    result = execute_pick(
        env,
        target,
        motion,
        recorder=recorder,
        target_euler_xyz=target_euler,
        approach_axis_base=approach_axis,
        verbose=verbose,
    )
    final_object_z = _target_object_world_z(env)
    if initial_object_z is not None and final_object_z is not None:
        result["object_lift_delta_m"] = float(final_object_z - initial_object_z)
    if residual_action is not None:
        result["rl_action"] = residual_action.to_dict()
    return result


def run_rl_tuned_episode(
    env: Any,
    grasp: GraspRecord,
    args: argparse.Namespace,
    episode_index: int,
    seed: int,
    recorder: VideoRecorder | None = None,
) -> dict[str, Any]:
    """Tune residual pose/gripper controls on the current scene seed, then execute best action."""

    config = RLTuningConfig(
        iterations=args.rl_iters,
        population=args.rl_population,
        elite_fraction=args.rl_elite_fraction,
        seed=args.rl_seed if args.rl_seed is not None else seed,
        bounds=_residual_action_bounds_from_args(args),
        stop_on_success=bool(args.rl_stop_on_success),
    )
    print(
        "  RL residual tuning enabled: "
        f"method=CEM iters={config.iterations} population={config.population} "
        f"elite_fraction={config.elite_fraction:.2f}"
    )

    def evaluate(
        action: ResidualGraspAction,
        iteration: int,
        candidate_index: int,
    ) -> RLRolloutEvaluation:
        result = run_grasp_episode_once(
            env,
            grasp,
            args,
            episode_index,
            seed,
            recorder=None,
            residual_action=action,
            save_zg_input=False,
            show_grasp_marker=False,
            verbose=False,
        )
        reward = reward_from_rollout_result(result, lift_offset_m=args.lift_offset)
        print(
            f"    rl_iter={iteration} cand={candidate_index} "
            f"reward={reward:.4f} success={bool(result.get('success'))} "
            f"stage={result.get('stage')} "
            f"approach_offset={action.approach_offset_m:.4f} "
            f"closed={action.gripper_closed:.3f}"
        )
        return RLRolloutEvaluation(
            reward=reward,
            success=bool(result.get("success")),
            result=result,
        )

    trace = tune_residual_policy(config, evaluate)
    best_action = ResidualGraspAction(**trace["best_action"])
    print(
        "  RL best residual: "
        f"reward={trace['best_reward']:.4f} success={trace['best_success']} "
        f"action={_round_dict(best_action.to_dict())}"
    )

    final_result = run_grasp_episode_once(
        env,
        grasp,
        args,
        episode_index,
        seed,
        recorder=recorder,
        residual_action=best_action,
        save_zg_input=bool(args.save_zg_input_dir),
        show_grasp_marker=bool(args.show_grasp_marker),
        verbose=True,
    )
    final_result["rl_tuning"] = {
        "method": trace["method"],
        "best_reward": trace["best_reward"],
        "best_success": trace["best_success"],
        "best_action": trace["best_action"],
        "iterations_completed": trace["iterations_completed"],
    }
    _write_rl_trace(args, episode_index, trace, final_result)
    return final_result


def apply_residual_grasp_action(
    control_pose: GraspControlPose,
    action: ResidualGraspAction,
) -> GraspControlPose:
    """Return a control pose corrected by a learned residual action."""

    position, rotation = apply_residual_to_pose(
        control_pose.position_base,
        control_pose.rotation_base_tcp,
        action,
    )
    return GraspControlPose(
        position_base=position,
        rotation_base_tcp=rotation,
        euler_xyz_base=matrix_to_euler_xyz(rotation),
        approach_axis_base=_unit(rotation[:, 2]),
    )


def _residual_action_bounds_from_args(args: argparse.Namespace) -> ResidualActionBounds:
    return ResidualActionBounds.from_ranges(
        approach_offset_range=_pair(args.rl_approach_offset_range, "rl_approach_offset_range"),
        tcp_x_offset_range=_pair(args.rl_tcp_x_offset_range, "rl_tcp_x_offset_range"),
        tcp_y_offset_range=_pair(args.rl_tcp_y_offset_range, "rl_tcp_y_offset_range"),
        roll_delta_range=_pair(args.rl_roll_delta_range, "rl_roll_delta_range"),
        gripper_open_range=_pair(args.rl_gripper_open_range, "rl_gripper_open_range"),
        gripper_closed_range=_pair(args.rl_gripper_closed_range, "rl_gripper_closed_range"),
    )


def _pair(value: Iterable[float], name: str) -> tuple[float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return values


def _write_rl_trace(
    args: argparse.Namespace,
    episode_index: int,
    trace: dict[str, Any],
    final_result: dict[str, Any],
) -> None:
    output = args.rl_output
    if output is None:
        output = str(Path(args.zerograsp_output).expanduser().resolve() / "rl_tuning.json")
    path = Path(output).expanduser().resolve()
    if int(getattr(args, "episodes", 1)) > 1:
        path = path.with_name(f"{path.stem}_episode_{episode_index:03d}{path.suffix}")
    payload = dict(trace)
    payload["episode_index"] = int(episode_index)
    payload["final_result"] = _summarize_result(final_result)
    if "object_lift_delta_m" in final_result:
        payload["final_result"]["object_lift_delta_m"] = final_result["object_lift_delta_m"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  RL tuning trace saved: {path}")


def add_current_grasp_marker(
    env: Any,
    grasp: GraspRecord,
    camera_name: str,
    approach_axis: str = "negative-x",
    center_world: np.ndarray | None = None,
) -> list[Any]:
    """Add the selected ZeroGrasp grasp as visual-only scene markers."""

    geometry = build_grasp_marker_geometry(
        grasp,
        _camera_model_matrix(env, camera_name),
        approach_axis=approach_axis,
        center_world=center_world,
    )
    print(
        "  grasp marker execution_center_world="
        f"{np.round(geometry.center_world, 4)} "
        f"approach={np.round(geometry.approach_axis_world, 4)} "
        f"width_axis={np.round(geometry.width_axis_world, 4)}"
    )
    return add_grasp_marker_to_scene(env.unwrapped.scene, geometry)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    grasp = load_best_grasp(args.zerograsp_output)
    print("ZeroGrasp offline grasp loaded")
    print(f"  source={grasp.source} score={grasp.score:.6f} width={grasp.width_m:.4f}m")

    env = build_env(
        args.width,
        args.height,
        render_width=args.render_width,
        render_height=args.render_height,
        env_id=args.env_id,
        camera_name=args.camera,
        camera_eye=args.camera_eye,
        camera_target=args.camera_target,
    )
    recorder = VideoRecorder(args.video_out, fps=args.video_fps)
    successes = 0
    try:
        for ep in range(args.episodes):
            result = run_episode(env, grasp, args, ep, recorder=recorder)
            print(f"  final_result={_summarize_result(result)}")
            successes += int(bool(result.get("success")))
        video_path = recorder.save()
        if video_path is not None:
            print(f"Video saved: {video_path}")
    finally:
        env.close()

    print(f"\nSummary: {successes}/{args.episodes} success")
    return 0


def make_action(target_base: np.ndarray, euler_xyz: np.ndarray, gripper: float) -> np.ndarray:
    """Build one ManiSkill pd_ee_pose action row."""

    target = np.asarray(target_base, dtype=np.float64).reshape(3)
    euler = np.asarray(euler_xyz, dtype=np.float64).reshape(3)
    return np.array([*target, *euler, gripper], dtype=np.float32)[None, :]


def _orthonormalize_rotation(rotation_matrix: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    u, _, vh = np.linalg.svd(R)
    out = u @ vh
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vh
    return out


def _tcp_rotation_from_approach_and_width(
    approach_axis_base: np.ndarray,
    width_axis_base: np.ndarray,
) -> np.ndarray:
    tcp_z = _unit(approach_axis_base)
    width = _unit(width_axis_base)
    tcp_y = width - tcp_z * float(np.dot(width, tcp_z))
    if np.linalg.norm(tcp_y) < 1e-6:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper, tcp_z))) > 0.95:
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tcp_y = helper - tcp_z * float(np.dot(helper, tcp_z))
    tcp_y = _unit(tcp_y)
    tcp_x = _unit(np.cross(tcp_y, tcp_z))
    tcp_y = _unit(np.cross(tcp_z, tcp_x))
    return np.stack([tcp_x, tcp_y, tcp_z], axis=1)


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return arr / norm


def _camera_model_matrix(env: Any, camera_name: str) -> np.ndarray:
    sensors = env.unwrapped.scene.sensors
    if camera_name not in sensors:
        available = ", ".join(sorted(sensors))
        raise KeyError(f"Camera sensor {camera_name!r} not found. Available sensors: {available}")
    matrix = sensors[camera_name].camera.get_model_matrix()
    return _first_matrix(matrix, "camera model matrix")


def _robot_base_matrix(env: Any) -> np.ndarray:
    pose = env.unwrapped.agent.robot.get_pose()
    return _first_matrix(pose.to_transformation_matrix(), "robot base matrix")


def base_point_to_world(position_base: np.ndarray, world_from_base_matrix: np.ndarray) -> np.ndarray:
    """Convert one base-frame point to world coordinates."""

    point = np.asarray(position_base, dtype=np.float64).reshape(3)
    world_from_base = _first_matrix(world_from_base_matrix, "world_from_base_matrix")
    return (world_from_base @ np.array([*point, 1.0], dtype=np.float64))[:3]


def _tcp_base_position(env: Any) -> np.ndarray | None:
    tcp_world = _tcp_world_position(env)
    if tcp_world is None:
        return None
    base_from_world = np.linalg.inv(_robot_base_matrix(env))
    return (base_from_world @ np.array([*tcp_world, 1.0], dtype=np.float64))[:3]


def _tcp_base_euler_xyz(env: Any) -> np.ndarray:
    arm_controller = getattr(env.unwrapped.agent.controller, "controllers", {}).get("arm")
    if arm_controller is None or not hasattr(arm_controller, "ee_pose_at_base"):
        return np.zeros(3, dtype=np.float64)

    try:
        from mani_skill.utils.geometry.rotation_conversions import (
            matrix_to_euler_angles,
            quaternion_to_matrix,
        )

        pose = arm_controller.ee_pose_at_base
        euler = matrix_to_euler_angles(quaternion_to_matrix(pose.q), "XYZ")
        return _first_vector3(euler, "tcp base euler")
    except Exception as exc:
        print(f"  warning: could not read TCP orientation, using zero Euler target: {exc}")
        return np.zeros(3, dtype=np.float64)


def _tcp_world_position(env: Any) -> np.ndarray | None:
    agent = env.unwrapped.agent
    tcp = getattr(agent, "tcp", None)
    if tcp is not None:
        pose = getattr(tcp, "pose", None)
        if pose is not None:
            return _pose_position(pose)

    for link in agent.robot.get_links():
        name = _link_name(link)
        if name in {"panda_hand_tcp", "tcp", "panda_tcp"}:
            return _pose_position(link.pose)
    return None


def _target_object_world_z(env: Any) -> float | None:
    root = getattr(env, "unwrapped", env)
    for source_name in ("target_object", "obj", "_objs"):
        for actor in _iter_actor_like(getattr(root, source_name, None)):
            pose = getattr(actor, "pose", None)
            if pose is None:
                get_pose = getattr(actor, "get_pose", None)
                if callable(get_pose):
                    pose = get_pose()
            if pose is None:
                continue
            try:
                return float(_pose_position(pose)[2])
            except Exception:
                continue
    return None


def _iter_actor_like(value: Any):
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_actor_like(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_actor_like(item)
        return
    yield value


def _link_name(link: Any) -> str:
    get_name = getattr(link, "get_name", None)
    if callable(get_name):
        return str(get_name())
    return str(getattr(link, "name", ""))


def _pose_position(pose: Any) -> np.ndarray:
    if hasattr(pose, "p"):
        return _first_vector3(pose.p, "pose position")
    raw_pose = getattr(pose, "raw_pose", None)
    if raw_pose is not None:
        return _first_vector3(_to_numpy(raw_pose)[..., :3], "pose raw position")
    raise ValueError(f"Cannot extract position from pose object {pose!r}.")


def _first_matrix(value: Any, name: str) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.shape == (4, 4):
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[0] >= 1 and arr.shape[1:] == (4, 4):
        return arr[0].astype(np.float64)
    raise ValueError(f"{name} must have shape (4, 4) or (N, 4, 4), got {arr.shape}.")


def _first_vector3(value: Any, name: str) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.shape == (3,):
        return arr.astype(np.float64)
    if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 3:
        return arr[0, :3].astype(np.float64)
    raise ValueError(f"{name} must have shape (3,) or (N, >=3), got {arr.shape}.")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize_rgb_frame(frame: Any) -> np.ndarray:
    arr = _to_numpy(frame)
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


def _info_success(info: dict[str, Any]) -> bool:
    if not isinstance(info, dict) or "success" not in info:
        return False
    return _as_bool(info["success"])


def _as_bool(value: Any) -> bool:
    arr = _to_numpy(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(np.any(arr))


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(result.get("success", False)),
        "stage": result.get("stage"),
        "truncated": bool(result.get("truncated", False)),
    }


def _round_dict(values: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {key: float(round(float(value), digits)) for key, value in values.items()}


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
