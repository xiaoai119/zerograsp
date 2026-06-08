import tempfile
import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

import maniskill_codex.execute_zerograsp_pick as execute_mod
from maniskill_codex.execute_zerograsp_pick import (
    apply_residual_grasp_action,
    base_point_to_world,
    build_env,
    build_stage_targets,
    compute_grasp_control_pose,
    execute_pick,
    MotionConfig,
    VideoRecorder,
    clamp_base_target,
    make_action,
    matrix_to_euler_xyz,
    parse_args,
)
from maniskill_codex.grasp_rl_tuning import ResidualGraspAction
from maniskill_codex.zerograsp_outputs import GraspRecord


class FakeRenderEnv:
    def __init__(self, frame):
        self.frame = frame

    def render(self):
        return self.frame


class ExecuteRunnerTests(unittest.TestCase):
    def test_parse_args_uses_safe_defaults(self):
        args = parse_args(["--zerograsp-output", "output"])

        self.assertEqual(args.zerograsp_output, "output")
        self.assertEqual(args.episodes, 1)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.env_id, "PickSingleYCB-v1")
        self.assertEqual(args.camera, "base_camera")
        self.assertEqual(args.render_width, 1280)
        self.assertEqual(args.render_height, 1024)
        self.assertIsNone(args.video_out)
        self.assertEqual(args.video_fps, 20)
        self.assertIsNone(args.save_zg_input_dir)
        self.assertEqual(args.mask_mode, "task-target")
        self.assertEqual(args.approach_axis, "negative-x")
        self.assertEqual(args.camera_eye, [-0.30, 0.0, 0.55])
        self.assertEqual(args.camera_target, [0.05, 0.0, 0.08])
        self.assertEqual(args.workspace_z_min, 0.02)
        self.assertEqual(args.pregrasp_max_steps, 200)
        self.assertEqual(args.stage_max_steps, 80)
        self.assertEqual(args.settle_pos_tolerance, 0.01)
        self.assertEqual(args.descend_settle_pos_tolerance, 0.02)
        self.assertFalse(args.show_grasp_marker)
        self.assertFalse(args.position_only)
        self.assertFalse(args.rl_tune)
        self.assertEqual(args.rl_iters, 3)
        self.assertEqual(args.rl_population, 8)
        self.assertEqual(tuple(args.rl_approach_offset_range), (0.0, 0.05))
        self.assertEqual(tuple(args.rl_gripper_closed_range), (-1.0, -0.2))

    def test_parse_args_accepts_video_output(self):
        args = parse_args(
            [
                "--zerograsp-output",
                "output",
                "--video-out",
                "maniskill_codex/videos/out.mp4",
                "--video-fps",
                "12",
            ]
        )

        self.assertEqual(args.video_out, "maniskill_codex/videos/out.mp4")
        self.assertEqual(args.video_fps, 12)

    def test_parse_args_accepts_zerograsp_input_output_dir(self):
        args = parse_args(
            [
                "--zerograsp-output",
                "output",
                "--save-zg-input-dir",
                "maniskill_codex/zg_inputs",
            ]
        )

        self.assertEqual(args.save_zg_input_dir, "maniskill_codex/zg_inputs")

    def test_parse_args_accepts_grasp_marker_flag(self):
        args = parse_args(["--zerograsp-output", "output", "--show-grasp-marker"])

        self.assertTrue(args.show_grasp_marker)

    def test_parse_args_accepts_position_only_mode(self):
        args = parse_args(["--zerograsp-output", "output", "--position-only"])

        self.assertTrue(args.position_only)

    def test_parse_args_accepts_positive_x_approach_axis(self):
        args = parse_args(["--zerograsp-output", "output", "--approach-axis", "positive-x"])

        self.assertEqual(args.approach_axis, "positive-x")

    def test_parse_args_accepts_flip_world_z_approach_axis(self):
        args = parse_args(["--zerograsp-output", "output", "--approach-axis", "flip-world-z"])

        self.assertEqual(args.approach_axis, "flip-world-z")

    def test_parse_args_accepts_env_id(self):
        args = parse_args(["--zerograsp-output", "output", "--env-id", "PickClutterYCB-v1"])

        self.assertEqual(args.env_id, "PickClutterYCB-v1")

    def test_parse_args_accepts_render_resolution(self):
        args = parse_args(
            [
                "--zerograsp-output",
                "output",
                "--render-width",
                "1920",
                "--render-height",
                "1080",
            ]
        )

        self.assertEqual(args.render_width, 1920)
        self.assertEqual(args.render_height, 1080)

    def test_build_env_configures_sensor_and_render_resolution(self):
        calls = []

        fake_gym = types.SimpleNamespace(
            make=lambda *args, **kwargs: calls.append((args, kwargs)) or "env"
        )
        fake_mani_skill = types.ModuleType("mani_skill")
        fake_mani_skill_envs = types.ModuleType("mani_skill.envs")

        originals = {
            name: sys.modules.get(name)
            for name in ("gymnasium", "mani_skill", "mani_skill.envs")
        }
        try:
            sys.modules["gymnasium"] = fake_gym
            sys.modules["mani_skill"] = fake_mani_skill
            sys.modules["mani_skill.envs"] = fake_mani_skill_envs

            env = build_env(640, 480, render_width=1920, render_height=1080, env_id="PickClutterYCB-v1")
        finally:
            for name, module in originals.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(env, "env")
        self.assertEqual(calls[0][0], ("PickClutterYCB-v1",))
        self.assertEqual(calls[0][1]["sensor_configs"], {"width": 640, "height": 480})
        self.assertEqual(
            calls[0][1]["human_render_camera_configs"],
            {"width": 1920, "height": 1080},
        )

    def test_build_env_can_override_camera_pose(self):
        calls = []

        fake_gym = types.SimpleNamespace(
            make=lambda *args, **kwargs: calls.append((args, kwargs)) or "env"
        )
        fake_mani_skill = types.ModuleType("mani_skill")
        fake_mani_skill_envs = types.ModuleType("mani_skill.envs")
        fake_mani_skill_utils = types.ModuleType("mani_skill.utils")
        fake_mani_skill_utils.sapien_utils = types.SimpleNamespace(
            look_at=lambda eye, target: ("look_at", tuple(eye), tuple(target))
        )

        originals = {
            name: sys.modules.get(name)
            for name in ("gymnasium", "mani_skill", "mani_skill.envs", "mani_skill.utils")
        }
        try:
            sys.modules["gymnasium"] = fake_gym
            sys.modules["mani_skill"] = fake_mani_skill
            sys.modules["mani_skill.envs"] = fake_mani_skill_envs
            sys.modules["mani_skill.utils"] = fake_mani_skill_utils

            env = build_env(
                640,
                480,
                env_id="PickSingleYCB-v1",
                camera_name="base_camera",
                camera_eye=[-0.30, 0.0, 0.55],
                camera_target=[0.05, 0.0, 0.08],
            )
        finally:
            for name, module in originals.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(env, "env")
        self.assertEqual(
            calls[0][1]["sensor_configs"],
            {
                "width": 640,
                "height": 480,
                "base_camera": {
                        "pose": (
                            "look_at",
                            (-0.30, 0.0, 0.55),
                            (0.05, 0.0, 0.08),
                        )
                },
            },
        )

    def test_compute_grasp_control_pose_maps_grasp_axes_to_tcp_axes(self):
        grasp = GraspRecord(
            score=0.5,
            width_m=0.04,
            height_m=0.02,
            depth_m=0.03,
            translation_m_camera=np.array([0.1, 0.2, 0.3]),
            rotation_matrix_camera=np.eye(3),
            source="test",
        )

        pose = compute_grasp_control_pose(
            grasp,
            camera_model_matrix=np.eye(4),
            world_from_base_matrix=np.eye(4),
        )

        np.testing.assert_allclose(pose.position_base, [0.1, -0.2, -0.3])
        np.testing.assert_allclose(pose.approach_axis_base, [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(pose.rotation_base_tcp[:, 1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(pose.rotation_base_tcp[:, 2], pose.approach_axis_base)

    def test_compute_grasp_control_pose_can_use_positive_x_as_approach(self):
        grasp = GraspRecord(
            score=0.5,
            width_m=0.04,
            height_m=0.02,
            depth_m=0.03,
            translation_m_camera=np.array([0.1, 0.2, 0.3]),
            rotation_matrix_camera=np.eye(3),
            source="test",
        )

        pose = compute_grasp_control_pose(
            grasp,
            camera_model_matrix=np.eye(4),
            world_from_base_matrix=np.eye(4),
            approach_axis="positive-x",
        )

        np.testing.assert_allclose(pose.approach_axis_base, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(pose.rotation_base_tcp[:, 1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(pose.rotation_base_tcp[:, 2], pose.approach_axis_base)

    def test_compute_grasp_control_pose_can_flip_only_world_z_approach(self):
        grasp = GraspRecord(
            score=0.5,
            width_m=0.04,
            height_m=0.02,
            depth_m=0.03,
            translation_m_camera=np.array([0.1, 0.2, 0.3]),
            rotation_matrix_camera=np.eye(3),
            source="test",
        )
        camera_model = np.eye(4)
        camera_model[:3, :3] = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        negative_x_pose = compute_grasp_control_pose(
            grasp,
            camera_model_matrix=camera_model,
            world_from_base_matrix=np.eye(4),
            approach_axis="negative-x",
        )
        flipped_pose = compute_grasp_control_pose(
            grasp,
            camera_model_matrix=camera_model,
            world_from_base_matrix=np.eye(4),
            approach_axis="flip-world-z",
        )

        expected = negative_x_pose.approach_axis_base.copy()
        expected[2] *= -1.0
        np.testing.assert_allclose(flipped_pose.approach_axis_base, expected)
        self.assertGreater(np.linalg.det(flipped_pose.rotation_base_tcp), 0.99)
        np.testing.assert_allclose(flipped_pose.rotation_base_tcp[:, 2], flipped_pose.approach_axis_base)

    def test_matrix_to_euler_xyz_round_trips_expected_convention(self):
        euler = np.array([0.2, -0.3, 0.4])
        rotation = _euler_xyz_to_matrix(euler)

        round_trip = matrix_to_euler_xyz(rotation)

        np.testing.assert_allclose(round_trip, euler, atol=1e-7)

    def test_build_stage_targets_uses_grasp_approach_for_pregrasp(self):
        motion = MotionConfig(pregrasp_offset_m=0.10, lift_offset_m=0.15)

        stages = build_stage_targets(
            np.array([0.5, 0.0, 0.2]),
            np.array([0.0, 0.0, -1.0]),
            motion,
        )

        np.testing.assert_allclose(stages["pre"], [0.5, 0.0, 0.3])
        np.testing.assert_allclose(stages["grasp"], [0.5, 0.0, 0.2])
        np.testing.assert_allclose(stages["lift"], [0.5, 0.0, 0.35])

    def test_apply_residual_grasp_action_moves_deeper_along_approach(self):
        control_pose = compute_grasp_control_pose(
            GraspRecord(
                score=0.5,
                width_m=0.04,
                height_m=0.02,
                depth_m=0.03,
                translation_m_camera=np.array([0.1, 0.2, 0.3]),
                rotation_matrix_camera=np.eye(3),
                source="test",
            ),
            camera_model_matrix=np.eye(4),
            world_from_base_matrix=np.eye(4),
        )

        adjusted = apply_residual_grasp_action(
            control_pose,
            ResidualGraspAction(approach_offset_m=0.02),
        )

        np.testing.assert_allclose(
            adjusted.position_base,
            control_pose.position_base + control_pose.approach_axis_base * 0.02,
        )
        np.testing.assert_allclose(adjusted.approach_axis_base, control_pose.approach_axis_base)

    def test_clamp_base_target_keeps_point_inside_workspace(self):
        target = clamp_base_target(np.array([0.1, -1.0, 2.0]))

        np.testing.assert_allclose(target, [0.25, -0.45, 0.6])

    def test_clamp_base_target_uses_lower_default_z_floor(self):
        target = clamp_base_target(np.array([0.5, 0.0, 0.0]))

        np.testing.assert_allclose(target, [0.5, 0.0, 0.02])

    def test_build_stage_targets_uses_configured_workspace_z_floor(self):
        motion = MotionConfig(pregrasp_offset_m=0.10, lift_offset_m=0.15, workspace_z_min=0.03)

        stages = build_stage_targets(
            np.array([0.5, 0.0, 0.01]),
            np.array([0.0, 0.0, -1.0]),
            motion,
        )

        np.testing.assert_allclose(stages["grasp"], [0.5, 0.0, 0.03])

    def test_base_point_to_world_applies_robot_base_transform(self):
        world_from_base = np.eye(4)
        world_from_base[:3, 3] = [1.0, 2.0, 3.0]

        point = base_point_to_world(np.array([0.5, -0.25, 0.1]), world_from_base)

        np.testing.assert_allclose(point, [1.5, 1.75, 3.1])

    def test_motion_config_uses_normalized_gripper_commands(self):
        motion = MotionConfig()

        self.assertEqual(motion.gripper_open, 1.0)
        self.assertEqual(motion.gripper_closed, -1.0)
        self.assertEqual(motion.pregrasp_max_steps, 200)
        self.assertEqual(motion.max_stage_steps, 80)
        self.assertEqual(motion.settle_pos_tolerance_m, 0.01)
        self.assertEqual(motion.descend_settle_pos_tolerance_m, 0.02)

    def test_make_action_combines_position_rotation_and_gripper(self):
        action = make_action(
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            -1.0,
        )

        np.testing.assert_allclose(action, [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]])
        self.assertEqual(action.dtype, np.float32)

    def test_video_recorder_captures_rgb_frame(self):
        frame = np.zeros((1, 4, 5, 3), dtype=np.uint8)
        frame[0, :, :, 1] = 128

        with tempfile.TemporaryDirectory() as tmp:
            recorder = VideoRecorder(Path(tmp) / "out.mp4", fps=10)
            recorder.capture(FakeRenderEnv(frame))

            self.assertEqual(len(recorder.frames), 1)
            self.assertEqual(recorder.frames[0].shape, (4, 5, 3))
            self.assertEqual(recorder.frames[0].dtype, np.uint8)
            self.assertEqual(int(recorder.frames[0][0, 0, 1]), 128)

    def test_execute_pick_waits_until_tcp_reaches_pregrasp(self):
        env = FakeStepEnv(reach_after=3)
        motion = MotionConfig(
            pregrasp_offset_m=0.10,
            lift_offset_m=0.15,
            stage_steps=2,
            pregrasp_max_steps=5,
            max_stage_steps=5,
            settle_pos_tolerance_m=0.01,
        )

        with patch.object(execute_mod, "_tcp_base_position", _fake_tcp_base_position):
            result = execute_pick(
                env,
                np.array([0.5, 0.0, 0.2]),
                motion,
                target_euler_xyz=np.zeros(3),
                approach_axis_base=np.array([0.0, 0.0, -1.0]),
            )

        pre_target = np.array([0.5, 0.0, 0.3])
        pre_steps = _count_initial_target_steps(env.actions, pre_target)
        self.assertEqual(pre_steps, 3)
        self.assertEqual(result["stage"], "lift")

    def test_execute_pick_aborts_when_pregrasp_does_not_converge(self):
        env = FakeStepEnv(reach_after=99)
        motion = MotionConfig(
            pregrasp_offset_m=0.10,
            lift_offset_m=0.15,
            stage_steps=2,
            pregrasp_max_steps=6,
            max_stage_steps=4,
            settle_pos_tolerance_m=0.01,
        )

        with patch.object(execute_mod, "_tcp_base_position", _fake_tcp_base_position):
            result = execute_pick(
                env,
                np.array([0.5, 0.0, 0.2]),
                motion,
                target_euler_xyz=np.zeros(3),
                approach_axis_base=np.array([0.0, 0.0, -1.0]),
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "pre")
        self.assertTrue(result["not_converged"])
        self.assertEqual(len(env.actions), 6)

    def test_execute_pick_uses_looser_descend_tolerance_before_closing(self):
        env = FakeStepEnv(reach_after=1)
        motion = MotionConfig(
            pregrasp_offset_m=0.10,
            lift_offset_m=0.15,
            stage_steps=2,
            pregrasp_max_steps=5,
            max_stage_steps=5,
            settle_pos_tolerance_m=0.01,
            descend_settle_pos_tolerance_m=0.02,
        )

        with patch.object(execute_mod, "_tcp_base_position", _fake_tcp_base_position_with_descend_error):
            result = execute_pick(
                env,
                np.array([0.5, 0.0, 0.2]),
                motion,
                target_euler_xyz=np.zeros(3),
                approach_axis_base=np.array([0.0, 0.0, -1.0]),
            )

        closed_actions = [action for action in env.actions if action[0, 6] == -1.0]
        self.assertEqual(result["stage"], "lift")
        self.assertGreater(len(closed_actions), 0)


if __name__ == "__main__":
    unittest.main()


def _euler_xyz_to_matrix(euler):
    x, y, z = euler
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


class FakeStepEnv:
    def __init__(self, reach_after):
        self.reach_after = reach_after
        self.actions = []
        self.current_target = np.zeros(3)
        self.current_target_count = 0
        self.target_counts = {}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        target = np.asarray(action[0, :3], dtype=np.float64)
        key = tuple(np.round(target, 6))
        self.target_counts[key] = self.target_counts.get(key, 0) + 1
        self.current_target = target
        self.current_target_count = self.target_counts[key]
        return None, None, None, False, {"success": False}


def _fake_tcp_base_position(env):
    if env.current_target_count < env.reach_after:
        return env.current_target + np.array([0.05, 0.0, 0.0])
    return env.current_target.copy()


def _fake_tcp_base_position_with_descend_error(env):
    if np.allclose(env.current_target, [0.5, 0.0, 0.2]):
        return env.current_target + np.array([0.015, 0.0, 0.0])
    return env.current_target.copy()


def _count_initial_target_steps(actions, target):
    count = 0
    for action in actions:
        if np.allclose(action[0, :3], target):
            count += 1
        else:
            break
    return count
