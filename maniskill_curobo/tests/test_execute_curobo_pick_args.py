from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from maniskill_curobo.scripts.execute_curobo_pick import (
    ControlPose,
    MotionConfig,
    append_planning_diagnostic,
    apply_grasp_depth_offset,
    apply_hand_tcp_calibration,
    build_planning_diagnostic,
    build_control_pose_marker_geometry,
    build_stage_targets,
    copy_zerograsp_output,
    grasp_depth_attempt_scales,
    load_grasp_candidates,
    parse_args,
    repeat_last_action,
    save_grasp_projection,
)


class ExecuteCuroboPickArgsTest(unittest.TestCase):
    def test_video_defaults_match_maniskill_codex(self) -> None:
        args = parse_args(["--target-base", "0.5", "0.0", "0.2"])

        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 1024)
        self.assertEqual(args.render_width, 1280)
        self.assertEqual(args.render_height, 1024)
        self.assertEqual(args.video_fps, 20)
        self.assertEqual(args.mask_mode, "task-target")
        self.assertEqual(args.camera_eye, [-0.30, 0.0, 0.55])
        self.assertEqual(args.camera_target, [0.05, 0.0, 0.08])
        self.assertEqual(args.grasp_depth_scale, 0.0)
        self.assertEqual(args.grasp_depth_max_offset, 0.04)
        self.assertFalse(args.grasp_depth_auto_fallback)
        self.assertEqual(args.candidate_top_k, 1)
        self.assertEqual(
            args.grasp_depth_fallback_fractions,
            [1.0, 0.75, 0.5, 0.25, 0.0],
        )

    def test_grasp_depth_options_are_configurable(self) -> None:
        args = parse_args(
            [
                "--target-base",
                "0.5",
                "0.0",
                "0.2",
                "--grasp-depth-scale",
                "1.0",
                "--grasp-depth-max-offset",
                "0.03",
                "--grasp-depth-auto-fallback",
            ]
        )

        self.assertEqual(args.grasp_depth_scale, 1.0)
        self.assertEqual(args.grasp_depth_max_offset, 0.03)
        self.assertTrue(args.grasp_depth_auto_fallback)

    def test_grasp_marker_is_enabled_by_default(self) -> None:
        args = parse_args(["--target-base", "0.5", "0.0", "0.2"])

        self.assertTrue(args.show_grasp_marker)
        self.assertEqual(args.scene_source, "maniskill")
        self.assertFalse(args.scene_include_target_object)

    def test_fixed_scene_source_keeps_configured_scene_model(self) -> None:
        args = parse_args(
            [
                "--target-base",
                "0.5",
                "0.0",
                "0.2",
                "--scene-source",
                "fixed",
                "--scene-model",
                "collision_test.yml",
            ]
        )

        self.assertEqual(args.scene_source, "fixed")
        self.assertEqual(args.scene_model, "collision_test.yml")

    def test_grasp_marker_can_be_disabled(self) -> None:
        args = parse_args(["--target-base", "0.5", "0.0", "0.2", "--no-grasp-marker"])

        self.assertFalse(args.show_grasp_marker)

    def test_stop_after_stage_can_limit_execution_to_pregrasp(self) -> None:
        default_args = parse_args(["--target-base", "0.5", "0.0", "0.2"])
        pre_only_args = parse_args(
            [
                "--target-base",
                "0.5",
                "0.0",
                "0.2",
                "--stop-after-stage",
                "pre",
            ]
        )

        self.assertIsNone(default_args.stop_after_stage)
        self.assertEqual(pre_only_args.stop_after_stage, "pre")

    def test_marker_geometry_matches_center_approach_and_width_semantics(self) -> None:
        control_pose = ControlPose(
            position_base=np.array([0.5, 0.0, 0.2], dtype=np.float64),
            rotation_base_tool=np.eye(3, dtype=np.float64),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            approach_axis_base=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        world_from_base = np.eye(4, dtype=np.float64)

        geometry = build_control_pose_marker_geometry(
            control_pose=control_pose,
            world_from_base_matrix=world_from_base,
            width_m=0.08,
        )

        np.testing.assert_allclose(geometry.center_world, np.array([0.5, 0.0, 0.2]))
        np.testing.assert_allclose(geometry.approach_axis_world, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(geometry.approach_end_world, np.array([0.5, 0.0, 0.28]))
        np.testing.assert_allclose(geometry.width_endpoints_world[0], np.array([0.5, 0.04, 0.2]))
        np.testing.assert_allclose(geometry.width_endpoints_world[1], np.array([0.5, -0.04, 0.2]))

    def test_hand_tcp_calibration_keeps_marker_center_and_offsets_planner_target(self) -> None:
        control_pose = ControlPose(
            position_base=np.array([0.5, 0.0, 0.4], dtype=np.float64),
            rotation_base_tool=np.eye(3, dtype=np.float64),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            approach_axis_base=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

        calibrated = apply_hand_tcp_calibration(
            control_pose,
            hand_to_tcp_translation=np.array([0.0, 0.0, 0.1], dtype=np.float64),
            workspace_z_min=0.0,
        )
        targets = build_stage_targets(
            calibrated,
            MotionConfig(pregrasp_offset_m=0.2, lift_offset_m=0.5, workspace_z_min=0.0),
        )

        np.testing.assert_allclose(calibrated.position_base, np.array([0.5, 0.0, 0.4]))
        np.testing.assert_allclose(calibrated.planner_position_base, np.array([0.5, 0.0, 0.3]))
        np.testing.assert_allclose(targets["grasp"]["position"], np.array([0.5, 0.0, 0.3]))
        np.testing.assert_allclose(targets["pre"]["position"], np.array([0.5, 0.0, 0.1]))
        np.testing.assert_allclose(targets["lift"]["position"], np.array([0.5, 0.0, 0.6]))

    def test_grasp_depth_moves_tcp_forward_along_approach_axis(self) -> None:
        control_pose = ControlPose(
            position_base=np.array([0.5, 0.0, 0.2], dtype=np.float64),
            rotation_base_tool=np.eye(3, dtype=np.float64),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            approach_axis_base=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )

        adjusted, manifest = apply_grasp_depth_offset(
            control_pose,
            depth_m=0.03,
            scale=0.5,
            max_offset_m=0.04,
            workspace_z_min=0.01,
        )

        np.testing.assert_allclose(adjusted.position_base, np.array([0.5, 0.0, 0.185]))
        self.assertAlmostEqual(manifest["requested_offset_m"], 0.015)
        self.assertAlmostEqual(manifest["applied_distance_m"], 0.015)
        self.assertFalse(manifest["workspace_clamped"])

    def test_grasp_depth_offset_is_limited_and_workspace_clamped(self) -> None:
        control_pose = ControlPose(
            position_base=np.array([0.5, 0.0, 0.02], dtype=np.float64),
            rotation_base_tool=np.eye(3, dtype=np.float64),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            approach_axis_base=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )

        adjusted, manifest = apply_grasp_depth_offset(
            control_pose,
            depth_m=0.04,
            scale=1.0,
            max_offset_m=0.03,
            workspace_z_min=0.01,
        )

        np.testing.assert_allclose(adjusted.position_base, np.array([0.5, 0.0, 0.01]))
        self.assertAlmostEqual(manifest["requested_offset_m"], 0.03)
        self.assertAlmostEqual(manifest["applied_distance_m"], 0.01)
        self.assertTrue(manifest["workspace_clamped"])

    def test_pregrasp_can_remain_anchored_to_zero_depth_pose(self) -> None:
        zero_pose = ControlPose(
            position_base=np.array([0.5, 0.0, 0.2], dtype=np.float64),
            rotation_base_tool=np.eye(3, dtype=np.float64),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            approach_axis_base=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        depth_pose, _ = apply_grasp_depth_offset(
            zero_pose,
            depth_m=0.04,
            scale=1.0,
            max_offset_m=0.04,
            workspace_z_min=0.01,
        )

        targets = build_stage_targets(
            depth_pose,
            MotionConfig(pregrasp_offset_m=0.1, lift_offset_m=0.15),
            pregrasp_control_pose=zero_pose,
        )

        np.testing.assert_allclose(targets["pre"]["position"], [0.4, 0.0, 0.2])
        np.testing.assert_allclose(targets["grasp"]["position"], [0.54, 0.0, 0.2])

    def test_grasp_depth_attempt_scales_are_progressively_shallower(self) -> None:
        self.assertEqual(
            grasp_depth_attempt_scales(1.0, [1.0, 0.75, 0.5, 0.25, 0.0]),
            [1.0, 0.75, 0.5, 0.25, 0.0],
        )
        self.assertEqual(
            grasp_depth_attempt_scales(0.5, [1.0, 0.5, 0.0]),
            [0.5, 0.25, 0.0],
        )

    def test_grasp_depth_attempt_scales_reject_invalid_fractions(self) -> None:
        with self.assertRaises(ValueError):
            grasp_depth_attempt_scales(1.0, [1.2])

    def test_grasp_settle_holds_the_closed_gripper_action(self) -> None:
        grasp_actions = np.array(
            [
                [0.1, 0.2, 0.3, 1.0],
                [0.4, 0.5, 0.6, 1.0],
            ],
            dtype=np.float32,
        )
        close_actions = repeat_last_action(grasp_actions, steps=2)
        close_actions[:, -1] = -1.0

        hold_actions = repeat_last_action(close_actions, steps=3)

        self.assertEqual(hold_actions.shape, (3, 4))
        np.testing.assert_allclose(hold_actions[:, :3], np.array([[0.4, 0.5, 0.6]] * 3))
        np.testing.assert_allclose(hold_actions[:, -1], np.array([-1.0, -1.0, -1.0]))

    def test_build_planning_diagnostic_summarizes_curobo_result(self) -> None:
        result = SimpleNamespace(
            success=np.array([False]),
            status="IK_FAIL",
            planning_time=0.125,
            position_error=np.array([[0.043]]),
            rotation_error=np.array([[0.21]]),
            feasible=np.array([[False, True, False]]),
            seed_cost=np.array([[5.0, 3.0, 7.0]]),
            seed_rank=np.array([[1, 0, 2]]),
            debug_info={"collision": np.array([1])},
        )

        diagnostic = build_planning_diagnostic(
            stage_name="grasp",
            q_start=np.array([0.1, 0.2], dtype=np.float32),
            target_position=np.array([0.5, 0.0, 0.2], dtype=np.float32),
            target_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            result=result,
        )

        self.assertEqual(diagnostic["stage"], "grasp")
        self.assertFalse(diagnostic["success"])
        self.assertEqual(diagnostic["status"], "IK_FAIL")
        self.assertEqual(diagnostic["failure_reason"], "curobo_planning_failed")
        self.assertEqual(diagnostic["feasible_count"], 1)
        self.assertEqual(diagnostic["feasible_total"], 3)
        self.assertAlmostEqual(diagnostic["seed_cost_min"], 3.0)
        self.assertAlmostEqual(diagnostic["seed_cost_mean"], 5.0)
        self.assertEqual(diagnostic["debug_info"], {"collision": [1]})
        json.dumps(diagnostic)

    def test_planning_diagnostic_stringifies_non_json_debug_values(self) -> None:
        custom_debug_value = object()
        result = SimpleNamespace(
            success=np.array([False]),
            status="IK_FAIL",
            debug_info={"raw_object": custom_debug_value},
        )

        diagnostic = build_planning_diagnostic(
            stage_name="pre",
            q_start=np.array([0.1], dtype=np.float32),
            target_position=np.array([0.5, 0.0, 0.2], dtype=np.float32),
            target_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            result=result,
        )

        self.assertIsInstance(diagnostic["debug_info"]["raw_object"], str)
        json.dumps(diagnostic)

    def test_append_planning_diagnostic_writes_stage_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "planning_diagnostics.json"

            append_planning_diagnostic(path, {"stage": "pre", "success": True})
            append_planning_diagnostic(path, {"stage": "grasp", "success": False})

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                data,
                {
                    "stages": [
                        {"stage": "pre", "success": True},
                        {"stage": "grasp", "success": False},
                    ]
                },
            )

    def test_copy_zerograsp_output_keeps_recommended_and_raw_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            raw = source / "raw_outputs"
            raw.mkdir(parents=True)
            (source / "recommended_grasp_top1.json").write_text('{"score": 1.0}', encoding="utf-8")
            (raw / "obj_001.grasp.npy").write_bytes(b"fake")
            destination = root / "run" / "zg_output"

            copied = copy_zerograsp_output(source, destination)

            self.assertEqual(copied, destination.resolve())
            self.assertEqual(
                (copied / "recommended_grasp_top1.json").read_text(encoding="utf-8"),
                '{"score": 1.0}',
            )
            self.assertEqual((copied / "raw_outputs" / "obj_001.grasp.npy").read_bytes(), b"fake")

    def test_load_grasp_candidates_reads_topk_json_by_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            payload = {
                "grasps": [
                    {
                        "score": 0.4,
                        "width_m": 0.05,
                        "height_m": 0.02,
                        "depth_m": 0.03,
                        "rotation_matrix_camera": np.eye(3).tolist(),
                        "translation_m_camera": [0.0, 0.0, 0.5],
                        "object_id": 1,
                    },
                    {
                        "score": 0.9,
                        "width_m": 0.06,
                        "height_m": 0.02,
                        "depth_m": 0.04,
                        "rotation_matrix_camera": np.eye(3).tolist(),
                        "translation_m_camera": [0.1, 0.0, 0.5],
                        "object_id": 1,
                    },
                ]
            }
            (output_dir / "recommended_grasps_topk.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            candidates = load_grasp_candidates(output_dir, top_k=2)

            self.assertEqual(len(candidates), 2)
            self.assertAlmostEqual(candidates[0].score, 0.9)
            self.assertAlmostEqual(candidates[1].score, 0.4)

    def test_load_grasp_candidates_prefers_raw_arrays_for_topk_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            raw_dir = output_dir / "raw_outputs"
            raw_dir.mkdir()
            (output_dir / "recommended_grasp_top1.json").write_text(
                json.dumps(
                    {
                        "score": 0.1,
                        "width_m": 0.04,
                        "height_m": 0.02,
                        "depth_m": 0.02,
                        "rotation_matrix_camera": np.eye(3).tolist(),
                        "translation_m_camera": [0.0, 0.0, 0.5],
                    }
                ),
                encoding="utf-8",
            )
            rows = np.array(
                [
                    [0.2, 0.04, 0.02, 0.02, *np.eye(3).reshape(-1), 0.0, 0.0, 0.5, 1],
                    [0.8, 0.05, 0.02, 0.03, *np.eye(3).reshape(-1), 0.1, 0.0, 0.5, 1],
                ],
                dtype=np.float64,
            )
            np.save(raw_dir / "object_000_label_1.grasp.npy", rows)

            top1 = load_grasp_candidates(output_dir, top_k=1)
            topk = load_grasp_candidates(output_dir, top_k=2)

            self.assertEqual(len(top1), 1)
            self.assertAlmostEqual(top1[0].score, 0.1)
            self.assertEqual(len(topk), 2)
            self.assertAlmostEqual(topk[0].score, 0.8)

    def test_copy_zerograsp_output_is_noop_when_source_is_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "zg_output"
            output_dir.mkdir()
            (output_dir / "recommended_grasp_top1.json").write_text('{"score": 1.0}', encoding="utf-8")

            copied = copy_zerograsp_output(output_dir, output_dir)

            self.assertEqual(copied, output_dir.resolve())
            self.assertTrue((output_dir / "recommended_grasp_top1.json").is_file())

    def test_save_grasp_projection_writes_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "zg_input"
            output_dir = root / "zg_output"
            input_dir.mkdir()
            output_dir.mkdir()
            Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(input_dir / "rgb.png")
            (input_dir / "camera.json").write_text(
                json.dumps({"cam_K": [100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0]}),
                encoding="utf-8",
            )
            (output_dir / "recommended_grasp_top1.json").write_text(
                json.dumps(
                    {
                        "score": 0.9,
                        "width_m": 0.04,
                        "height_m": 0.02,
                        "depth_m": 0.02,
                        "translation_m_camera": [0.0, 0.0, 1.0],
                        "rotation_matrix_camera": np.eye(3).tolist(),
                    }
                ),
                encoding="utf-8",
            )

            projection = save_grasp_projection(
                input_dir=input_dir,
                zerograsp_output_dir=output_dir,
                output_path=root / "grasp_projection.png",
                approach_axis="negative-x",
            )

            self.assertEqual(projection, (root / "grasp_projection.png").resolve())
            self.assertTrue(projection.is_file())


if __name__ == "__main__":
    unittest.main()
