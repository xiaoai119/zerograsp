from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from maniskill_curobo.scripts.run_full_pipeline import (
    PipelineLayout,
    build_pipeline_steps,
    parse_args,
    prepare_run_layout,
)


class CuroboFullPipelineTest(unittest.TestCase):
    def test_parse_args_uses_curobo_defaults(self) -> None:
        args = parse_args([])

        self.assertEqual(args.output_root, "maniskill_curobo/runs")
        self.assertEqual(args.env_id, "PickSingleYCB-v1")
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.camera, "base_camera")
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 1024)
        self.assertEqual(args.render_width, 1280)
        self.assertEqual(args.render_height, 1024)
        self.assertEqual(args.mask_mode, "task-target")
        self.assertEqual(args.approach_axis, "positive-x")
        self.assertEqual(args.scene_source, "maniskill")
        self.assertFalse(args.scene_include_target_object)
        self.assertEqual(args.zerograsp_env_name, "graduate")

    def test_prepare_run_layout_places_artifacts_under_one_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = prepare_run_layout(Path(tmp), "test_run")

            self.assertEqual(layout.run_dir, Path(tmp).resolve() / "test_run")
            self.assertEqual(layout.input_dir, layout.run_dir / "zg_input")
            self.assertEqual(layout.output_dir, layout.run_dir / "zg_output")
            self.assertEqual(layout.projection_path, layout.run_dir / "grasp_projection.png")
            self.assertEqual(layout.video_path, layout.run_dir / "execution.mp4")
            self.assertEqual(layout.manifest_path, layout.run_dir / "pipeline_manifest.json")
            self.assertTrue(layout.logs_dir.is_dir())

    def test_build_pipeline_steps_uses_curobo_executor(self) -> None:
        args = argparse.Namespace(
            input_dir=None,
            env_id="PickSingleYCB-v1",
            seed=3,
            camera="base_camera",
            width=320,
            height=240,
            camera_eye=[-0.3, 0.0, 0.55],
            camera_target=[0.05, 0.0, 0.08],
            mask_mode="task-target",
            approach_axis="positive-x",
            render_width=640,
            render_height=480,
            pregrasp_offset=0.1,
            lift_offset=0.1,
            workspace_z_min=0.02,
            close_steps=5,
            settle_steps=2,
            action_repeat=1,
            max_waypoints_per_stage=20,
            robot_config="franka.yml",
            scene_source="maniskill",
            scene_include_target_object=False,
            scene_min_cuboid_dimension=0.005,
            scene_model="collision_test.yml",
            warmup_iterations=1,
            video_fps=12,
            checkpoint="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
            config="configs/maniskill.yaml",
            device=None,
            enable_collision_detection=False,
            no_grasp_marker=False,
            maniskill_python="/tmp/maniskill_curobo/bin/python",
            zerograsp_env_name="graduate",
            no_conda=False,
        )
        layout = PipelineLayout(
            run_dir=Path("/tmp/run"),
            input_dir=Path("/tmp/run/zg_input"),
            output_dir=Path("/tmp/run/zg_output"),
            logs_dir=Path("/tmp/run/logs"),
            projection_path=Path("/tmp/run/grasp_projection.png"),
            video_path=Path("/tmp/run/execution.mp4"),
            manifest_path=Path("/tmp/run/pipeline_manifest.json"),
        )

        steps = build_pipeline_steps(args, layout)

        self.assertEqual([step.name for step in steps], ["export_input", "zerograsp", "projection", "execute"])
        self.assertEqual(steps[0].module, "maniskill_codex.export_zerograsp_input")
        self.assertEqual(steps[1].module, "maniskill_codex.run_zerograsp_inference")
        self.assertEqual(steps[2].module, "maniskill_codex.grasp_projection")
        self.assertEqual(steps[3].module, "maniskill_curobo.scripts.execute_curobo_pick")
        self.assertIn(str(layout.output_dir), steps[3].module_args)
        self.assertIn(str(layout.run_dir), steps[3].module_args)
        self.assertIn("--video-out", steps[3].module_args)
        self.assertIn(str(layout.video_path), steps[3].module_args)
        self.assertIn("--scene-source", steps[3].module_args)
        self.assertIn("maniskill", steps[3].module_args)


if __name__ == "__main__":
    unittest.main()
