import argparse
import tempfile
import unittest
from pathlib import Path

from maniskill_codex.run_full_pipeline import (
    PipelineLayout,
    build_pipeline_steps,
    conda_python_command,
    parse_args,
    prepare_run_layout,
)


class FullPipelineTests(unittest.TestCase):
    def test_parse_args_uses_reproducible_defaults(self):
        args = parse_args([])

        self.assertEqual(args.env_id, "PickClutterYCB-v1")
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.camera, "base_camera")
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 1024)
        self.assertEqual(args.mask_mode, "task-target")
        self.assertEqual(args.approach_axis, "negative-x")
        self.assertEqual(args.camera_eye, [-0.30, 0.0, 0.55])
        self.assertEqual(args.camera_target, [0.05, 0.0, 0.08])
        self.assertEqual(args.render_width, 1280)
        self.assertEqual(args.render_height, 1024)
        self.assertEqual(args.workspace_z_min, 0.02)
        self.assertEqual(args.pregrasp_max_steps, 200)
        self.assertEqual(args.stage_max_steps, 80)
        self.assertEqual(args.settle_pos_tolerance, 0.01)
        self.assertEqual(args.descend_settle_pos_tolerance, 0.02)
        self.assertEqual(args.maniskill_env_name, "maniskill")
        self.assertEqual(args.zerograsp_env_name, "graduate")
        self.assertFalse(args.position_only)

    def test_prepare_run_layout_places_all_artifacts_under_one_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = prepare_run_layout(Path(tmp), "test_run")

            self.assertEqual(layout.run_dir, Path(tmp).resolve() / "test_run")
            self.assertEqual(layout.input_dir, layout.run_dir / "zg_input")
            self.assertEqual(layout.output_dir, layout.run_dir / "zg_output")
            self.assertEqual(layout.projection_path, layout.run_dir / "grasp_projection.png")
            self.assertEqual(layout.video_path, layout.run_dir / "execution.mp4")
            self.assertEqual(layout.manifest_path, layout.run_dir / "run_manifest.json")
            self.assertTrue(layout.logs_dir.is_dir())

    def test_build_pipeline_steps_uses_layout_paths(self):
        args = argparse.Namespace(
            env_id="PickClutterYCB-v1",
            seed=7,
            camera="base_camera",
            width=320,
            height=240,
            mask_mode="task-target",
            approach_axis="flip-world-z",
            camera_eye=[-0.30, 0.0, 0.55],
            camera_target=[0.05, 0.0, 0.08],
            render_width=640,
            render_height=480,
            stage_steps=3,
            pregrasp_max_steps=11,
            stage_max_steps=9,
            settle_pos_tolerance=0.02,
            descend_settle_pos_tolerance=0.03,
            workspace_z_min=0.03,
            pregrasp_offset=0.1,
            lift_offset=0.15,
            video_fps=12,
            checkpoint="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
            config="configs/maniskill.yaml",
            device=None,
            enable_collision_detection=False,
            no_grasp_marker=False,
            position_only=False,
            input_dir=None,
            maniskill_env_name="maniskill",
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
            manifest_path=Path("/tmp/run/run_manifest.json"),
        )

        steps = build_pipeline_steps(args, layout)

        self.assertEqual([step.name for step in steps], ["export_input", "zerograsp", "projection", "execute"])
        self.assertEqual(steps[0].conda_env, "maniskill")
        self.assertIn("--output-dir", steps[0].module_args)
        self.assertIn(str(layout.input_dir), steps[0].module_args)
        self.assertIn("--mask-mode", steps[0].module_args)
        self.assertIn("task-target", steps[0].module_args)
        self.assertIn("--camera-eye", steps[0].module_args)
        self.assertIn("-0.3", steps[0].module_args)
        self.assertIn("--camera-target", steps[0].module_args)
        self.assertEqual(steps[1].conda_env, "graduate")
        self.assertIn(str(layout.output_dir), steps[1].module_args)
        self.assertIn(str(layout.projection_path), steps[2].module_args)
        self.assertIn("--approach-axis", steps[2].module_args)
        self.assertIn("flip-world-z", steps[2].module_args)
        self.assertIn(str(layout.video_path), steps[3].module_args)
        self.assertIn("--approach-axis", steps[3].module_args)
        self.assertIn("--camera-eye", steps[3].module_args)
        self.assertIn("--camera-target", steps[3].module_args)
        self.assertIn("--stage-max-steps", steps[3].module_args)
        self.assertIn("9", steps[3].module_args)
        self.assertIn("--pregrasp-max-steps", steps[3].module_args)
        self.assertIn("11", steps[3].module_args)
        self.assertIn("--settle-pos-tolerance", steps[3].module_args)
        self.assertIn("0.02", steps[3].module_args)
        self.assertIn("--descend-settle-pos-tolerance", steps[3].module_args)
        self.assertIn("0.03", steps[3].module_args)
        self.assertIn("--workspace-z-min", steps[3].module_args)
        self.assertIn("0.03", steps[3].module_args)
        self.assertIn("--show-grasp-marker", steps[3].module_args)

    def test_conda_python_command_quotes_module_args(self):
        command = conda_python_command(
            "maniskill",
            "maniskill_codex.some_module",
            ["--path", "/tmp/a path/file.txt"],
            cwd=Path("/repo"),
            use_conda=True,
        )

        self.assertEqual(command[:2], ["bash", "-lc"])
        self.assertIn("conda activate maniskill", command[2])
        self.assertIn("python -m maniskill_codex.some_module", command[2])
        self.assertIn("'/tmp/a path/file.txt'", command[2])


if __name__ == "__main__":
    unittest.main()
