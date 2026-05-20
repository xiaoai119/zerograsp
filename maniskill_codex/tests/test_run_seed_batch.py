import json
import tempfile
import unittest
from pathlib import Path

from maniskill_codex.run_seed_batch import (
    build_pipeline_command,
    collect_run_record,
    parse_args,
    parse_seed_range,
    write_summary_files,
)


class SeedBatchTests(unittest.TestCase):
    def test_parse_seed_range_is_inclusive(self):
        self.assertEqual(parse_seed_range("1-3"), [1, 2, 3])
        self.assertEqual(parse_seed_range("7"), [7])

    def test_parse_args_defaults_to_picksingle_flip_world_z(self):
        args = parse_args(["--seed-range", "1-2", "--output-root", "runs"])

        self.assertEqual(args.seed_range, "1-2")
        self.assertEqual(args.output_root, "runs")
        self.assertEqual(args.env_id, "PickSingleYCB-v1")
        self.assertEqual(args.mask_mode, "task-target")
        self.assertEqual(args.approach_axis, "flip-world-z")
        self.assertEqual(args.camera_eye, [-0.30, 0.0, 0.55])
        self.assertEqual(args.camera_target, [0.05, 0.0, 0.08])
        self.assertEqual(args.workspace_z_min, 0.02)
        self.assertEqual(args.pregrasp_max_steps, 200)
        self.assertEqual(args.stage_max_steps, 80)
        self.assertEqual(args.settle_pos_tolerance, 0.01)
        self.assertEqual(args.descend_settle_pos_tolerance, 0.02)

    def test_build_pipeline_command_uses_seed_output_root_and_flip_z(self):
        args = parse_args(
            [
                "--seed-range",
                "5",
                "--output-root",
                "/tmp/batch",
                "--batch-name",
                "check",
                "--no-conda",
            ]
        )

        command = build_pipeline_command(args, seed=5, run_name="check_seed5")

        self.assertEqual(command[:3], ["python", "-m", "maniskill_codex.run_full_pipeline"])
        self.assertIn("--output-root", command)
        self.assertIn("/tmp/batch", command)
        self.assertIn("--run-name", command)
        self.assertIn("check_seed5", command)
        self.assertIn("--seed", command)
        self.assertIn("5", command)
        self.assertIn("--approach-axis", command)
        self.assertIn("flip-world-z", command)
        self.assertIn("--camera-eye", command)
        self.assertIn("-0.3", command)
        self.assertIn("--camera-target", command)
        self.assertIn("--stage-max-steps", command)
        self.assertIn("80", command)
        self.assertIn("--pregrasp-max-steps", command)
        self.assertIn("200", command)
        self.assertIn("--settle-pos-tolerance", command)
        self.assertIn("0.01", command)
        self.assertIn("--descend-settle-pos-tolerance", command)
        self.assertIn("0.02", command)
        self.assertIn("--workspace-z-min", command)
        self.assertIn("0.02", command)
        self.assertIn("--no-conda", command)

    def test_collect_run_record_and_write_summary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed1"
            (run_dir / "zg_input").mkdir(parents=True)
            (run_dir / "zg_output").mkdir()
            (run_dir / "logs").mkdir()
            (run_dir / "zg_input" / "scene.json").write_text(
                json.dumps({"objects": [{"display_name": "banana"}]}),
                encoding="utf-8",
            )
            (run_dir / "zg_output" / "recommended_grasp_top1.json").write_text(
                json.dumps({"score": 0.4, "width_m": 0.05}),
                encoding="utf-8",
            )
            (run_dir / "logs" / "execute.stdout.log").write_text(
                "approach_axis_base=[0.1 0.2 -0.9]\n"
                "  final_result={'success': True, 'stage': 'close', 'truncated': False}\n"
                "Summary: 1/1 success\n",
                encoding="utf-8",
            )

            record = collect_run_record(seed=1, run_name="seed1", run_dir=run_dir, exit_code=0)
            tsv_path, json_path = write_summary_files([record], Path(tmp), "batch")

            self.assertTrue(record["success"])
            self.assertEqual(record["object"], "banana")
            self.assertEqual(record["stage"], "close")
            self.assertEqual(record["approach_z"], "-0.9")
            self.assertIn("banana", tsv_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["success_count"], 1)


if __name__ == "__main__":
    unittest.main()
