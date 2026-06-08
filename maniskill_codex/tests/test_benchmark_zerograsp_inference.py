from __future__ import annotations

import unittest

from maniskill_codex.benchmark_zerograsp_inference import parse_args, timing_summary


class BenchmarkZeroGraspInferenceTest(unittest.TestCase):
    def test_parse_args_defaults_to_one_warmup_and_five_measured_runs(self) -> None:
        args = parse_args(["--input-dir", "input"])

        self.assertEqual(args.warmup_runs, 1)
        self.assertEqual(args.benchmark_runs, 5)
        self.assertFalse(args.enable_collision_detection)

    def test_timing_summary_reports_distribution(self) -> None:
        summary = timing_summary([3.0, 1.0, 2.0])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["values_sec"], [1.0, 2.0, 3.0])
        self.assertEqual(summary["mean_sec"], 2.0)
        self.assertEqual(summary["median_sec"], 2.0)
        self.assertEqual(summary["p90_sec"], 3.0)


if __name__ == "__main__":
    unittest.main()
