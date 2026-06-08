from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from maniskill_curobo_graspnet.scripts.summarize_comparison import main


class SummarizeComparisonTest(unittest.TestCase):
    def test_pairwise_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graspnet_root = root / "graspnet"
            graspnet_root.mkdir()
            (graspnet_root / "execution_summary.json").write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "seed": 1,
                                "status": "complete",
                                "execution": {
                                    "object_lift_success": True,
                                    "outcome": "success",
                                },
                            },
                            {
                                "seed": 2,
                                "status": "complete",
                                "execution": {
                                    "object_lift_success": False,
                                    "outcome": "object_not_lifted",
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            zero_path = root / "zero.json"
            zero_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "seed": 1,
                                "variants": {
                                    "depth": {
                                        "object_lift_success": False,
                                        "outcome": "object_not_lifted",
                                    }
                                },
                            },
                            {
                                "seed": 2,
                                "variants": {
                                    "depth": {
                                        "object_lift_success": True,
                                        "outcome": "success",
                                    }
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "out"
            self.assertEqual(
                main(
                    [
                        "--graspnet-root",
                        str(graspnet_root),
                        "--zerograsp-summary",
                        str(zero_path),
                        "--output-dir",
                        str(output),
                        "--seed-start",
                        "1",
                        "--seed-end",
                        "2",
                    ]
                ),
                0,
            )
            summary = json.loads(
                (output / "comparison_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["counts"]["graspnet_only"], 1)
            self.assertEqual(summary["counts"]["zerograsp_only"], 1)


if __name__ == "__main__":
    unittest.main()
