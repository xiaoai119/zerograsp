import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from maniskill_codex.zerograsp_outputs import GraspRecord, load_best_grasp


class ZeroGraspOutputTests(unittest.TestCase):
    def test_loads_recommended_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "recommended_grasp_top1.json").write_text(
                json.dumps(
                    {
                        "score": 0.9,
                        "width_m": 0.04,
                        "height_m": 0.02,
                        "depth_m": 0.03,
                        "rotation_matrix_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "translation_m_camera": [0.1, 0.2, 0.3],
                        "source_file": "000_0.grasp.npy",
                        "object_index": 0,
                    }
                ),
                encoding="utf-8",
            )

            record = load_best_grasp(root)

            self.assertIsInstance(record, GraspRecord)
            self.assertEqual(record.source, "recommended_grasp_top1.json")
            self.assertEqual(record.score, 0.9)
            np.testing.assert_allclose(record.translation_m_camera, [0.1, 0.2, 0.3])

    def test_selects_highest_score_from_raw_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw_outputs"
            raw.mkdir()
            np.save(
                raw / "000_0.grasp.npy",
                np.array(
                    [[0.1, 0.04, 0.02, 0.03, *np.eye(3).reshape(-1), 0.1, 0.2, 0.3, -1]],
                    dtype=np.float64,
                ),
            )
            np.save(
                raw / "000_1.grasp.npy",
                np.array(
                    [[0.8, 0.05, 0.02, 0.04, *np.eye(3).reshape(-1), 0.4, 0.5, 0.6, -1]],
                    dtype=np.float64,
                ),
            )

            record = load_best_grasp(root)

            self.assertEqual(record.score, 0.8)
            self.assertEqual(record.source, "raw_outputs/000_1.grasp.npy")
            np.testing.assert_allclose(record.translation_m_camera, [0.4, 0.5, 0.6])

    def test_raises_when_no_grasp_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "recommended_grasp_top1.json"):
                load_best_grasp(Path(tmp))


if __name__ == "__main__":
    unittest.main()
