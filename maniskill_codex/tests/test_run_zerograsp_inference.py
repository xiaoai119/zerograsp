import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from maniskill_codex.run_zerograsp_inference import (
    grasp_prediction_to_json,
    save_zerograsp_result,
)


class FakeGrasp:
    def __init__(self):
        self.score = 0.7
        self.width = 0.04
        self.height = 0.02
        self.depth = 0.03
        self.rotation_matrix = np.eye(3)
        self.translation = np.array([0.1, 0.2, 0.3])
        self.object_id = 5


class FakeObjectResult:
    object_id = 5
    object_index = 0
    n_grasps_before_collision = 3
    n_grasps_after_collision = 2
    n_grasps_final = 1
    grasp_group_array = np.array(
        [[0.7, 0.04, 0.02, 0.03, *np.eye(3).reshape(-1), 0.1, 0.2, 0.3, 5.0]],
        dtype=np.float64,
    )


class FakeResult:
    runtime_sec = 1.25
    objects = [FakeObjectResult()]

    def recommended_grasp(self):
        return FakeGrasp()


class RunZeroGraspInferenceTests(unittest.TestCase):
    def test_grasp_prediction_to_json_uses_loader_field_names(self):
        data = grasp_prediction_to_json(FakeGrasp(), source_file="raw_outputs/object_000_label_5.grasp.npy")

        self.assertEqual(data["score"], 0.7)
        self.assertEqual(data["width_m"], 0.04)
        self.assertEqual(data["height_m"], 0.02)
        self.assertEqual(data["depth_m"], 0.03)
        self.assertEqual(data["translation_m_camera"], [0.1, 0.2, 0.3])
        self.assertEqual(data["object_id"], 5)
        self.assertEqual(data["source_file"], "raw_outputs/object_000_label_5.grasp.npy")

    def test_save_zerograsp_result_writes_raw_outputs_and_recommended_grasp(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            report = save_zerograsp_result(FakeResult(), output_dir)

            raw_file = output_dir / "raw_outputs" / "object_000_label_5.grasp.npy"
            self.assertTrue(raw_file.is_file())
            self.assertTrue((output_dir / "recommended_grasp_top1.json").is_file())
            self.assertTrue((output_dir / "run_report.json").is_file())
            self.assertEqual(report["n_objects"], 1)
            self.assertEqual(report["recommended_grasp"]["object_id"], 5)
            loaded = json.loads((output_dir / "recommended_grasp_top1.json").read_text())
            self.assertEqual(loaded["object_id"], 5)


if __name__ == "__main__":
    unittest.main()
