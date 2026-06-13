"""Tests for ZeroGrasp reconstruction persistence and coordinate conversion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from maniskill_curobo_real.zerograsp_reconstruction import (
    load_zerograsp_reconstructed_instances,
    opencv_camera_points_mm_to_base,
)


class ZeroGraspReconstructionTest(unittest.TestCase):
    def test_opencv_points_are_converted_from_mm_to_base_meters(self) -> None:
        points = opencv_camera_points_mm_to_base(
            np.asarray([[100.0, 200.0, 300.0]]),
            camera_model_matrix=np.eye(4),
            world_from_base_matrix=np.eye(4),
        )
        np.testing.assert_allclose(points, [[0.1, -0.2, -0.3]])

    def test_load_instances_joins_camera_object_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            raw_dir = output_dir / "raw_outputs"
            input_dir.mkdir()
            raw_dir.mkdir(parents=True)
            (input_dir / "camera.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "label": 3,
                                "segmentation_id": 17,
                                "actor_name": "cracker_box",
                                "is_task_target": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                raw_dir / "object_000_label_3.reconstruction.npz",
                points_mm=np.asarray([[100.0, 0.0, 200.0]], dtype=np.float32),
                normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            )
            (output_dir / "run_report.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "object_id": 3,
                                "reconstruction_file": (
                                    "raw_outputs/object_000_label_3.reconstruction.npz"
                                ),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            instances = load_zerograsp_reconstructed_instances(
                input_dir=input_dir,
                output_dir=output_dir,
                camera_model_matrix=np.eye(4),
                world_from_base_matrix=np.eye(4),
            )

            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].segmentation_id, 17)
            self.assertEqual(instances[0].actor_name, "cracker_box")
            self.assertTrue(instances[0].is_task_target)
            np.testing.assert_allclose(instances[0].points_base, [[0.1, 0.0, -0.2]])


if __name__ == "__main__":
    unittest.main()
