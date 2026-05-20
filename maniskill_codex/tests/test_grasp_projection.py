import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from maniskill_codex.grasp_projection import draw_grasp_projection, parse_args, project_3d_to_2d


class GraspProjectionTests(unittest.TestCase):
    def test_parse_args_accepts_approach_axis(self):
        args = parse_args(
            [
                "--rgb",
                "rgb.png",
                "--camera",
                "camera.json",
                "--grasp",
                "grasp.json",
                "--output",
                "overlay.png",
                "--approach-axis",
                "flip-world-z",
            ]
        )

        self.assertEqual(args.approach_axis, "flip-world-z")

    def test_project_3d_to_2d_uses_intrinsics(self):
        K = np.array([[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])

        self.assertEqual(project_3d_to_2d(np.array([0.2, -0.1, 2.0]), K), (60, 50))

    def test_project_3d_to_2d_rejects_points_behind_camera(self):
        K = np.eye(3)

        self.assertIsNone(project_3d_to_2d(np.array([0.0, 0.0, 0.0]), K))
        self.assertIsNone(project_3d_to_2d(np.array([0.0, 0.0, -1.0]), K))

    def test_draw_grasp_projection_writes_overlay_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "rgb.png"
            camera_path = root / "camera.json"
            grasp_path = root / "grasp.json"
            output_path = root / "overlay.png"

            Image.fromarray(np.full((100, 120, 3), 40, dtype=np.uint8)).save(rgb_path)
            camera_path.write_text(
                json.dumps({"cam_K": [100.0, 0.0, 60.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]}),
                encoding="utf-8",
            )
            grasp_path.write_text(
                json.dumps(
                    {
                        "score": 0.7,
                        "width_m": 0.04,
                        "height_m": 0.02,
                        "depth_m": 0.03,
                        "rotation_matrix_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "translation_m_camera": [0.0, 0.0, 1.0],
                        "object_id": 3,
                    }
                ),
                encoding="utf-8",
            )

            written = draw_grasp_projection(rgb_path, camera_path, grasp_path, output_path)

            self.assertEqual(written, output_path)
            overlay = np.asarray(Image.open(output_path))
            self.assertEqual(overlay.shape, (100, 120, 3))
            self.assertGreater(np.count_nonzero(overlay != 40), 0)


if __name__ == "__main__":
    unittest.main()
