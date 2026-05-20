import unittest

import numpy as np

from maniskill_codex.transforms import opencv_camera_to_base, opencv_camera_to_sapien_camera


class TransformTests(unittest.TestCase):
    def test_opencv_to_sapien_flips_y_and_z(self):
        np.testing.assert_allclose(
            opencv_camera_to_sapien_camera(np.array([1.0, 2.0, 3.0])),
            np.array([1.0, -2.0, -3.0]),
        )

    def test_opencv_camera_to_base_applies_camera_and_base_matrices(self):
        camera_model = np.eye(4)
        camera_model[:3, 3] = [10.0, 20.0, 30.0]
        world_from_base = np.eye(4)
        world_from_base[:3, 3] = [1.0, 2.0, 3.0]

        result = opencv_camera_to_base(
            np.array([0.5, 0.25, 0.75]),
            camera_model,
            world_from_base,
        )

        np.testing.assert_allclose(result, [9.5, 17.75, 26.25])


if __name__ == "__main__":
    unittest.main()
