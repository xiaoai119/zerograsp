import unittest

import numpy as np

from maniskill_codex.grasp_markers import (
    GraspMarkerGeometry,
    build_grasp_marker_geometry,
    opencv_grasp_rotation_to_world_axes,
)
from maniskill_codex.zerograsp_outputs import GraspRecord


class GraspMarkerTests(unittest.TestCase):
    def test_opencv_grasp_rotation_to_world_axes_applies_camera_transform(self):
        camera_model = np.eye(4)
        camera_model[:3, :3] = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        axes = opencv_grasp_rotation_to_world_axes(np.eye(3), camera_model)

        np.testing.assert_allclose(axes[:, 0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(axes[:, 1], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(axes[:, 2], [0.0, 0.0, -1.0])

    def test_build_grasp_marker_geometry_uses_world_center_and_width_axis(self):
        grasp = GraspRecord(
            score=0.5,
            width_m=0.04,
            height_m=0.02,
            depth_m=0.03,
            rotation_matrix_camera=np.eye(3),
            translation_m_camera=np.array([0.1, 0.2, 0.3]),
            source="test",
            object_id=7,
        )
        camera_model = np.eye(4)
        camera_model[:3, 3] = [1.0, 2.0, 3.0]

        geom = build_grasp_marker_geometry(grasp, camera_model, approach_length=0.08)

        self.assertIsInstance(geom, GraspMarkerGeometry)
        np.testing.assert_allclose(geom.center_world, [1.1, 1.8, 2.7])
        np.testing.assert_allclose(geom.approach_axis_world, [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(geom.width_axis_world, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(geom.approach_end_world, [1.02, 1.8, 2.7])
        np.testing.assert_allclose(geom.width_endpoints_world[0], [1.1, 1.82, 2.7])
        np.testing.assert_allclose(geom.width_endpoints_world[1], [1.1, 1.78, 2.7])
        self.assertEqual(geom.object_id, 7)

    def test_build_grasp_marker_geometry_can_draw_positive_x_approach(self):
        grasp = GraspRecord(
            score=0.5,
            width_m=0.04,
            height_m=0.02,
            depth_m=0.03,
            rotation_matrix_camera=np.eye(3),
            translation_m_camera=np.array([0.1, 0.2, 0.3]),
            source="test",
            object_id=7,
        )

        geom = build_grasp_marker_geometry(
            grasp,
            np.eye(4),
            approach_length=0.08,
            approach_axis="positive-x",
        )

        np.testing.assert_allclose(geom.approach_axis_world, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(geom.approach_end_world, [0.18, -0.2, -0.3])

    def test_build_grasp_marker_geometry_can_override_center_for_execution_target(self):
        grasp = GraspRecord(
            score=0.5,
            width_m=0.04,
            height_m=0.02,
            depth_m=0.03,
            rotation_matrix_camera=np.eye(3),
            translation_m_camera=np.array([0.1, 0.2, 0.3]),
            source="test",
            object_id=7,
        )

        geom = build_grasp_marker_geometry(
            grasp,
            np.eye(4),
            approach_length=0.08,
            center_world=np.array([1.0, 2.0, 3.0]),
        )

        np.testing.assert_allclose(geom.center_world, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(geom.approach_end_world, [0.92, 2.0, 3.0])

    def test_build_grasp_marker_geometry_can_flip_only_world_z(self):
        grasp = GraspRecord(
            score=0.5,
            width_m=0.04,
            height_m=0.02,
            depth_m=0.03,
            rotation_matrix_camera=np.eye(3),
            translation_m_camera=np.array([0.1, 0.2, 0.3]),
            source="test",
            object_id=7,
        )
        camera_model = np.eye(4)
        camera_model[:3, :3] = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        negative_x_geom = build_grasp_marker_geometry(
            grasp,
            camera_model,
            approach_axis="negative-x",
        )
        flipped_geom = build_grasp_marker_geometry(
            grasp,
            camera_model,
            approach_axis="flip-world-z",
        )

        expected = negative_x_geom.approach_axis_world.copy()
        expected[2] *= -1.0
        np.testing.assert_allclose(flipped_geom.approach_axis_world, expected)


if __name__ == "__main__":
    unittest.main()
