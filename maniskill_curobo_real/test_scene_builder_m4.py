"""Small deterministic geometry tests for the M4 scene builders."""

from __future__ import annotations

import math
import unittest

import numpy as np

from maniskill_curobo_real.scene_builder import (
    _voxelize_closed_instance,
    points_to_aabb_cuboid,
    points_to_convex_hull_mesh,
    points_to_yaw_obb_cuboid,
)


class M4GeometryTest(unittest.TestCase):
    @staticmethod
    def rotated_box_points() -> np.ndarray:
        rng = np.random.default_rng(7)
        local = rng.uniform(
            [-0.10, -0.025, -0.04],
            [0.10, 0.025, 0.04],
            size=(3000, 3),
        )
        yaw = math.radians(37.0)
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        )
        points = local.copy()
        points[:, :2] = local[:, :2] @ rotation.T + np.asarray([0.55, 0.08])
        points[:, 2] += 0.10
        return points

    def test_yaw_obb_is_tighter_than_aabb(self) -> None:
        points = self.rotated_box_points()
        aabb = points_to_aabb_cuboid(points, padding=0.0, min_dimension=0.005)
        obb = points_to_yaw_obb_cuboid(points, padding=0.0, min_dimension=0.005)
        self.assertIsNotNone(aabb)
        self.assertIsNotNone(obb)
        assert aabb is not None and obb is not None
        self.assertLess(np.prod(obb["dims"]), 0.7 * np.prod(aabb["dims"]))
        self.assertAlmostEqual(abs(math.degrees(obb["yaw_rad"])), 37.0, delta=3.0)

    def test_convex_hull_mesh_is_nonempty(self) -> None:
        mesh = points_to_convex_hull_mesh(self.rotated_box_points(), max_points=1500)
        self.assertIsNotNone(mesh)
        assert mesh is not None
        self.assertGreaterEqual(len(mesh["vertices"]), 4)
        self.assertGreaterEqual(len(mesh["faces"]) // 3, 4)

    def test_closed_instance_voxelization_fills_volume(self) -> None:
        points = self.rotated_box_points()[::3]
        axes = [
            np.linspace(0.35, 0.75, 41),
            np.linspace(-0.15, 0.25, 41),
            np.linspace(0.01, 0.21, 21),
        ]
        occupied = np.zeros((41, 41, 21), dtype=bool)
        added, used_fallback = _voxelize_closed_instance(
            occupied=occupied,
            axes=axes,
            points=points,
            voxel_size=0.01,
        )
        self.assertGreater(added, 0)
        self.assertFalse(used_fallback)
        self.assertTrue(occupied[20, 23, 9])


if __name__ == "__main__":
    unittest.main()
