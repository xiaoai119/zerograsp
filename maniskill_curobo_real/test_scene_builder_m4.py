"""Small deterministic geometry tests for the M4 scene builders."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from maniskill_curobo_real.scene_builder import (
    _voxelize_closed_instance,
    build_zerograsp_instance_cuboid_scene,
    build_zerograsp_instance_mesh_scene,
    build_zerograsp_instance_obb_scene,
    build_zerograsp_instance_voxel_esdf_scene,
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

    def test_zerograsp_builders_exclude_target_instance(self) -> None:
        obstacle_points = self.rotated_box_points()
        target_points = obstacle_points.copy()
        target_points[:, 1] += 0.25
        instances = [
            SimpleNamespace(
                label=1,
                segmentation_id=10,
                actor_name="obstacle",
                is_task_target=False,
                points_base=obstacle_points,
                reconstruction_file="obstacle.npz",
            ),
            SimpleNamespace(
                label=2,
                segmentation_id=11,
                actor_name="target",
                is_task_target=True,
                points_base=target_points,
                reconstruction_file="target.npz",
            ),
        ]

        cuboid = build_zerograsp_instance_cuboid_scene(
            instances=instances,
            min_instance_points=100,
        )
        obb = build_zerograsp_instance_obb_scene(
            instances=instances,
            min_instance_points=100,
        )
        mesh = build_zerograsp_instance_mesh_scene(
            instances=instances,
            min_instance_points=100,
        )
        voxel = build_zerograsp_instance_voxel_esdf_scene(
            instances=instances,
            min_instance_points=100,
            voxel_size=0.02,
        )

        for result in (cuboid, obb, mesh, voxel):
            self.assertEqual(result.metadata["n_pointcloud_obstacles"], 1)
            self.assertEqual(result.metadata["point_cloud_source"], "zerograsp_reconstruction")
            self.assertTrue(result.metadata["target_exclusion_applied"])
            self.assertFalse(result.metadata["uses_maniskill_depth_geometry"])
            skipped = result.metadata["skipped_instance_records"]
            self.assertTrue(
                any(record["reason"] == "target_excluded" for record in skipped)
            )


if __name__ == "__main__":
    unittest.main()
