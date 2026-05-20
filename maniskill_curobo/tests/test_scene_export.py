from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from maniskill_curobo.scene_export import (
    export_maniskill_scene_to_curobo,
    maniskill_scene_to_curobo_dict,
    target_segmentation_ids_from_zerograsp_scene,
)


class FakePose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


class FakeBoxShape:
    half_size = np.array([0.2, 0.1, 0.05], dtype=np.float64)
    local_pose = FakePose([0.1, 0.0, 0.05])


class FakeConvexShape:
    vertices = np.array(
        [
            [-0.1, -0.2, -0.05],
            [0.3, 0.2, 0.15],
        ],
        dtype=np.float64,
    )
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    local_pose = FakePose([0.0, 0.1, 0.0])


class FakePlaneShape:
    local_pose = FakePose([0.0, 0.0, -0.5])


class FakeComponent:
    def __init__(self, name, pose, shapes):
        self.name = name
        self.entity_pose = pose
        self._shapes = shapes

    def get_collision_shapes(self):
        return list(self._shapes)


class FakeActor:
    def __init__(self, name, segmentation_id, components):
        self.name = name
        self.per_scene_id = np.array([segmentation_id], dtype=np.int32)
        self.components = components


class FakeScene:
    def __init__(self, actors):
        self._actors = actors

    def get_all_actors(self):
        return list(self._actors)


class FakeRoot:
    def __init__(self, scene):
        self.scene = scene


class FakeEnv:
    def __init__(self, actors):
        self.unwrapped = FakeRoot(FakeScene(actors))


class SceneExportTest(unittest.TestCase):
    def test_exports_box_shapes_as_base_frame_cuboids(self) -> None:
        env = FakeEnv(
            [
                FakeActor(
                    "scene-0_table-workspace",
                    16,
                    [FakeComponent("table", FakePose([1.0, 0.0, 0.0]), [FakeBoxShape()])],
                )
            ]
        )
        world_from_base = np.eye(4)
        world_from_base[:3, 3] = [0.5, 0.0, 0.0]

        scene, records = maniskill_scene_to_curobo_dict(env, world_from_base)

        self.assertEqual(len(scene["cuboid"]), 1)
        obstacle = next(iter(scene["cuboid"].values()))
        self.assertEqual(obstacle["dims"], [0.4, 0.2, 0.1])
        np.testing.assert_allclose(obstacle["pose"][:3], [0.6, 0.0, 0.05])
        self.assertEqual(records[0]["shape_type"], "box")

    def test_exports_convex_shapes_as_local_bounding_cuboids(self) -> None:
        env = FakeEnv(
            [
                FakeActor(
                    "scene-0_set_0_object",
                    18,
                    [FakeComponent("object", FakePose([0.0, 0.0, 0.0]), [FakeConvexShape()])],
                )
            ]
        )

        scene, records = maniskill_scene_to_curobo_dict(env, np.eye(4))

        obstacle = next(iter(scene["cuboid"].values()))
        self.assertEqual(obstacle["dims"], [0.4, 0.4, 0.2])
        np.testing.assert_allclose(obstacle["pose"][:3], [0.1, 0.1, 0.05])
        self.assertEqual(records[0]["shape_type"], "convex_mesh")

    def test_skips_goal_plane_and_excluded_target_actor(self) -> None:
        env = FakeEnv(
            [
                FakeActor("scene-0_goal_site", 23, [FakeComponent("goal", FakePose([0, 0, 0]), [FakeBoxShape()])]),
                FakeActor("scene-0_ground", 17, [FakeComponent("ground", FakePose([0, 0, 0]), [FakePlaneShape()])]),
                FakeActor("scene-0_target", 19, [FakeComponent("target", FakePose([0, 0, 0]), [FakeBoxShape()])]),
                FakeActor("scene-0_clutter", 20, [FakeComponent("clutter", FakePose([0, 0, 0]), [FakeBoxShape()])]),
            ]
        )

        scene, records = maniskill_scene_to_curobo_dict(
            env,
            np.eye(4),
            exclude_segmentation_ids={19},
        )

        self.assertEqual(list(scene["cuboid"].keys()), ["scene_0_clutter_box_00"])
        self.assertEqual(records[0]["actor_name"], "scene-0_clutter")

    def test_export_writes_yaml_and_metadata(self) -> None:
        env = FakeEnv(
            [
                FakeActor(
                    "scene-0_table-workspace",
                    16,
                    [FakeComponent("table", FakePose([0.0, 0.0, 0.0]), [FakeBoxShape()])],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "curobo_scene.yml"
            metadata_path = Path(tmp) / "curobo_scene_metadata.json"

            export = export_maniskill_scene_to_curobo(
                env,
                scene_path,
                metadata_path=metadata_path,
                world_from_base_matrix=np.eye(4),
            )

            self.assertEqual(export.scene_path, scene_path.resolve())
            self.assertEqual(export.metadata_path, metadata_path.resolve())
            self.assertTrue(scene_path.is_file())
            self.assertTrue(metadata_path.is_file())
            loaded_scene = json.loads(scene_path.read_text(encoding="utf-8"))
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded_scene["cuboid"]), 1)
            self.assertEqual(loaded_metadata["n_obstacles"], 1)

    def test_maps_zerograsp_label_to_maniskill_segmentation_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = Path(tmp) / "scene.json"
            scene_json.write_text(
                json.dumps(
                    {
                        "objects": [
                            {"label": 1, "segmentation_id": 18, "actor_name": "target"},
                            {"label": 2, "segmentation_id": 20, "actor_name": "clutter"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                target_segmentation_ids_from_zerograsp_scene(scene_json, object_id=2),
                {20},
            )

    def test_falls_back_to_task_target_segmentation_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = Path(tmp) / "scene.json"
            scene_json.write_text(
                json.dumps(
                    {
                        "objects": [
                            {"label": 1, "segmentation_id": 18, "is_task_target": True},
                            {"label": 2, "segmentation_id": 20, "is_task_target": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                target_segmentation_ids_from_zerograsp_scene(scene_json, object_id=None),
                {18},
            )


if __name__ == "__main__":
    unittest.main()
