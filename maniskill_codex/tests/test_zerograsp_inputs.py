import json
import types
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from maniskill_codex.zerograsp_inputs import (
    ZeroGraspInputBundle,
    build_instance_mask,
    collect_mask_actor_records,
    extract_zerograsp_input,
    save_zerograsp_input_bundle,
)


class ZeroGraspInputTests(unittest.TestCase):
    def test_build_instance_mask_filters_background_noise_and_table(self):
        segmentation = np.array(
            [
                [0, 10, 10, 20],
                [0, 10, 10, 20],
                [99, 99, 99, 99],
                [99, 99, 99, 99],
            ],
            dtype=np.int32,
        )

        mask, records = build_instance_mask(segmentation, min_pixels=3, max_pixels=6)

        expected = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(records, [{"label": 1, "segmentation_id": 10, "pixel_count": 4}])

    def test_build_instance_mask_uses_actor_records_as_object_whitelist(self):
        segmentation = np.array(
            [
                [16, 18, 18, 21],
                [99, 18, 18, 21],
            ],
            dtype=np.int32,
        )
        actor_records = [
            {"segmentation_id": 18, "actor_name": "set_0_065-j_cups", "is_task_target": False},
            {"segmentation_id": 21, "actor_name": "target_object", "is_task_target": True},
        ]

        mask, records = build_instance_mask(
            segmentation,
            min_pixels=0,
            max_pixels=3,
            actor_records=actor_records,
        )

        expected = np.array(
            [
                [0, 1, 1, 2],
                [0, 1, 1, 2],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(
            records,
            [
                {
                    "label": 1,
                    "segmentation_id": 18,
                    "pixel_count": 4,
                    "actor_name": "set_0_065-j_cups",
                    "is_task_target": False,
                },
                {
                    "label": 2,
                    "segmentation_id": 21,
                    "pixel_count": 2,
                    "actor_name": "target_object",
                    "is_task_target": True,
                },
            ],
        )

    def test_collect_mask_actor_records_supports_task_target_and_all_objects(self):
        root = types.SimpleNamespace()
        target = _fake_actor("target_object", 21)
        cup = _fake_actor("set_0_065-j_cups", 18)
        scene_target = _fake_actor("scene-0_set_0_051_large_clamp", 21)
        root.target_object = target
        root._hidden_objects = [_fake_actor("goal_site", 23)]
        root.scene = types.SimpleNamespace(
            get_all_actors=lambda: [
                _fake_actor("table", 16),
                cup,
                scene_target,
                _fake_actor("goal_site", 23),
                _fake_actor("ground", 17),
            ]
        )
        env = types.SimpleNamespace(unwrapped=root)

        target_records = collect_mask_actor_records(env, "task-target")
        all_object_records = collect_mask_actor_records(env, "all-objects")

        self.assertEqual([record["segmentation_id"] for record in target_records], [21])
        self.assertEqual(target_records[0]["actor_name"], "scene-0_set_0_051_large_clamp")
        self.assertEqual(target_records[0]["model_id"], "051_large_clamp")
        self.assertEqual(target_records[0]["category"], "large_clamp")
        self.assertEqual(target_records[0]["display_name"], "large clamp")
        self.assertTrue(target_records[0]["is_task_target"])
        self.assertEqual([record["segmentation_id"] for record in all_object_records], [18, 21])
        self.assertEqual(all_object_records[0]["actor_name"], "set_0_065-j_cups")
        self.assertEqual(all_object_records[0]["model_id"], "065-j_cups")
        self.assertEqual(all_object_records[0]["category"], "cups")
        self.assertTrue(all_object_records[1]["is_task_target"])

    def test_visible_area_mask_keeps_large_visible_segmentation_regions(self):
        color = np.zeros((1, 4, 4, 4), dtype=np.uint8)
        position_segmentation = np.zeros((1, 4, 4, 4), dtype=np.float32)
        position_segmentation[0, :, :, 2] = -0.5
        position_segmentation[0, :, :, 3] = 6
        obs = {
            "sensor_data": {
                "base_camera": {
                    "Color": color,
                    "PositionSegmentation": position_segmentation,
                }
            }
        }
        env = _fake_sensor_env("base_camera")

        bundle = extract_zerograsp_input(
            obs,
            env,
            "base_camera",
            min_pixels=0,
            max_pixels=2,
            mask_mode="visible-area",
        )

        expected = np.ones((4, 4), dtype=np.uint8)
        expected[0, 0] = 0
        np.testing.assert_array_equal(bundle.mask, expected)
        self.assertEqual(
            bundle.object_records,
            [{"label": 1, "segmentation_id": 6, "pixel_count": 15}],
        )

    def test_save_zerograsp_input_bundle_writes_images_and_camera_json(self):
        bundle = ZeroGraspInputBundle(
            rgb=np.full((3, 4, 3), 7, dtype=np.uint8),
            depth=np.full((3, 4), 123, dtype=np.float32),
            mask=np.ones((3, 4), dtype=np.uint8),
            camera_matrix=np.eye(3, dtype=np.float32),
            depth_scale=1.0,
            camera_name="base_camera",
            object_records=[{"label": 1, "segmentation_id": 10, "pixel_count": 12}],
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = save_zerograsp_input_bundle(bundle, Path(tmp))

            self.assertEqual(out, Path(tmp))
            self.assertTrue((out / "rgb.png").is_file())
            self.assertTrue((out / "depth.png").is_file())
            self.assertTrue((out / "mask.png").is_file())
            self.assertTrue((out / "camera.json").is_file())
            self.assertTrue((out / "rgbd.npz").is_file())

            with Image.open(out / "rgb.png") as rgb_img:
                self.assertEqual(rgb_img.mode, "RGB")
            with Image.open(out / "depth.png") as depth_img:
                self.assertEqual(np.asarray(depth_img).dtype, np.uint16)
            meta = json.loads((out / "camera.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["cam_K"], np.eye(3).reshape(-1).tolist())
            self.assertEqual(meta["depth_scale"], 1.0)
            self.assertEqual(meta["camera_name"], "base_camera")
            self.assertEqual(meta["objects"][0]["segmentation_id"], 10)


if __name__ == "__main__":
    unittest.main()


def _fake_actor(name: str, per_scene_id: int):
    return types.SimpleNamespace(name=name, per_scene_id=np.array([per_scene_id], dtype=np.int32))


def _fake_sensor_env(camera_name: str):
    camera = types.SimpleNamespace(
        get_intrinsic_matrix=lambda: np.eye(3, dtype=np.float32)[None, :, :]
    )
    sensor = types.SimpleNamespace(camera=camera)
    scene = types.SimpleNamespace(sensors={camera_name: sensor})
    root = types.SimpleNamespace(scene=scene)
    return types.SimpleNamespace(unwrapped=root)
