from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from maniskill_codex.zerograsp_outputs import load_best_grasp
from maniskill_curobo_graspnet.graspnet_adapter import (
    filter_grasps_by_target_limits,
    grasp_to_standard_json,
    load_point_cloud_input,
    resolve_depth_scale_to_m,
)


class FakeGrasp:
    score = 0.9
    width = 0.06
    height = 0.02
    depth = 0.03
    rotation_matrix = np.eye(3)
    translation = np.array([0.1, -0.2, 0.5])


class FakeGraspGroup:
    def __init__(self, translations: np.ndarray):
        self.translations = translations
        self.last_mask = None

    def __len__(self) -> int:
        return len(self.translations)

    def __getitem__(self, mask):
        self.last_mask = mask
        return FakeGraspGroup(self.translations[mask])


class GraspNetAdapterTest(unittest.TestCase):
    def test_auto_depth_unit(self) -> None:
        self.assertEqual(resolve_depth_scale_to_m(np.array([500.0]), "auto"), 0.001)
        self.assertEqual(resolve_depth_scale_to_m(np.array([0.5]), "auto"), 1.0)

    def test_loads_target_and_context_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rgb = np.full((4, 5, 3), 128, dtype=np.uint8)
            depth = np.full((4, 5), 500.0, dtype=np.float32)
            mask = np.zeros((4, 5), dtype=np.uint8)
            mask[1:3, 1:4] = 1
            intrinsic = np.array(
                [[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            np.savez_compressed(
                root / "rgbd.npz",
                rgb=rgb,
                depth=depth,
                mask=mask,
                cam_K=intrinsic,
            )
            (root / "camera.json").write_text(
                json.dumps({"cam_K": intrinsic.ravel().tolist()}),
                encoding="utf-8",
            )
            prepared = load_point_cloud_input(
                root,
                min_target_points=4,
                target_margin_m=0.0,
                context_margin_m=1.0,
            )
            self.assertEqual(prepared.target_label, 1)
            self.assertEqual(prepared.target_point_count, 6)
            self.assertEqual(prepared.context_point_count, 20)
            self.assertAlmostEqual(prepared.depth_scale_to_m, 0.001)
            self.assertTrue(np.allclose(prepared.points[:, 2], 0.5))

    def test_filter_grasps_by_target_limits(self) -> None:
        group = FakeGraspGroup(
            np.array([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5], [0.0, 0.3, 0.5]])
        )
        filtered = filter_grasps_by_target_limits(
            group,
            [-0.1, 0.1, -0.1, 0.1, 0.4, 0.6],
        )
        self.assertEqual(len(filtered), 1)

    def test_grasp_json_matches_shared_loader_schema(self) -> None:
        result = grasp_to_standard_json(FakeGrasp(), object_id=1)
        self.assertEqual(result["model"], "graspnet-baseline")
        self.assertEqual(result["object_id"], 1)
        self.assertAlmostEqual(result["depth_m"], 0.03)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "recommended_grasp_top1.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            loaded = load_best_grasp(output_dir)
            self.assertAlmostEqual(loaded.score, FakeGrasp.score)
            self.assertAlmostEqual(loaded.depth_m, FakeGrasp.depth)
            self.assertEqual(loaded.object_id, 1)
            self.assertTrue(
                np.allclose(loaded.translation_m_camera, FakeGrasp.translation)
            )


if __name__ == "__main__":
    unittest.main()
