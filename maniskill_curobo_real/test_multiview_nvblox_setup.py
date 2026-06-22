"""Tests for multiview RGB-D capture geometry and nvblox setup helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from maniskill_curobo_real.capture_multiview_rgbd import default_three_view_specs
from maniskill_curobo_real.nvblox_fusion import (
    check_nvblox_availability,
    inspect_input_root,
)


class MultiviewNvbloxSetupTest(unittest.TestCase):
    def test_default_three_views_keep_existing_camera_and_close_aux_views(self) -> None:
        main_eye = (-0.20, 0.0, 0.27)
        target = (0.05, 0.0, 0.08)
        views = default_three_view_specs(
            main_eye=main_eye,
            main_target=target,
            side_yaw_deg=30.0,
            side_distance=0.25,
            side_height_above_target=0.19,
        )

        self.assertEqual([view.name for view in views], [
            "view_0_existing_rgbd",
            "view_1_close_left",
            "view_2_close_right",
        ])
        self.assertEqual(views[0].eye, main_eye)
        self.assertEqual(views[0].target, target)
        self.assertEqual(views[0].role, "existing_baseline_camera")

        target_xy = np.asarray(target[:2], dtype=np.float64)
        for view in views[1:]:
            eye = np.asarray(view.eye, dtype=np.float64)
            self.assertAlmostEqual(float(np.linalg.norm(eye[:2] - target_xy)), 0.25)
            self.assertAlmostEqual(float(eye[2]), target[2] + 0.19)
            self.assertEqual(view.target, target)
        self.assertAlmostEqual(views[1].eye[0], views[2].eye[0])
        self.assertAlmostEqual(views[1].eye[1], -views[2].eye[1])

    def test_inspect_input_root_counts_seed_view_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_dir = root / "seed001"
            for name in ["view_0_existing_rgbd", "view_1_close_left", "view_2_close_right"]:
                view_dir = seed_dir / name
                view_dir.mkdir(parents=True)
                (view_dir / "view_metadata.json").write_text("{}\n", encoding="utf-8")
            (seed_dir / "multiview_manifest.json").write_text(
                json.dumps({"seed": 1}) + "\n",
                encoding="utf-8",
            )

            status = inspect_input_root(root, seed=1)
            self.assertTrue(status["exists"])
            self.assertEqual(status["n_seed_dirs"], 1)
            self.assertEqual(status["records"][0]["n_view_metadata"], 3)

    def test_nvblox_availability_probe_is_well_formed(self) -> None:
        status = check_nvblox_availability()
        self.assertIn("nvblox_torch", status.checked_modules)
        self.assertIn("nvblox", status.checked_modules)
        self.assertEqual(status.available, status.module_name is not None)


if __name__ == "__main__":
    unittest.main()
