"""Tests for the reusable multi-view RGB-D capture geometry."""

from __future__ import annotations

import unittest

import numpy as np

from maniskill_curobo_real.capture_multiview_rgbd import default_three_view_specs


class MultiviewCaptureTest(unittest.TestCase):
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

        self.assertEqual(
            [view.name for view in views],
            [
                "view_0_existing_rgbd",
                "view_1_close_left",
                "view_2_close_right",
            ],
        )
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


if __name__ == "__main__":
    unittest.main()
