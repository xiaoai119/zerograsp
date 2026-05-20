from __future__ import annotations

import unittest

import numpy as np

from maniskill_curobo.joint_trajectory_utils import (
    make_pd_joint_pos_actions,
    ordered_values,
    sample_waypoints,
    squeeze_trajectory_positions,
)


class JointTrajectoryUtilsTest(unittest.TestCase):
    def test_ordered_values_maps_by_name(self) -> None:
        values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        ordered = ordered_values(
            source_names=["finger", "joint2", "joint1"],
            values=values,
            target_names=["joint1", "joint2"],
        )

        np.testing.assert_allclose(ordered, np.array([0.3, 0.2], dtype=np.float32))

    def test_ordered_values_raises_for_missing_name(self) -> None:
        with self.assertRaisesRegex(KeyError, "joint3"):
            ordered_values(
                source_names=["joint1", "joint2"],
                values=np.array([1.0, 2.0]),
                target_names=["joint3"],
            )

    def test_squeeze_trajectory_positions_accepts_curobo_batch_shape(self) -> None:
        trajectory = np.arange(1 * 1 * 3 * 2, dtype=np.float32).reshape(1, 1, 3, 2)

        squeezed = squeeze_trajectory_positions(trajectory)

        self.assertEqual(squeezed.shape, (3, 2))
        np.testing.assert_allclose(squeezed, np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32))

    def test_squeeze_trajectory_positions_keeps_time_joint_shape(self) -> None:
        trajectory = np.arange(6, dtype=np.float32).reshape(3, 2)

        squeezed = squeeze_trajectory_positions(trajectory)

        self.assertEqual(squeezed.shape, (3, 2))
        np.testing.assert_allclose(squeezed, trajectory)

    def test_make_pd_joint_pos_actions_reorders_arm_and_appends_gripper(self) -> None:
        trajectory = np.array(
            [
                [1.0, 2.0, 3.0, 0.04, 0.04],
                [4.0, 5.0, 6.0, 0.04, 0.04],
            ],
            dtype=np.float32,
        )

        actions = make_pd_joint_pos_actions(
            trajectory=trajectory,
            trajectory_joint_names=[
                "panda_joint2",
                "panda_joint1",
                "panda_joint3",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ],
            arm_action_joint_names=["panda_joint1", "panda_joint2", "panda_joint3"],
            gripper=1.0,
        )

        expected = np.array(
            [
                [2.0, 1.0, 3.0, 1.0],
                [5.0, 4.0, 6.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(actions.dtype, np.float32)
        np.testing.assert_allclose(actions, expected)

    def test_sample_waypoints_includes_first_and_last(self) -> None:
        waypoints = np.arange(10, dtype=np.float32).reshape(10, 1)

        sampled = sample_waypoints(waypoints, max_waypoints=4)

        np.testing.assert_allclose(sampled[:, 0], np.array([0, 3, 6, 9], dtype=np.float32))

    def test_sample_waypoints_does_not_duplicate_small_inputs(self) -> None:
        waypoints = np.arange(3, dtype=np.float32).reshape(3, 1)

        sampled = sample_waypoints(waypoints, max_waypoints=5)

        np.testing.assert_allclose(sampled, waypoints)


if __name__ == "__main__":
    unittest.main()
