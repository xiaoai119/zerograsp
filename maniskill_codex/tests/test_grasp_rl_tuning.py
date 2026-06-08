import unittest

import numpy as np

from maniskill_codex.grasp_rl_tuning import (
    RLRolloutEvaluation,
    RLTuningConfig,
    ResidualActionBounds,
    ResidualGraspAction,
    apply_residual_to_pose,
    reward_from_rollout_result,
    tune_residual_policy,
)


class GraspRLTuningTests(unittest.TestCase):
    def test_apply_residual_to_pose_offsets_in_tcp_frame_and_rolls_about_approach(self):
        action = ResidualGraspAction(
            approach_offset_m=0.03,
            tcp_x_offset_m=0.01,
            tcp_y_offset_m=-0.02,
            roll_delta_rad=np.pi / 2,
        )

        position, rotation = apply_residual_to_pose(
            np.array([0.5, 0.0, 0.2]),
            np.eye(3),
            action,
        )

        np.testing.assert_allclose(position, [0.51, -0.02, 0.23])
        np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-7)
        self.assertGreater(np.linalg.det(rotation), 0.99)

    def test_reward_prefers_success_and_object_lift(self):
        self.assertGreater(
            reward_from_rollout_result({"success": True, "object_lift_delta_m": 0.04}),
            reward_from_rollout_result({"success": False, "stage": "close"}),
        )
        self.assertGreater(
            reward_from_rollout_result({"success": False, "stage": "lift", "object_lift_delta_m": 0.05}),
            reward_from_rollout_result({"success": False, "stage": "pre", "not_converged": True}),
        )

    def test_cem_returns_best_evaluated_residual_action(self):
        bounds = ResidualActionBounds.from_ranges(
            approach_offset_range=(0.0, 0.05),
            tcp_x_offset_range=(0.0, 0.0),
            tcp_y_offset_range=(0.0, 0.0),
            roll_delta_range=(0.0, 0.0),
            gripper_open_range=(1.0, 1.0),
            gripper_closed_range=(-1.0, -1.0),
        )
        config = RLTuningConfig(
            iterations=2,
            population=4,
            elite_fraction=0.5,
            seed=7,
            bounds=bounds,
            stop_on_success=False,
        )
        seen = []

        def evaluate(action, iteration, candidate):
            seen.append((iteration, candidate, action.approach_offset_m))
            reward = action.approach_offset_m
            return RLRolloutEvaluation(
                reward=reward,
                success=reward > 0.045,
                result={"success": reward > 0.045, "stage": "lift"},
            )

        trace = tune_residual_policy(config, evaluate)

        self.assertEqual(len(seen), 8)
        self.assertEqual(trace["iterations_completed"], 2)
        self.assertAlmostEqual(
            trace["best_reward"],
            trace["best_action"]["approach_offset_m"],
        )
        self.assertGreaterEqual(trace["best_action"]["approach_offset_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
