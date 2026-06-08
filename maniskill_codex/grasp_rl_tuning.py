"""Lightweight residual policy search for ManiSkill grasp execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


ACTION_NAMES = (
    "approach_offset_m",
    "tcp_x_offset_m",
    "tcp_y_offset_m",
    "roll_delta_rad",
    "gripper_open",
    "gripper_closed",
)


@dataclass(frozen=True)
class ResidualGraspAction:
    """Small executable correction applied after ZeroGrasp pose conversion."""

    approach_offset_m: float = 0.0
    tcp_x_offset_m: float = 0.0
    tcp_y_offset_m: float = 0.0
    roll_delta_rad: float = 0.0
    gripper_open: float = 1.0
    gripper_closed: float = -1.0

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "ResidualGraspAction":
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != len(ACTION_NAMES):
            raise ValueError(
                f"Residual action vector must contain {len(ACTION_NAMES)} values, got {values.size}."
            )
        return cls(**{name: float(value) for name, value in zip(ACTION_NAMES, values)})

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in ACTION_NAMES], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in ACTION_NAMES}


@dataclass(frozen=True)
class ResidualActionBounds:
    """Box constraints for residual grasp actions."""

    low: np.ndarray
    high: np.ndarray

    @classmethod
    def from_ranges(
        cls,
        *,
        approach_offset_range: tuple[float, float],
        tcp_x_offset_range: tuple[float, float],
        tcp_y_offset_range: tuple[float, float],
        roll_delta_range: tuple[float, float],
        gripper_open_range: tuple[float, float],
        gripper_closed_range: tuple[float, float],
    ) -> "ResidualActionBounds":
        low = np.array(
            [
                approach_offset_range[0],
                tcp_x_offset_range[0],
                tcp_y_offset_range[0],
                roll_delta_range[0],
                gripper_open_range[0],
                gripper_closed_range[0],
            ],
            dtype=np.float64,
        )
        high = np.array(
            [
                approach_offset_range[1],
                tcp_x_offset_range[1],
                tcp_y_offset_range[1],
                roll_delta_range[1],
                gripper_open_range[1],
                gripper_closed_range[1],
            ],
            dtype=np.float64,
        )
        bounds = cls(low=low, high=high)
        bounds.validate()
        return bounds

    def validate(self) -> None:
        low = np.asarray(self.low, dtype=np.float64).reshape(-1)
        high = np.asarray(self.high, dtype=np.float64).reshape(-1)
        if low.size != len(ACTION_NAMES) or high.size != len(ACTION_NAMES):
            raise ValueError("Residual action bounds must match the residual action dimension.")
        if np.any(high < low):
            raise ValueError("Each residual action upper bound must be >= its lower bound.")
        if low[4] < -1.0 or high[4] > 1.0 or low[5] < -1.0 or high[5] > 1.0:
            raise ValueError("Gripper command bounds must stay inside ManiSkill's [-1, 1] range.")
        if high[5] > low[4]:
            raise ValueError("The closed-gripper range should not overlap above the open-gripper range.")

    @property
    def span(self) -> np.ndarray:
        return self.high - self.low

    def clip(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(vector, dtype=np.float64), self.low, self.high)


@dataclass(frozen=True)
class RLTuningConfig:
    """Configuration for CEM residual policy search."""

    iterations: int
    population: int
    elite_fraction: float
    seed: int
    bounds: ResidualActionBounds
    initial_action: ResidualGraspAction = ResidualGraspAction()
    initial_std_fraction: float = 0.35
    min_std_fraction: float = 0.05
    include_baseline: bool = True
    stop_on_success: bool = True

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("RL tuning iterations must be >= 1.")
        if self.population < 1:
            raise ValueError("RL tuning population must be >= 1.")
        if not (0.0 < self.elite_fraction <= 1.0):
            raise ValueError("RL elite fraction must be in (0, 1].")
        if self.initial_std_fraction < 0.0:
            raise ValueError("RL initial std fraction must be non-negative.")
        if self.min_std_fraction < 0.0:
            raise ValueError("RL min std fraction must be non-negative.")
        self.bounds.validate()


@dataclass(frozen=True)
class RLRolloutEvaluation:
    """Reward and metadata from one residual action rollout."""

    reward: float
    success: bool
    result: dict[str, Any]


EvaluationFn = Callable[[ResidualGraspAction, int, int], RLRolloutEvaluation]


def tune_residual_policy(config: RLTuningConfig, evaluate: EvaluationFn) -> dict[str, Any]:
    """Optimize a residual grasp action with the cross-entropy method."""

    config.validate()
    rng = np.random.default_rng(int(config.seed))
    bounds = config.bounds
    baseline = bounds.clip(config.initial_action.to_vector())
    mean = baseline.copy()
    span = bounds.span
    min_std = np.maximum(span * float(config.min_std_fraction), 1e-9)
    std = np.maximum(span * float(config.initial_std_fraction), min_std)

    elite_count = max(1, int(np.ceil(float(config.population) * float(config.elite_fraction))))
    trials: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    completed_iterations = 0

    for iteration in range(int(config.iterations)):
        candidates = _sample_candidates(
            rng,
            mean=mean,
            std=std,
            bounds=bounds,
            population=int(config.population),
            baseline=baseline if config.include_baseline and iteration == 0 else None,
        )
        iteration_trials: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            action = ResidualGraspAction.from_vector(candidate)
            evaluation = evaluate(action, iteration, candidate_index)
            trial = {
                "iteration": int(iteration),
                "candidate": int(candidate_index),
                "reward": float(evaluation.reward),
                "success": bool(evaluation.success),
                "action": action.to_dict(),
                "result": _compact_result(evaluation.result),
            }
            trials.append(trial)
            iteration_trials.append(trial)
            if best_trial is None or float(trial["reward"]) > float(best_trial["reward"]):
                best_trial = trial

        ranked = sorted(iteration_trials, key=lambda item: float(item["reward"]), reverse=True)
        elites = np.stack(
            [
                ResidualGraspAction(**elite["action"]).to_vector()
                for elite in ranked[:elite_count]
            ],
            axis=0,
        )
        mean = bounds.clip(elites.mean(axis=0))
        std = np.maximum(elites.std(axis=0), min_std)
        completed_iterations = iteration + 1
        if config.stop_on_success and best_trial is not None and bool(best_trial["success"]):
            break

    if best_trial is None:
        raise RuntimeError("RL tuning did not evaluate any residual action.")

    return {
        "method": "cem_residual_policy",
        "action_names": list(ACTION_NAMES),
        "iterations_completed": int(completed_iterations),
        "best_reward": float(best_trial["reward"]),
        "best_success": bool(best_trial["success"]),
        "best_action": dict(best_trial["action"]),
        "final_mean": ResidualGraspAction.from_vector(mean).to_dict(),
        "final_std": {name: float(value) for name, value in zip(ACTION_NAMES, std)},
        "trials": trials,
    }


def apply_residual_to_pose(
    position_base: np.ndarray,
    rotation_base_tcp: np.ndarray,
    action: ResidualGraspAction,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply local-frame residual translation and approach-axis roll."""

    position = np.asarray(position_base, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_base_tcp, dtype=np.float64).reshape(3, 3)
    offset = (
        rotation[:, 2] * float(action.approach_offset_m)
        + rotation[:, 0] * float(action.tcp_x_offset_m)
        + rotation[:, 1] * float(action.tcp_y_offset_m)
    )
    adjusted_rotation = rotation @ _local_z_rotation(float(action.roll_delta_rad))
    return position + offset, _orthonormalize(adjusted_rotation)


def reward_from_rollout_result(result: dict[str, Any], lift_offset_m: float = 0.15) -> float:
    """Dense reward for grasp rollouts when the environment only exposes sparse success."""

    if bool(result.get("success")):
        return 1.0 + _positive_float(result.get("object_lift_delta_m"))

    stage_progress = {
        "pre": 0.0,
        "descend": 0.08,
        "close": 0.16,
        "lift": 0.28,
    }.get(str(result.get("stage", "")), 0.0)
    lift_scale = max(float(lift_offset_m), 1e-6)
    lift_reward = min(_positive_float(result.get("object_lift_delta_m")) / lift_scale, 1.0) * 0.5
    penalty = 0.0
    if bool(result.get("not_converged")):
        penalty += 0.05
    if bool(result.get("truncated")):
        penalty += 0.05
    return float(stage_progress + lift_reward - penalty)


def _sample_candidates(
    rng: np.random.Generator,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    bounds: ResidualActionBounds,
    population: int,
    baseline: np.ndarray | None,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    if baseline is not None:
        candidates.append(bounds.clip(baseline))
    while len(candidates) < population:
        candidates.append(bounds.clip(rng.normal(loc=mean, scale=std)))
    return candidates


def _local_z_rotation(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _orthonormalize(rotation_matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3))
    out = u @ vh
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vh
    return out


def _positive_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "success",
            "stage",
            "object_lift_delta_m",
            "not_converged",
            "truncated",
        )
        if key in result
    }
