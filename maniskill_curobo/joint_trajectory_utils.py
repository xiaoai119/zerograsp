"""Utilities for feeding cuRobo joint trajectories into ManiSkill controllers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ordered_values(
    source_names: Sequence[str],
    values: np.ndarray,
    target_names: Sequence[str],
) -> np.ndarray:
    """Return values ordered by target_names using source_names as labels."""

    source = list(source_names)
    target = list(target_names)
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D values, got shape {arr.shape}.")
    if len(source) != arr.shape[0]:
        raise ValueError(
            f"source_names length ({len(source)}) does not match values length ({arr.shape[0]})."
        )

    name_to_index = {name: index for index, name in enumerate(source)}
    missing = [name for name in target if name not in name_to_index]
    if missing:
        raise KeyError(f"Missing joint names in source trajectory: {missing}")

    return arr[[name_to_index[name] for name in target]]


def squeeze_trajectory_positions(trajectory: np.ndarray) -> np.ndarray:
    """Normalize cuRobo trajectory positions to a (time, joints) array."""

    arr = np.asarray(trajectory)
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
        arr = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            "Expected trajectory positions shaped (T, J), (1, T, J), or (1, 1, T, J); "
            f"got {arr.shape}."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Trajectory must be non-empty, got shape {arr.shape}.")
    return np.asarray(arr, dtype=np.float32)


def make_pd_joint_pos_actions(
    trajectory: np.ndarray,
    trajectory_joint_names: Sequence[str],
    arm_action_joint_names: Sequence[str],
    gripper: float,
) -> np.ndarray:
    """Convert a cuRobo trajectory to ManiSkill pd_joint_pos actions.

    ManiSkill Panda `pd_joint_pos` uses 7 arm joint targets plus one mimic-gripper
    scalar. cuRobo may return arm-only or arm-plus-finger joint trajectories, so
    joints are selected by name instead of by column index.
    """

    positions = squeeze_trajectory_positions(trajectory)
    source = list(trajectory_joint_names)
    if len(source) != positions.shape[1]:
        raise ValueError(
            "trajectory_joint_names length "
            f"({len(source)}) does not match trajectory width ({positions.shape[1]})."
        )

    arm_columns = [
        ordered_values(source, row, arm_action_joint_names)
        for row in positions
    ]
    arm = np.asarray(arm_columns, dtype=np.float32)
    gripper_column = np.full((arm.shape[0], 1), float(gripper), dtype=np.float32)
    return np.concatenate([arm, gripper_column], axis=1)


def sample_waypoints(waypoints: np.ndarray, max_waypoints: int) -> np.ndarray:
    """Uniformly downsample waypoints while preserving the first and last rows."""

    arr = np.asarray(waypoints)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D waypoints, got shape {arr.shape}.")
    if max_waypoints <= 0:
        raise ValueError(f"max_waypoints must be positive, got {max_waypoints}.")
    if arr.shape[0] <= max_waypoints:
        return arr

    indices = np.linspace(0, arr.shape[0] - 1, num=max_waypoints, dtype=np.int64)
    indices = np.unique(indices)
    if indices[-1] != arr.shape[0] - 1:
        indices = np.append(indices, arr.shape[0] - 1)
    return arr[indices]
