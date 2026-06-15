"""Helpers that bridge Lift2's ManiSkill joint order and cuRobo right-arm order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .lift2_constants import LIFT2_JOINT_NAMES, LIFT2_REST_QPOS


RIGHT_ARM_CUROBO_JOINT_NAMES = (
    "joint4",
    "right_joint21",
    "right_joint22",
    "right_joint23",
    "right_joint24",
    "right_joint25",
    "right_joint26",
)
RIGHT_GRIPPER_JOINT_NAMES = ("right_joint27", "right_joint28")
RIGHT_ARM_ACTION_INDICES = tuple(LIFT2_JOINT_NAMES.index(name) for name in RIGHT_ARM_CUROBO_JOINT_NAMES)
RIGHT_GRIPPER_ACTION_INDICES = tuple(LIFT2_JOINT_NAMES.index(name) for name in RIGHT_GRIPPER_JOINT_NAMES)


def right_arm_qpos_from_maniskill(
    active_joint_names: Sequence[str],
    qpos: Sequence[float] | np.ndarray,
    *,
    curobo_joint_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Extract cuRobo's right-arm joint vector from ManiSkill qpos."""

    joint_names = tuple(RIGHT_ARM_CUROBO_JOINT_NAMES if curobo_joint_names is None else curobo_joint_names)
    name_to_value = {
        str(name): float(value)
        for name, value in zip(active_joint_names, np.asarray(qpos, dtype=np.float64).reshape(-1))
    }
    return np.asarray(
        [name_to_value[name] for name in joint_names],
        dtype=np.float32,
    )


def make_lift2_action_from_right_arm_qpos(
    right_arm_qpos: Sequence[float] | Mapping[str, float] | np.ndarray,
    *,
    base_action: Sequence[float] | np.ndarray | None = None,
    curobo_joint_names: Sequence[str] | None = None,
    gripper_qpos: float | None = None,
) -> np.ndarray:
    """Return a full ManiSkill pd_joint_pos action from cuRobo right-arm qpos."""

    joint_names = tuple(RIGHT_ARM_CUROBO_JOINT_NAMES if curobo_joint_names is None else curobo_joint_names)
    action = (
        np.asarray(base_action, dtype=np.float32).reshape(-1).copy()
        if base_action is not None
        else LIFT2_REST_QPOS.astype(np.float32).copy()
    )
    if action.shape != LIFT2_REST_QPOS.shape:
        raise ValueError(
            f"base_action must have shape {LIFT2_REST_QPOS.shape}, got {action.shape}."
        )

    if isinstance(right_arm_qpos, Mapping):
        values = [float(right_arm_qpos[name]) for name in joint_names]
    else:
        values = np.asarray(right_arm_qpos, dtype=np.float64).reshape(-1).tolist()
    if len(values) != len(joint_names):
        raise ValueError(
            f"right_arm_qpos must have {len(joint_names)} values, "
            f"got {len(values)}."
        )
    for name, value in zip(joint_names, values):
        index = LIFT2_JOINT_NAMES.index(name)
        action[index] = float(value)

    if gripper_qpos is not None:
        for index in RIGHT_GRIPPER_ACTION_INDICES:
            action[index] = float(gripper_qpos)
    return action
