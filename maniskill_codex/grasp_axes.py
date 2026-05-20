"""Shared ZeroGrasp axis conventions for projection, markers, and execution."""

from __future__ import annotations

import numpy as np


APPROACH_AXIS_CHOICES = ("negative-x", "positive-x", "flip-world-z")


def validate_approach_axis(approach_axis: str) -> str:
    """Return a valid approach axis name or raise a clear error."""

    if approach_axis not in APPROACH_AXIS_CHOICES:
        raise ValueError(
            f"Unsupported approach_axis={approach_axis!r}. "
            f"Expected one of {APPROACH_AXIS_CHOICES}."
        )
    return approach_axis


def zerograsp_approach_vector(rotation_matrix: np.ndarray, approach_axis: str) -> np.ndarray:
    """Return the selected ZeroGrasp approach vector in the matrix frame."""

    validate_approach_axis(approach_axis)
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    if approach_axis == "flip-world-z":
        approach = -rotation[:, 0].copy()
        approach[2] *= -1.0
        return approach
    sign = 1.0 if approach_axis == "positive-x" else -1.0
    return sign * rotation[:, 0]


def zerograsp_width_vector(rotation_matrix: np.ndarray) -> np.ndarray:
    """Return the ZeroGrasp gripper width vector in the matrix frame."""

    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    return -rotation[:, 1]


def panda_tcp_axes_in_zerograsp_frame(approach_axis: str) -> np.ndarray:
    """Map ZeroGrasp axes to Panda TCP axes while keeping a right-handed frame."""

    validate_approach_axis(approach_axis)
    if approach_axis == "positive-x":
        return np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
