"""Coordinate transforms for ZeroGrasp outputs in ManiSkill."""

from __future__ import annotations

import numpy as np

OPENCV_TO_SAPIEN_CAMERA = np.diag([1.0, -1.0, -1.0])


def opencv_camera_to_sapien_camera(position_m: np.ndarray) -> np.ndarray:
    """Convert an OpenCV camera-frame point to SAPIEN camera-frame coordinates."""

    pos = _vector3(position_m, "position_m")
    return OPENCV_TO_SAPIEN_CAMERA @ pos


def opencv_camera_to_world(position_m: np.ndarray, camera_model_matrix: np.ndarray) -> np.ndarray:
    """Convert an OpenCV camera-frame point to world coordinates."""

    camera_from_opencv = opencv_camera_to_sapien_camera(position_m)
    world_from_camera = _matrix4(camera_model_matrix, "camera_model_matrix")
    return (world_from_camera @ np.array([*camera_from_opencv, 1.0], dtype=np.float64))[:3]


def opencv_camera_to_base(
    position_m: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
) -> np.ndarray:
    """Convert an OpenCV camera-frame point to robot base coordinates."""

    p_world = opencv_camera_to_world(position_m, camera_model_matrix)
    world_from_base = _matrix4(world_from_base_matrix, "world_from_base_matrix")
    base_from_world = np.linalg.inv(world_from_base)
    return (base_from_world @ np.array([*p_world, 1.0], dtype=np.float64))[:3]


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _matrix4(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {arr.shape}.")
    return arr
