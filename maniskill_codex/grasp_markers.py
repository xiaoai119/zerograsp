"""Visual-only SAPIEN markers for ZeroGrasp poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from maniskill_codex.grasp_axes import zerograsp_approach_vector, zerograsp_width_vector
from maniskill_codex.transforms import OPENCV_TO_SAPIEN_CAMERA, opencv_camera_to_world
from maniskill_codex.zerograsp_outputs import GraspRecord


@dataclass(frozen=True)
class GraspMarkerGeometry:
    """World-frame geometry used to draw a grasp marker."""

    center_world: np.ndarray
    approach_axis_world: np.ndarray
    width_axis_world: np.ndarray
    depth_axis_world: np.ndarray
    approach_end_world: np.ndarray
    width_endpoints_world: tuple[np.ndarray, np.ndarray]
    width_m: float
    score: float
    object_id: int | None


def opencv_grasp_rotation_to_world_axes(
    rotation_matrix_camera: np.ndarray,
    camera_model_matrix: np.ndarray,
) -> np.ndarray:
    """Convert ZeroGrasp camera-frame rotation axes to world-frame axes."""

    R_cv = np.asarray(rotation_matrix_camera, dtype=np.float64).reshape(3, 3)
    camera_model = np.asarray(camera_model_matrix, dtype=np.float64).reshape(4, 4)
    R_sapien = OPENCV_TO_SAPIEN_CAMERA @ R_cv
    R_world = camera_model[:3, :3] @ R_sapien
    return _normalize_columns(R_world)


def build_grasp_marker_geometry(
    grasp: GraspRecord,
    camera_model_matrix: np.ndarray,
    approach_length: float = 0.08,
    approach_axis: str = "negative-x",
    center_world: np.ndarray | None = None,
) -> GraspMarkerGeometry:
    """Build world-frame marker geometry from one ZeroGrasp grasp."""

    center = (
        opencv_camera_to_world(grasp.translation_m_camera, camera_model_matrix)
        if center_world is None
        else np.asarray(center_world, dtype=np.float64).reshape(3)
    )
    axes = opencv_grasp_rotation_to_world_axes(grasp.rotation_matrix_camera, camera_model_matrix)
    approach = _unit(zerograsp_approach_vector(axes, approach_axis))
    width_axis = _unit(zerograsp_width_vector(axes))
    depth_axis = _unit(axes[:, 2])
    half_width = float(grasp.width_m) / 2.0
    return GraspMarkerGeometry(
        center_world=center,
        approach_axis_world=approach,
        width_axis_world=width_axis,
        depth_axis_world=depth_axis,
        approach_end_world=center + approach * float(approach_length),
        width_endpoints_world=(center + width_axis * half_width, center - width_axis * half_width),
        width_m=float(grasp.width_m),
        score=float(grasp.score),
        object_id=grasp.object_id,
    )


def add_grasp_marker_to_scene(
    scene: Any,
    geometry: GraspMarkerGeometry,
    name_prefix: str = "zerograsp_marker",
) -> list[Any]:
    """Add visual-only marker actors to a ManiSkill/SAPIEN scene."""

    import sapien

    actors = []
    center = geometry.center_world

    sphere_builder = scene.create_actor_builder()
    sphere_builder.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        radius=0.018,
        material=sapien.render.RenderMaterial(base_color=[0.0, 1.0, 0.0, 0.85]),
    )
    sphere_builder.set_initial_pose(sapien.Pose(p=center.tolist()))
    sphere = sphere_builder.build_kinematic(name=f"{name_prefix}_center")
    actors.append(sphere)

    actors.append(
        _add_bar_actor(
            scene,
            start=center,
            end=geometry.approach_end_world,
            thickness=0.006,
            color=[1.0, 0.0, 0.0, 0.85],
            name=f"{name_prefix}_approach",
        )
    )
    actors.append(
        _add_bar_actor(
            scene,
            start=geometry.width_endpoints_world[0],
            end=geometry.width_endpoints_world[1],
            thickness=0.005,
            color=[0.0, 0.25, 1.0, 0.85],
            name=f"{name_prefix}_width",
        )
    )

    tip_builder = scene.create_actor_builder()
    tip_builder.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        radius=0.012,
        material=sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 0.9]),
    )
    tip_builder.set_initial_pose(sapien.Pose(p=geometry.approach_end_world.tolist()))
    tip = tip_builder.build_kinematic(name=f"{name_prefix}_approach_tip")
    actors.append(tip)
    return actors


def _add_bar_actor(
    scene: Any,
    start: np.ndarray,
    end: np.ndarray,
    thickness: float,
    color: list[float],
    name: str,
) -> Any:
    import sapien

    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        length = 1e-6
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    center = (start + end) / 2.0
    pose = sapien.Pose(
        p=center.tolist(),
        q=_quat_from_x_axis(direction).tolist(),
    )
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        pose=sapien.Pose(),
        half_size=[length / 2.0, thickness, thickness],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    builder.set_initial_pose(pose)
    actor = builder.build_kinematic(name=name)
    return actor


def _quat_from_x_axis(direction: np.ndarray) -> np.ndarray:
    x_axis = _unit(direction)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x_axis, helper))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = _unit(np.cross(helper, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return _quat_wxyz_from_matrix(R)


def _quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        q = np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
            q = np.array(
                [
                    (R[2, 1] - R[1, 2]) / s,
                    0.25 * s,
                    (R[0, 1] + R[1, 0]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                ]
            )
        elif idx == 1:
            s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
            q = np.array(
                [
                    (R[0, 2] - R[2, 0]) / s,
                    (R[0, 1] + R[1, 0]) / s,
                    0.25 * s,
                    (R[1, 2] + R[2, 1]) / s,
                ]
            )
        else:
            s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
            q = np.array(
                [
                    (R[1, 0] - R[0, 1]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s,
                    0.25 * s,
                ]
            )
    return _unit(q)


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64).reshape(3, 3).copy()
    for i in range(3):
        arr[:, i] = _unit(arr[:, i])
    return arr


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return arr / norm
