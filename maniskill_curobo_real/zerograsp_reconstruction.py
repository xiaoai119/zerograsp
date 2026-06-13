"""Load ZeroGrasp surface reconstructions as per-instance base-frame clouds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_curobo_real.scene_builder import (
    OPENCV_TO_SAPIEN_CAMERA,
    transform_points,
)


@dataclass(frozen=True)
class ReconstructedInstance:
    label: int
    segmentation_id: int | None
    actor_name: str
    is_task_target: bool
    points_base: np.ndarray
    normals_camera: np.ndarray
    reconstruction_file: str


def opencv_camera_points_mm_to_base(
    points_mm: np.ndarray,
    *,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
) -> np.ndarray:
    """Transform ZeroGrasp OpenCV-camera points in millimeters to base meters."""

    points_cv_m = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3) / 1000.0
    points_sapien_camera = points_cv_m @ OPENCV_TO_SAPIEN_CAMERA.T
    world_from_camera = np.asarray(camera_model_matrix, dtype=np.float64).reshape(4, 4)
    base_from_world = np.linalg.inv(
        np.asarray(world_from_base_matrix, dtype=np.float64).reshape(4, 4)
    )
    points_world = transform_points(world_from_camera, points_sapien_camera)
    return transform_points(base_from_world, points_world)


def load_zerograsp_reconstructed_instances(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
) -> list[ReconstructedInstance]:
    """Load saved reconstructions and attach the original instance metadata."""

    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    camera_payload = json.loads((input_path / "camera.json").read_text(encoding="utf-8"))
    report = json.loads((output_path / "run_report.json").read_text(encoding="utf-8"))
    records_by_label = {
        int(record["label"]): record
        for record in camera_payload.get("objects", [])
        if "label" in record
    }

    instances: list[ReconstructedInstance] = []
    for object_report in report.get("objects", []):
        label = int(object_report["object_id"])
        record = records_by_label.get(label, {})
        relative_file = object_report.get("reconstruction_file")
        if not relative_file:
            continue
        reconstruction_path = output_path / str(relative_file)
        if not reconstruction_path.is_file():
            continue
        with np.load(reconstruction_path) as payload:
            points_mm = np.asarray(payload["points_mm"], dtype=np.float64).reshape(-1, 3)
            normals = np.asarray(payload["normals"], dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(points_mm).all(axis=1)
        points_mm = points_mm[finite]
        normals = normals[finite]
        points_base = opencv_camera_points_mm_to_base(
            points_mm,
            camera_model_matrix=camera_model_matrix,
            world_from_base_matrix=world_from_base_matrix,
        )
        segmentation_id = record.get("segmentation_id")
        instances.append(
            ReconstructedInstance(
                label=label,
                segmentation_id=(
                    int(segmentation_id) if segmentation_id is not None else None
                ),
                actor_name=str(record.get("actor_name") or f"label_{label}"),
                is_task_target=bool(record.get("is_task_target", False)),
                points_base=points_base,
                normals_camera=normals,
                reconstruction_file=str(reconstruction_path),
            )
        )
    return instances


def reconstructed_instances_to_metadata(
    instances: list[ReconstructedInstance],
) -> list[dict[str, Any]]:
    return [
        {
            "label": int(instance.label),
            "segmentation_id": instance.segmentation_id,
            "actor_name": instance.actor_name,
            "is_task_target": bool(instance.is_task_target),
            "points": int(instance.points_base.shape[0]),
            "reconstruction_file": instance.reconstruction_file,
        }
        for instance in instances
    ]
