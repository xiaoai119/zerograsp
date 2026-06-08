"""Prepare ManiSkill RGB-D inputs and serialize GraspNet predictions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PointCloudInput:
    points: np.ndarray
    colors: np.ndarray
    target_limits: list[float]
    target_label: int
    target_point_count: int
    context_point_count: int
    depth_scale_to_m: float


def load_point_cloud_input(
    input_dir: str | Path,
    *,
    depth_unit: str = "auto",
    target_margin_m: float = 0.02,
    context_margin_m: float = 0.08,
    min_target_points: int = 64,
) -> PointCloudInput:
    """Load an exported ManiSkill bundle as a local camera-frame point cloud."""

    root = Path(input_dir).expanduser().resolve()
    camera = json.loads((root / "camera.json").read_text(encoding="utf-8"))
    if (root / "rgbd.npz").is_file():
        bundle = np.load(root / "rgbd.npz")
        rgb = np.asarray(bundle["rgb"])
        depth = np.asarray(bundle["depth"], dtype=np.float32)
        mask = np.asarray(bundle["mask"])
        intrinsic = np.asarray(bundle["cam_K"], dtype=np.float32).reshape(3, 3)
    else:
        rgb = np.asarray(Image.open(root / "rgb.png").convert("RGB"))
        depth = np.asarray(Image.open(root / "depth.png"), dtype=np.float32)
        mask = np.asarray(Image.open(root / "mask.png"))
        intrinsic = np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3)

    scale_to_m = resolve_depth_scale_to_m(depth, depth_unit)
    depth_m = depth * scale_to_m
    valid = np.isfinite(depth_m) & (depth_m > 0.02) & (depth_m < 3.0)
    target = valid & (mask > 0)
    target_count = int(target.sum())
    if target_count < min_target_points:
        raise ValueError(
            f"{root} has only {target_count} valid target points; "
            f"at least {min_target_points} are required."
        )

    height, width = depth_m.shape
    xmap, ymap = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    z = depth_m
    x = (xmap - cx) * z / fx
    y = (ymap - cy) * z / fy
    points_image = np.stack((x, y, z), axis=-1)

    target_points = points_image[target]
    target_min = target_points.min(axis=0) - float(target_margin_m)
    target_max = target_points.max(axis=0) + float(target_margin_m)
    target_min[2] = max(target_min[2], 0.02)

    context_min = target_min - float(context_margin_m)
    context_max = target_max + float(context_margin_m)
    context_min[2] = max(context_min[2], 0.02)
    context = valid & np.all(points_image >= context_min, axis=-1)
    context &= np.all(points_image <= context_max, axis=-1)

    points = np.ascontiguousarray(points_image[context], dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.float32) / 255.0
    colors = np.ascontiguousarray(colors[context], dtype=np.float32)
    target_labels, counts = np.unique(mask[target], return_counts=True)
    target_label = int(target_labels[int(np.argmax(counts))])
    limits = [
        float(target_min[0]),
        float(target_max[0]),
        float(target_min[1]),
        float(target_max[1]),
        float(target_min[2]),
        float(target_max[2]),
    ]
    return PointCloudInput(
        points=points,
        colors=colors,
        target_limits=limits,
        target_label=target_label,
        target_point_count=target_count,
        context_point_count=int(context.sum()),
        depth_scale_to_m=scale_to_m,
    )


def resolve_depth_scale_to_m(depth: np.ndarray, depth_unit: str) -> float:
    if depth_unit == "m":
        return 1.0
    if depth_unit == "mm":
        return 0.001
    if depth_unit != "auto":
        raise ValueError("depth_unit must be one of: auto, m, mm")
    finite = np.asarray(depth, dtype=np.float32)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        raise ValueError("Depth image contains no positive finite values.")
    return 0.001 if float(np.median(finite)) > 10.0 else 1.0


def filter_grasps_by_target_limits(grasp_group: Any, limits: list[float]) -> Any:
    if len(grasp_group) == 0:
        return grasp_group
    translations = np.asarray(grasp_group.translations)
    mask = (
        (translations[:, 0] >= limits[0])
        & (translations[:, 0] <= limits[1])
        & (translations[:, 1] >= limits[2])
        & (translations[:, 1] <= limits[3])
        & (translations[:, 2] >= limits[4])
        & (translations[:, 2] <= limits[5])
    )
    return grasp_group[mask]


def grasp_to_standard_json(
    grasp: Any,
    *,
    object_id: int,
    source_file: str = "raw_outputs/graspnet.grasp.npy",
) -> dict[str, Any]:
    """Convert one graspnetAPI grasp to the shared executor schema."""

    return {
        "score": float(grasp.score),
        "width_m": float(grasp.width),
        "height_m": float(grasp.height),
        "depth_m": float(grasp.depth),
        "rotation_matrix_camera": np.asarray(
            grasp.rotation_matrix, dtype=np.float64
        ).reshape(3, 3).tolist(),
        "translation_m_camera": np.asarray(
            grasp.translation, dtype=np.float64
        ).reshape(3).tolist(),
        "object_id": int(object_id),
        "source_file": source_file,
        "model": "graspnet-baseline",
    }
