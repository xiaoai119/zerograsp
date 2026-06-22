"""Instance-aware multi-view collision scene builders.

This module keeps the M5 instance-aware path separate from the earlier global
RGB-D fusion baseline. It uses per-view instance masks to merge points from the
same object across camera views, excludes the task target, and builds collision
geometry only for non-target tabletop objects. The table is intentionally not
included in these builders so this branch can isolate object reconstruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_curobo_real.scene_builder import (
    DEFAULT_TABLE_TOP_Z,
    DEFAULT_WORKSPACE_BOUNDS,
    OPENCV_TO_SAPIEN_CAMERA,
    SceneBuildResult,
    VoxelSceneBuildResult,
    _round_list,
    _safe_name,
    _voxelize_closed_instance,
    build_static_table_scene,
    depth_to_meters,
    deterministic_downsample,
    points_to_yaw_obb_cuboid,
    transform_points,
    valid_workspace_mask,
)


@dataclass(frozen=True)
class MultiviewInstance:
    segmentation_id: int
    actor_name: str
    is_task_target: bool
    points_base: np.ndarray
    labels_by_view: dict[str, int]
    points_by_view: dict[str, int]


def load_multiview_instances(
    *,
    input_root: Path,
    seed: int,
    workspace_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = DEFAULT_WORKSPACE_BOUNDS,
    depth_min: float = 0.05,
    depth_max: float = 2.5,
) -> tuple[list[MultiviewInstance], dict[str, Any]]:
    """Load three-view RGB-D bundles and merge points by segmentation id."""

    seed_dir = input_root.expanduser().resolve() / f"seed{int(seed):03d}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Missing multi-view seed directory: {seed_dir}")

    per_instance: dict[int, dict[str, Any]] = {}
    view_records: list[dict[str, Any]] = []
    valid_points = 0
    target_points = 0
    non_target_points = 0

    for metadata_path in sorted(seed_dir.glob("view_*/view_metadata.json")):
        view_dir = metadata_path.parent
        rgbd_path = view_dir / "rgbd.npz"
        camera_path = view_dir / "camera.json"
        if not rgbd_path.is_file() or not camera_path.is_file():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        with np.load(rgbd_path) as payload:
            depth = depth_to_meters(np.asarray(payload["depth"], dtype=np.float32))
            mask = np.asarray(payload["mask"], dtype=np.uint8)
            intrinsics = np.asarray(payload["cam_K"], dtype=np.float64).reshape(3, 3)

        base_from_camera = np.asarray(metadata["base_from_camera"], dtype=np.float64).reshape(4, 4)
        cloud = rgbd_to_base_cloud_from_base_camera(
            depth_m=depth,
            camera_matrix=intrinsics,
            base_from_camera_matrix=base_from_camera,
        )
        valid = valid_workspace_mask(cloud, depth, workspace_bounds)
        depth_flat = depth.reshape(-1)
        valid &= np.isfinite(depth_flat)
        valid &= depth_flat >= float(depth_min)
        valid &= depth_flat <= float(depth_max)
        mask_flat = mask.reshape(-1)

        view_name = str(metadata.get("view", {}).get("name") or view_dir.name)
        object_records = list(camera.get("objects", []))
        label_records = {
            int(record["label"]): record
            for record in object_records
            if "label" in record and "segmentation_id" in record
        }
        view_valid_points = int(np.count_nonzero(valid))
        view_target_points = 0
        view_non_target_points = 0

        for label, record in sorted(label_records.items()):
            instance_mask = valid & (mask_flat == int(label))
            points = cloud[instance_mask]
            is_target = bool(record.get("is_task_target", False))
            count = int(points.shape[0])
            if is_target:
                view_target_points += count
            else:
                view_non_target_points += count
            if count == 0:
                continue

            segmentation_id = int(record["segmentation_id"])
            slot = per_instance.setdefault(
                segmentation_id,
                {
                    "segmentation_id": segmentation_id,
                    "actor_name": str(record.get("actor_name") or f"seg_{segmentation_id}"),
                    "is_task_target": is_target,
                    "point_chunks": [],
                    "labels_by_view": {},
                    "points_by_view": {},
                },
            )
            slot["is_task_target"] = bool(slot["is_task_target"] or is_target)
            slot["labels_by_view"][view_name] = int(label)
            slot["points_by_view"][view_name] = count
            slot["point_chunks"].append(points)

        valid_points += view_valid_points
        target_points += view_target_points
        non_target_points += view_non_target_points
        view_records.append(
            {
                "name": view_name,
                "role": metadata.get("view", {}).get("role"),
                "rgbd": str(rgbd_path),
                "metadata": str(metadata_path),
                "n_object_records": len(label_records),
                "valid_points": view_valid_points,
                "target_points": int(view_target_points),
                "non_target_instance_points": int(view_non_target_points),
            }
        )

    instances: list[MultiviewInstance] = []
    for slot in per_instance.values():
        points = np.concatenate(slot["point_chunks"], axis=0)
        instances.append(
            MultiviewInstance(
                segmentation_id=int(slot["segmentation_id"]),
                actor_name=str(slot["actor_name"]),
                is_task_target=bool(slot["is_task_target"]),
                points_base=np.asarray(points, dtype=np.float64).reshape(-1, 3),
                labels_by_view=dict(slot["labels_by_view"]),
                points_by_view=dict(slot["points_by_view"]),
            )
        )
    instances.sort(key=lambda item: item.points_base.shape[0], reverse=True)
    metadata = {
        "source": "m5_multiview_instance_inputs",
        "seed": int(seed),
        "input_root": str(input_root.expanduser().resolve()),
        "n_views": len(view_records),
        "views": view_records,
        "valid_points_accumulated_over_views": int(valid_points),
        "target_points_accumulated_over_views": int(target_points),
        "non_target_instance_points_accumulated_over_views": int(non_target_points),
        "n_instances_seen": len(instances),
        "workspace_bounds": workspace_bounds,
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
        "target_exclusion_applied": True,
        "uses_oracle_segmentation_truth": True,
        "table_included": False,
    }
    return instances, metadata


def build_multiview_instance_obb_scene(
    *,
    instances: list[MultiviewInstance],
    input_metadata: dict[str, Any],
    min_instance_points: int = 400,
    instance_padding: float = 0.0,
    min_cuboid_dimension: float = 0.005,
    max_obstacles: int = 50,
    include_table: bool = False,
    table_top_z: float = DEFAULT_TABLE_TOP_Z,
) -> SceneBuildResult:
    """Build one yaw OBB per merged non-target instance."""

    source = (
        "m5_multiview_instance_obb_table"
        if bool(include_table)
        else "m5_multiview_instance_obb_no_table"
    )
    table = build_static_table_scene(table_top_z=float(table_top_z), source=source) if include_table else None
    candidates, skipped = _candidate_instances(
        instances,
        min_instance_points=min_instance_points,
        max_obstacles=max_obstacles,
    )
    cuboids: dict[str, Any] = dict(table.scene["cuboid"]) if table is not None else {}
    obstacle_records: list[dict[str, Any]] = []
    for index, instance in enumerate(candidates):
        obstacle = points_to_yaw_obb_cuboid(
            instance.points_base,
            padding=float(instance_padding),
            min_dimension=float(min_cuboid_dimension),
        )
        if obstacle is None:
            skipped.append(_instance_record(instance, reason="obb_fit_failed"))
            continue
        name = f"m5_instance_obb_{index:02d}_{_safe_name(instance.actor_name)}"
        cuboids[name] = {
            "dims": obstacle["dims"],
            "pose": obstacle["pose"],
        }
        obstacle_records.append(
            {
                **_instance_record(instance),
                "name": name,
                "geometry_type": "yaw_obb",
                "dims": obstacle["dims"],
                "pose": obstacle["pose"],
                "yaw_rad": obstacle["yaw_rad"],
            }
        )

    metadata = _build_metadata(
        source=source,
        input_metadata=input_metadata,
        instances=instances,
        obstacle_records=obstacle_records,
        skipped_records=skipped,
        min_instance_points=min_instance_points,
        max_obstacles=max_obstacles,
        extra={
            "geometry_type": "yaw_obb",
            "instance_padding": float(instance_padding),
            "min_cuboid_dimension": float(min_cuboid_dimension),
            **_table_metadata_extra(table, obstacle_records),
        },
    )
    return SceneBuildResult(scene={"cuboid": cuboids}, metadata=metadata)


def build_multiview_instance_voxel_esdf_scene(
    *,
    instances: list[MultiviewInstance],
    input_metadata: dict[str, Any],
    workspace_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = DEFAULT_WORKSPACE_BOUNDS,
    min_instance_points: int = 400,
    max_obstacles: int = 50,
    max_points_per_instance: int = 1500,
    voxel_size: float = 0.01,
    dilation_voxels: int = 0,
    include_table: bool = False,
    table_top_z: float = DEFAULT_TABLE_TOP_Z,
) -> VoxelSceneBuildResult:
    """Build one ESDF grid from merged non-target instances."""

    from scipy.ndimage import binary_dilation, distance_transform_edt

    source = (
        "m5_multiview_instance_voxel_esdf_table"
        if bool(include_table)
        else "m5_multiview_instance_voxel_esdf_no_table"
    )
    table = build_static_table_scene(table_top_z=float(table_top_z), source=source) if include_table else None
    voxel_size = float(voxel_size)
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive.")
    bounds = np.asarray(workspace_bounds, dtype=np.float64)
    counts = np.ceil((bounds[:, 1] - bounds[:, 0]) / voxel_size).astype(np.int64)
    dims = counts.astype(np.float64) * voxel_size
    center = 0.5 * (bounds[:, 0] + bounds[:, 1])
    axes = [
        center[axis]
        + (np.arange(counts[axis], dtype=np.float64) - (counts[axis] - 1) / 2.0)
        * voxel_size
        for axis in range(3)
    ]
    occupied = np.zeros(tuple(int(value) for value in counts), dtype=bool)
    candidates, skipped = _candidate_instances(
        instances,
        min_instance_points=min_instance_points,
        max_obstacles=max_obstacles,
    )
    obstacle_records: list[dict[str, Any]] = []
    fallback_count = 0
    for index, instance in enumerate(candidates):
        sampled = deterministic_downsample(instance.points_base, int(max_points_per_instance))
        filled, used_fallback = _voxelize_closed_instance(
            occupied=occupied,
            axes=axes,
            points=sampled,
            voxel_size=voxel_size,
        )
        if filled == 0:
            skipped.append(_instance_record(instance, reason="voxelization_empty"))
            continue
        fallback_count += int(used_fallback)
        obstacle_records.append(
            {
                **_instance_record(instance),
                "name": f"m5_instance_voxel_{index:02d}_{_safe_name(instance.actor_name)}",
                "geometry_type": "voxel_esdf",
                "occupied_voxels_added": int(filled),
                "voxelization": "yaw_obb_fallback" if used_fallback else "convex_hull",
            }
        )

    if int(dilation_voxels) > 0:
        occupied = binary_dilation(occupied, iterations=int(dilation_voxels))
    if np.any(occupied):
        outside = distance_transform_edt(~occupied, sampling=voxel_size)
        inside = distance_transform_edt(occupied, sampling=voxel_size)
        sdf = outside.astype(np.float32)
        sdf[occupied] = -inside[occupied].astype(np.float32)
    else:
        sdf = np.full(tuple(int(value) for value in counts), 10000.0, dtype=np.float32)

    metadata = _build_metadata(
        source=source,
        input_metadata=input_metadata,
        instances=instances,
        obstacle_records=obstacle_records,
        skipped_records=skipped,
        min_instance_points=min_instance_points,
        max_obstacles=max_obstacles,
        extra={
            "geometry_type": "voxel_esdf",
            "voxel_size": voxel_size,
            "voxel_dims": _round_list(dims),
            "voxel_center": _round_list(center),
            "voxel_shape": [int(value) for value in counts],
            "occupied_voxels": int(np.count_nonzero(occupied)),
            "occupied_fraction": float(np.mean(occupied)),
            "dilation_voxels": int(dilation_voxels),
            "n_voxel_fallbacks": int(fallback_count),
            "max_points_per_instance": int(max_points_per_instance),
            **_table_metadata_extra(table, obstacle_records),
        },
    )
    return VoxelSceneBuildResult(
        table_scene=table.scene if table is not None else {"cuboid": {}},
        voxel_center=center,
        voxel_dims=dims,
        voxel_size=voxel_size,
        feature_tensor=sdf.astype(np.float16),
        metadata=metadata,
    )


def write_multiview_instance_voxel_scene_result(
    result: VoxelSceneBuildResult,
    scene_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> None:
    """Write voxel scene npz understood by the persistent runner."""

    scene_out = Path(scene_path).expanduser().resolve()
    meta_out = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else scene_out.with_name("curobo_scene_metadata.json")
    )
    scene_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "feature_tensor": result.feature_tensor,
        "voxel_center": result.voxel_center.astype(np.float64),
        "voxel_dims": result.voxel_dims.astype(np.float64),
        "voxel_size": np.asarray(result.voxel_size, dtype=np.float64),
    }
    table_cuboids = result.table_scene.get("cuboid", {})
    if table_cuboids:
        table = next(iter(table_cuboids.values()))
        payload["table_pose"] = np.asarray(table["pose"], dtype=np.float64)
        payload["table_dims"] = np.asarray(table["dims"], dtype=np.float64)
    np.savez_compressed(scene_out, **payload)
    meta_out.write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def rgbd_to_base_cloud_from_base_camera(
    *,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    base_from_camera_matrix: np.ndarray,
) -> np.ndarray:
    depth = depth_to_meters(np.asarray(depth_m, dtype=np.float64))
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got {depth.shape}.")
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    h, w = depth.shape
    ys, xs = np.indices((h, w), dtype=np.float64)
    z = depth.reshape(-1)
    x = (xs.reshape(-1) - k[0, 2]) * z / k[0, 0]
    y = (ys.reshape(-1) - k[1, 2]) * z / k[1, 1]
    points_cv = np.stack([x, y, z], axis=1)
    points_cam = points_cv @ OPENCV_TO_SAPIEN_CAMERA.T
    return transform_points(base_from_camera_matrix, points_cam)


def _candidate_instances(
    instances: list[MultiviewInstance],
    *,
    min_instance_points: int,
    max_obstacles: int,
) -> tuple[list[MultiviewInstance], list[dict[str, Any]]]:
    candidates: list[MultiviewInstance] = []
    skipped: list[dict[str, Any]] = []
    for instance in instances:
        if instance.is_task_target:
            skipped.append(_instance_record(instance, reason="target_excluded"))
        elif instance.points_base.shape[0] < int(min_instance_points):
            skipped.append(_instance_record(instance, reason="too_few_points"))
        else:
            candidates.append(instance)
    candidates.sort(key=lambda item: item.points_base.shape[0], reverse=True)
    return candidates[: int(max_obstacles)], skipped


def _instance_record(
    instance: MultiviewInstance,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    record = {
        "segmentation_id": int(instance.segmentation_id),
        "actor_name": instance.actor_name,
        "is_task_target": bool(instance.is_task_target),
        "points": int(instance.points_base.shape[0]),
        "labels_by_view": dict(instance.labels_by_view),
        "points_by_view": dict(instance.points_by_view),
    }
    if reason is not None:
        record["reason"] = reason
    return record


def _build_metadata(
    *,
    source: str,
    input_metadata: dict[str, Any],
    instances: list[MultiviewInstance],
    obstacle_records: list[dict[str, Any]],
    skipped_records: list[dict[str, Any]],
    min_instance_points: int,
    max_obstacles: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(input_metadata)
    metadata.update(
        {
            "source": source,
            "point_cloud_source": "multiview_rgbd_instance_masks",
            "n_instances_seen": len(instances),
            "n_pointcloud_obstacles": len(obstacle_records),
            "n_obstacles": len(obstacle_records),
            "min_instance_points": int(min_instance_points),
            "max_obstacles": int(max_obstacles),
            "obstacle_records": obstacle_records,
            "skipped_instance_records": skipped_records,
            "target_exclusion_applied": True,
            "uses_actor_collision_truth": False,
            "uses_oracle_segmentation_truth": True,
            "uses_maniskill_depth_geometry": True,
            "uses_zerograsp_reconstruction": False,
            "table_included": False,
        }
    )
    metadata.update(extra)
    return metadata


def _table_metadata_extra(
    table: SceneBuildResult | None,
    obstacle_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if table is None:
        return {}
    table_cuboids = table.scene.get("cuboid", {})
    table_record = next(iter(table_cuboids.values()))
    return {
        "table_included": True,
        "table_top_z": table.metadata.get("table_top_z"),
        "table_dims": table_record.get("dims"),
        "table_pose": table_record.get("pose"),
        "n_obstacles": len(obstacle_records) + len(table_cuboids),
    }
