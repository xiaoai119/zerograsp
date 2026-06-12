"""Build cuRobo world collision models without ManiSkill actor collision truth.

The builders in this module deliberately avoid reading ManiSkill actor collision
shapes. They use only data that has a real-robot counterpart: configured table
geometry, RGB-D depth, a target mask, camera intrinsics, and camera/base
calibration. In simulation we still obtain those signals from ManiSkill so the
new world models can be compared against the old actor-truth baseline.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OPENCV_TO_SAPIEN_CAMERA = np.diag([1.0, -1.0, -1.0])

DEFAULT_TABLE_TOP_Z = 0.0
DEFAULT_TABLE_DIMS = (2.41799998, 1.20899999, 0.91964293)
DEFAULT_TABLE_CENTER_XY = (0.49500001, 0.0)
DEFAULT_TABLE_QUAT_WXYZ = (0.70710678, 0.0, 0.0, 0.70710678)
DEFAULT_WORKSPACE_BOUNDS = ((0.20, 0.95), (-0.60, 0.60), (-0.04, 0.45))


@dataclass(frozen=True)
class SceneBuildResult:
    scene: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VoxelSceneBuildResult:
    table_scene: dict[str, Any]
    voxel_center: np.ndarray
    voxel_dims: np.ndarray
    voxel_size: float
    feature_tensor: np.ndarray
    metadata: dict[str, Any]


def build_static_table_scene(
    *,
    table_top_z: float = DEFAULT_TABLE_TOP_Z,
    table_dims: tuple[float, float, float] = DEFAULT_TABLE_DIMS,
    table_center_xy: tuple[float, float] = DEFAULT_TABLE_CENTER_XY,
    table_quat_wxyz: tuple[float, float, float, float] = DEFAULT_TABLE_QUAT_WXYZ,
    source: str = "m1_static_table",
) -> SceneBuildResult:
    """Return a minimal cuRobo scene with only a configured table cuboid."""

    dims = np.asarray(table_dims, dtype=np.float64).reshape(3)
    pose = [
        float(table_center_xy[0]),
        float(table_center_xy[1]),
        float(table_top_z - dims[2] / 2.0),
        *[float(v) for v in table_quat_wxyz],
    ]
    scene = {
        "cuboid": {
            "real_table_static": {
                "dims": _round_list(dims),
                "pose": _round_list(pose),
            }
        }
    }
    metadata = {
        "source": source,
        "table_top_z": float(table_top_z),
        "table_dims": _round_list(dims),
        "n_obstacles": 1,
        "obstacles": ["real_table_static"],
        "uses_actor_collision_truth": False,
    }
    return SceneBuildResult(scene=scene, metadata=metadata)


def build_mask_exclusion_scene(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
) -> SceneBuildResult:
    """Build an M2 scene: table from depth, target cloud excluded, no object cuboids."""

    cloud = rgbd_to_base_cloud(
        depth_m=depth_m,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
    )
    valid = valid_workspace_mask(cloud, depth_m, workspace_bounds)
    target = np.asarray(mask) > 0
    non_target = valid & ~target.reshape(-1)
    table_top_z = estimate_table_top_z(cloud[non_target])
    result = build_static_table_scene(
        table_top_z=table_top_z,
        source="m2_mask_exclusion_table_from_depth",
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "valid_points": int(np.count_nonzero(valid)),
            "target_points": int(np.count_nonzero(valid & target.reshape(-1))),
            "non_target_points": int(np.count_nonzero(non_target)),
            "workspace_bounds": workspace_bounds,
            "target_exclusion_applied": True,
        }
    )
    return SceneBuildResult(scene=result.scene, metadata=metadata)


def build_pointcloud_cuboid_scene(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
    obstacle_min_height_above_table: float = 0.018,
    obstacle_max_height_above_table: float = 0.22,
    obstacle_min_points: int = 120,
    obstacle_padding: float = 0.015,
    grid_cell_m: float = 0.025,
    max_obstacles: int = 20,
) -> SceneBuildResult:
    """Build an M3 scene with a table cuboid and non-target point-cloud cuboids."""

    cloud = rgbd_to_base_cloud(
        depth_m=depth_m,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
    )
    valid = valid_workspace_mask(cloud, depth_m, workspace_bounds)
    target = np.asarray(mask) > 0
    non_target = valid & ~target.reshape(-1)
    table_top_z = estimate_table_top_z(cloud[non_target])

    table = build_static_table_scene(
        table_top_z=table_top_z,
        source="m3_pointcloud_cuboids",
    )
    cuboids = dict(table.scene["cuboid"])
    obstacle_points = cloud[
        non_target
        & (cloud[:, 2] > table_top_z + float(obstacle_min_height_above_table))
        & (cloud[:, 2] < table_top_z + float(obstacle_max_height_above_table))
    ]
    clusters = cluster_xy_points(
        obstacle_points,
        grid_cell_m=grid_cell_m,
        min_points=obstacle_min_points,
    )
    obstacle_records: list[dict[str, Any]] = []
    for index, points in enumerate(clusters[:max_obstacles]):
        obstacle = points_to_aabb_cuboid(points, padding=obstacle_padding)
        if obstacle is None:
            continue
        name = f"real_obstacle_cluster_{index:02d}"
        cuboids[name] = obstacle
        obstacle_records.append(
            {
                "name": name,
                "points": int(points.shape[0]),
                "dims": obstacle["dims"],
                "pose": obstacle["pose"],
            }
        )

    scene = {"cuboid": cuboids}
    metadata = dict(table.metadata)
    metadata.update(
        {
            "source": "m3_pointcloud_cuboids",
            "valid_points": int(np.count_nonzero(valid)),
            "target_points": int(np.count_nonzero(valid & target.reshape(-1))),
            "non_target_points": int(np.count_nonzero(non_target)),
            "obstacle_candidate_points": int(obstacle_points.shape[0]),
            "obstacle_min_height_above_table": float(obstacle_min_height_above_table),
            "obstacle_max_height_above_table": float(obstacle_max_height_above_table),
            "n_pointcloud_obstacles": len(obstacle_records),
            "n_obstacles": len(cuboids),
            "obstacle_records": obstacle_records,
            "workspace_bounds": workspace_bounds,
            "target_exclusion_applied": True,
            "uses_actor_collision_truth": False,
        }
    )
    return SceneBuildResult(scene=scene, metadata=metadata)


def build_oracle_instance_cuboid_scene(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    segmentation: np.ndarray,
    object_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
    min_instance_points: int = 400,
    instance_padding: float = 0.0,
    min_cuboid_dimension: float = 0.01,
    max_obstacles: int = 50,
) -> SceneBuildResult:
    """Build an M3 scene from per-object segmentation ids.

    This is an oracle simulation-only stage: it avoids ManiSkill collision
    meshes, but it still relies on ManiSkill's perfect per-object segmentation.
    The target object is excluded from the collision world so cuRobo can plan
    into the grasp.
    """

    cloud = rgbd_to_base_cloud(
        depth_m=depth_m,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
    )
    valid = valid_workspace_mask(cloud, depth_m, workspace_bounds)
    target = np.asarray(mask) > 0
    target_flat = target.reshape(-1)
    segmentation_flat = np.asarray(segmentation, dtype=np.int32).reshape(-1)
    target_ids = {
        int(record["segmentation_id"])
        for record in target_records
        if "segmentation_id" in record
    }
    object_by_id = {
        int(record["segmentation_id"]): record
        for record in object_records
        if "segmentation_id" in record
    }

    non_target = valid & ~target_flat
    table_top_z = estimate_table_top_z(cloud[non_target])
    table = build_static_table_scene(
        table_top_z=table_top_z,
        source="m3_oracle_instance_cuboids",
    )
    cuboids = dict(table.scene["cuboid"])

    obstacle_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    obstacle_candidates: list[tuple[int, dict[str, Any], np.ndarray]] = []
    for segmentation_id, record in sorted(object_by_id.items()):
        instance_mask = valid & (segmentation_flat == int(segmentation_id))
        points = cloud[instance_mask]
        is_target = int(segmentation_id) in target_ids or bool(
            np.count_nonzero(instance_mask & target_flat)
        )
        if is_target:
            skipped_records.append(
                {
                    "segmentation_id": int(segmentation_id),
                    "actor_name": record.get("actor_name"),
                    "reason": "target_excluded",
                    "points": int(points.shape[0]),
                }
            )
            continue
        if points.shape[0] < int(min_instance_points):
            skipped_records.append(
                {
                    "segmentation_id": int(segmentation_id),
                    "actor_name": record.get("actor_name"),
                    "reason": "too_few_points",
                    "points": int(points.shape[0]),
                }
            )
            continue
        obstacle_candidates.append((int(segmentation_id), record, points))

    obstacle_candidates.sort(key=lambda item: item[2].shape[0], reverse=True)
    for index, (segmentation_id, record, points) in enumerate(obstacle_candidates[:max_obstacles]):
        obstacle = points_to_aabb_cuboid(
            points,
            padding=float(instance_padding),
            min_dimension=float(min_cuboid_dimension),
        )
        if obstacle is None:
            continue
        actor_name = str(record.get("actor_name") or f"seg_{segmentation_id}")
        name = f"oracle_obstacle_{index:02d}_{_safe_name(actor_name)}"
        cuboids[name] = obstacle
        obstacle_records.append(
            {
                "name": name,
                "segmentation_id": int(segmentation_id),
                "actor_name": actor_name,
                "points": int(points.shape[0]),
                "dims": obstacle["dims"],
                "pose": obstacle["pose"],
            }
        )

    scene = {"cuboid": cuboids}
    metadata = dict(table.metadata)
    metadata.update(
        {
            "source": "m3_oracle_instance_cuboids",
            "valid_points": int(np.count_nonzero(valid)),
            "target_points": int(np.count_nonzero(valid & target_flat)),
            "non_target_points": int(np.count_nonzero(non_target)),
            "n_object_records": len(object_by_id),
            "n_pointcloud_obstacles": len(obstacle_records),
            "n_obstacles": len(cuboids),
            "instance_padding": float(instance_padding),
            "min_instance_points": int(min_instance_points),
            "min_cuboid_dimension": float(min_cuboid_dimension),
            "max_obstacles": int(max_obstacles),
            "obstacle_records": obstacle_records,
            "skipped_instance_records": skipped_records,
            "workspace_bounds": workspace_bounds,
            "target_exclusion_applied": True,
            "uses_actor_collision_truth": False,
            "uses_oracle_segmentation_truth": True,
        }
    )
    return SceneBuildResult(scene=scene, metadata=metadata)


def build_oracle_instance_obb_scene(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    segmentation: np.ndarray,
    object_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
    min_instance_points: int = 400,
    instance_padding: float = 0.0,
    min_cuboid_dimension: float = 0.01,
    max_obstacles: int = 50,
) -> SceneBuildResult:
    """Build M4-A with one tabletop yaw OBB per non-target instance."""

    context = _oracle_instance_context(
        depth_m=depth_m,
        mask=mask,
        segmentation=segmentation,
        object_records=object_records,
        target_records=target_records,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
        workspace_bounds=workspace_bounds,
        min_instance_points=min_instance_points,
        max_obstacles=max_obstacles,
    )
    table = build_static_table_scene(
        table_top_z=context["table_top_z"],
        source="m4a_oracle_instance_yaw_obb",
    )
    cuboids = dict(table.scene["cuboid"])
    obstacle_records: list[dict[str, Any]] = []
    for index, (segmentation_id, record, points) in enumerate(context["candidates"]):
        obstacle = points_to_yaw_obb_cuboid(
            points,
            padding=float(instance_padding),
            min_dimension=float(min_cuboid_dimension),
        )
        if obstacle is None:
            continue
        actor_name = str(record.get("actor_name") or f"seg_{segmentation_id}")
        name = f"oracle_obb_{index:02d}_{_safe_name(actor_name)}"
        cuboids[name] = {
            "dims": obstacle["dims"],
            "pose": obstacle["pose"],
        }
        obstacle_records.append(
            {
                "name": name,
                "segmentation_id": int(segmentation_id),
                "actor_name": actor_name,
                "points": int(points.shape[0]),
                "dims": obstacle["dims"],
                "pose": obstacle["pose"],
                "yaw_rad": obstacle["yaw_rad"],
            }
        )

    metadata = _oracle_scene_metadata(
        context=context,
        table_metadata=table.metadata,
        source="m4a_oracle_instance_yaw_obb",
        obstacle_records=obstacle_records,
        extra={
            "instance_padding": float(instance_padding),
            "min_cuboid_dimension": float(min_cuboid_dimension),
            "geometry_type": "yaw_obb",
        },
    )
    return SceneBuildResult(scene={"cuboid": cuboids}, metadata=metadata)


def build_oracle_instance_mesh_scene(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    segmentation: np.ndarray,
    object_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
    min_instance_points: int = 400,
    max_obstacles: int = 50,
    max_points_per_instance: int = 2500,
    fallback_padding: float = 0.0,
    min_cuboid_dimension: float = 0.01,
) -> SceneBuildResult:
    """Build M4-B with one closed convex-hull mesh per non-target instance."""

    context = _oracle_instance_context(
        depth_m=depth_m,
        mask=mask,
        segmentation=segmentation,
        object_records=object_records,
        target_records=target_records,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
        workspace_bounds=workspace_bounds,
        min_instance_points=min_instance_points,
        max_obstacles=max_obstacles,
    )
    table = build_static_table_scene(
        table_top_z=context["table_top_z"],
        source="m4b_oracle_instance_convex_mesh",
    )
    cuboids = dict(table.scene["cuboid"])
    meshes: dict[str, Any] = {}
    obstacle_records: list[dict[str, Any]] = []
    fallback_count = 0
    for index, (segmentation_id, record, points) in enumerate(context["candidates"]):
        actor_name = str(record.get("actor_name") or f"seg_{segmentation_id}")
        name = f"oracle_mesh_{index:02d}_{_safe_name(actor_name)}"
        mesh = points_to_convex_hull_mesh(
            points,
            max_points=int(max_points_per_instance),
        )
        if mesh is None:
            obstacle = points_to_yaw_obb_cuboid(
                points,
                padding=float(fallback_padding),
                min_dimension=float(min_cuboid_dimension),
            )
            if obstacle is None:
                continue
            fallback_name = f"{name}_obb_fallback"
            cuboids[fallback_name] = {
                "dims": obstacle["dims"],
                "pose": obstacle["pose"],
            }
            fallback_count += 1
            obstacle_records.append(
                {
                    "name": fallback_name,
                    "segmentation_id": int(segmentation_id),
                    "actor_name": actor_name,
                    "points": int(points.shape[0]),
                    "geometry_type": "yaw_obb_fallback",
                    "dims": obstacle["dims"],
                    "pose": obstacle["pose"],
                }
            )
            continue
        meshes[name] = mesh
        obstacle_records.append(
            {
                "name": name,
                "segmentation_id": int(segmentation_id),
                "actor_name": actor_name,
                "points": int(points.shape[0]),
                "geometry_type": "convex_hull_mesh",
                "vertices": len(mesh["vertices"]),
                "triangles": len(mesh["faces"]) // 3,
            }
        )

    scene: dict[str, Any] = {"cuboid": cuboids}
    if meshes:
        scene["mesh"] = meshes
    metadata = _oracle_scene_metadata(
        context=context,
        table_metadata=table.metadata,
        source="m4b_oracle_instance_convex_mesh",
        obstacle_records=obstacle_records,
        extra={
            "geometry_type": "convex_hull_mesh",
            "n_mesh_obstacles": len(meshes),
            "n_obb_fallbacks": fallback_count,
            "max_points_per_instance": int(max_points_per_instance),
        },
    )
    return SceneBuildResult(scene=scene, metadata=metadata)


def build_oracle_instance_voxel_esdf_scene(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    segmentation: np.ndarray,
    object_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
    min_instance_points: int = 400,
    max_obstacles: int = 50,
    max_points_per_instance: int = 1500,
    voxel_size: float = 0.01,
    dilation_voxels: int = 0,
) -> VoxelSceneBuildResult:
    """Build M4-C as one signed-distance voxel grid plus a table cuboid."""

    from scipy.ndimage import binary_dilation, distance_transform_edt

    context = _oracle_instance_context(
        depth_m=depth_m,
        mask=mask,
        segmentation=segmentation,
        object_records=object_records,
        target_records=target_records,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
        workspace_bounds=workspace_bounds,
        min_instance_points=min_instance_points,
        max_obstacles=max_obstacles,
    )
    table = build_static_table_scene(
        table_top_z=context["table_top_z"],
        source="m4c_oracle_instance_voxel_esdf",
    )
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
    occupied = np.zeros(tuple(int(v) for v in counts), dtype=bool)
    obstacle_records: list[dict[str, Any]] = []
    fallback_count = 0
    for index, (segmentation_id, record, points) in enumerate(context["candidates"]):
        sampled = deterministic_downsample(points, int(max_points_per_instance))
        filled, used_fallback = _voxelize_closed_instance(
            occupied=occupied,
            axes=axes,
            points=sampled,
            voxel_size=voxel_size,
        )
        if filled == 0:
            continue
        fallback_count += int(used_fallback)
        actor_name = str(record.get("actor_name") or f"seg_{segmentation_id}")
        obstacle_records.append(
            {
                "name": f"oracle_voxel_{index:02d}_{_safe_name(actor_name)}",
                "segmentation_id": int(segmentation_id),
                "actor_name": actor_name,
                "points": int(points.shape[0]),
                "occupied_voxels_added": int(filled),
                "voxelization": "yaw_obb_fallback" if used_fallback else "convex_hull",
            }
        )

    if int(dilation_voxels) > 0:
        occupied = binary_dilation(occupied, iterations=int(dilation_voxels))
    outside = distance_transform_edt(~occupied, sampling=voxel_size)
    inside = distance_transform_edt(occupied, sampling=voxel_size)
    sdf = outside.astype(np.float32)
    sdf[occupied] = -inside[occupied].astype(np.float32)
    sdf = sdf.astype(np.float16)

    metadata = _oracle_scene_metadata(
        context=context,
        table_metadata=table.metadata,
        source="m4c_oracle_instance_voxel_esdf",
        obstacle_records=obstacle_records,
        extra={
            "geometry_type": "voxel_esdf",
            "voxel_size": voxel_size,
            "voxel_dims": _round_list(dims),
            "voxel_center": _round_list(center),
            "voxel_shape": [int(v) for v in counts],
            "occupied_voxels": int(np.count_nonzero(occupied)),
            "occupied_fraction": float(np.mean(occupied)),
            "dilation_voxels": int(dilation_voxels),
            "n_voxel_fallbacks": fallback_count,
            "max_points_per_instance": int(max_points_per_instance),
        },
    )
    return VoxelSceneBuildResult(
        table_scene=table.scene,
        voxel_center=center,
        voxel_dims=dims,
        voxel_size=voxel_size,
        feature_tensor=sdf,
        metadata=metadata,
    )


def write_scene_result(
    result: SceneBuildResult,
    scene_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> None:
    scene_out = Path(scene_path).expanduser().resolve()
    meta_out = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else scene_out.with_name("curobo_scene_metadata.json")
    )
    scene_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    scene_out.write_text(json.dumps(result.scene, indent=2) + "\n", encoding="utf-8")
    meta_out.write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_voxel_scene_result(
    result: VoxelSceneBuildResult,
    scene_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> None:
    scene_out = Path(scene_path).expanduser().resolve()
    meta_out = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else scene_out.with_name("curobo_scene_metadata.json")
    )
    scene_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    table = next(iter(result.table_scene["cuboid"].values()))
    np.savez_compressed(
        scene_out,
        feature_tensor=result.feature_tensor,
        voxel_center=result.voxel_center.astype(np.float64),
        voxel_dims=result.voxel_dims.astype(np.float64),
        voxel_size=np.asarray(result.voxel_size, dtype=np.float64),
        table_pose=np.asarray(table["pose"], dtype=np.float64),
        table_dims=np.asarray(table["dims"], dtype=np.float64),
    )
    meta_out.write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def rgbd_to_base_cloud(
    *,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
) -> np.ndarray:
    """Back-project OpenCV depth to robot-base points."""

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
    world_from_camera = np.asarray(camera_model_matrix, dtype=np.float64).reshape(4, 4)
    base_from_world = np.linalg.inv(
        np.asarray(world_from_base_matrix, dtype=np.float64).reshape(4, 4)
    )
    points_world = transform_points(world_from_camera, points_cam)
    return transform_points(base_from_world, points_world)


def valid_workspace_mask(
    cloud_base: np.ndarray,
    depth_m: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    points = np.asarray(cloud_base, dtype=np.float64).reshape(-1, 3)
    depth = depth_to_meters(np.asarray(depth_m, dtype=np.float64)).reshape(-1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(depth)
    valid_depth = (depth > 0.05) & (depth < 2.5)
    in_bounds = (
        (points[:, 0] >= bounds[0][0])
        & (points[:, 0] <= bounds[0][1])
        & (points[:, 1] >= bounds[1][0])
        & (points[:, 1] <= bounds[1][1])
        & (points[:, 2] >= bounds[2][0])
        & (points[:, 2] <= bounds[2][1])
    )
    return finite & valid_depth & in_bounds


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """Convert ManiSkill/ZeroGrasp depth arrays to meters.

    ManiSkill's ``PositionSegmentation`` z channel is currently stored in a
    millimeter-like scale in our ZeroGrasp input bundles. Real RGB-D pipelines
    may already provide meters. This heuristic keeps both cases usable.
    """

    arr = np.asarray(depth, dtype=np.float64)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size and float(np.nanmedian(finite)) > 10.0:
        return arr / 1000.0
    return arr


def estimate_table_top_z(points_base: np.ndarray, fallback: float = DEFAULT_TABLE_TOP_Z) -> float:
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return float(fallback)
    z = points[:, 2]
    z = z[np.isfinite(z)]
    z = z[(z > -0.04) & (z < 0.12)]
    if z.size < 100:
        return float(fallback)
    # The visible table is the dominant low surface. A lower quantile is more
    # stable than a mean when object edges leak into the non-target cloud.
    value = float(np.quantile(z, 0.08))
    return float(np.clip(value, -0.02, 0.04))


def cluster_xy_points(
    points: np.ndarray,
    *,
    grid_cell_m: float,
    min_points: int,
) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < min_points:
        return []
    cells: dict[tuple[int, int], list[int]] = {}
    xy = np.floor(pts[:, :2] / float(grid_cell_m)).astype(np.int64)
    for index, cell in enumerate(map(tuple, xy)):
        cells.setdefault(cell, []).append(index)

    clusters: list[np.ndarray] = []
    visited: set[tuple[int, int]] = set()
    for start in cells:
        if start in visited:
            continue
        queue: deque[tuple[int, int]] = deque([start])
        visited.add(start)
        indices: list[int] = []
        while queue:
            cell = queue.popleft()
            indices.extend(cells[cell])
            cx, cy = cell
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    neighbor = (nx, ny)
                    if neighbor in cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        if len(indices) >= min_points:
            clusters.append(pts[np.asarray(indices, dtype=np.int64)])

    clusters.sort(key=lambda arr: arr.shape[0], reverse=True)
    return clusters


def points_to_aabb_cuboid(
    points: np.ndarray,
    *,
    padding: float,
    min_dimension: float = 0.02,
) -> dict[str, Any] | None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return None
    mins = pts.min(axis=0) - float(padding)
    maxs = pts.max(axis=0) + float(padding)
    dims = np.maximum(
        maxs - mins,
        np.full(3, float(min_dimension), dtype=np.float64),
    )
    center = 0.5 * (mins + maxs)
    return {
        "dims": _round_list(dims),
        "pose": _round_list([*center.tolist(), 1.0, 0.0, 0.0, 0.0]),
    }


def points_to_yaw_obb_cuboid(
    points: np.ndarray,
    *,
    padding: float,
    min_dimension: float = 0.02,
) -> dict[str, Any] | None:
    """Fit a z-upright OBB whose yaw follows the dominant xy point direction."""

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        return points_to_aabb_cuboid(
            pts,
            padding=padding,
            min_dimension=min_dimension,
        )
    xy_mean = pts[:, :2].mean(axis=0)
    centered = pts[:, :2] - xy_mean
    covariance = centered.T @ centered / max(1, pts.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    primary = eigenvectors[:, int(np.argmax(eigenvalues))]
    if primary[0] < 0.0:
        primary = -primary
    secondary = np.asarray([-primary[1], primary[0]], dtype=np.float64)
    axes = np.column_stack([primary, secondary])
    local_xy = centered @ axes
    local_min = local_xy.min(axis=0) - float(padding)
    local_max = local_xy.max(axis=0) + float(padding)
    local_center = 0.5 * (local_min + local_max)
    center_xy = xy_mean + axes @ local_center
    z_min = float(pts[:, 2].min() - padding)
    z_max = float(pts[:, 2].max() + padding)
    dims = np.maximum(
        np.asarray(
            [local_max[0] - local_min[0], local_max[1] - local_min[1], z_max - z_min],
            dtype=np.float64,
        ),
        float(min_dimension),
    )
    yaw = float(math.atan2(primary[1], primary[0]))
    half = 0.5 * yaw
    pose = [
        float(center_xy[0]),
        float(center_xy[1]),
        0.5 * (z_min + z_max),
        math.cos(half),
        0.0,
        0.0,
        math.sin(half),
    ]
    return {
        "dims": _round_list(dims),
        "pose": _round_list(pose),
        "yaw_rad": yaw,
    }


def points_to_convex_hull_mesh(
    points: np.ndarray,
    *,
    max_points: int = 2500,
) -> dict[str, Any] | None:
    """Create a closed convex-hull mesh with vertices expressed in base frame."""

    import trimesh

    pts = deterministic_downsample(points, int(max_points))
    if pts.shape[0] < 4 or np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 3:
        return None
    try:
        hull = trimesh.Trimesh(vertices=pts, process=False).convex_hull
    except Exception:
        return None
    if hull.vertices.shape[0] < 4 or hull.faces.shape[0] < 4:
        return None
    return {
        "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "vertices": np.asarray(hull.vertices, dtype=np.float64).round(8).tolist(),
        "faces": np.asarray(hull.faces, dtype=np.int64).reshape(-1).tolist(),
    }


def deterministic_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if max_points <= 0 or pts.shape[0] <= max_points:
        return pts
    indices = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
    return pts[indices]


def _oracle_instance_context(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    segmentation: np.ndarray,
    object_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    min_instance_points: int,
    max_obstacles: int,
) -> dict[str, Any]:
    cloud = rgbd_to_base_cloud(
        depth_m=depth_m,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
    )
    valid = valid_workspace_mask(cloud, depth_m, workspace_bounds)
    target_flat = (np.asarray(mask) > 0).reshape(-1)
    segmentation_flat = np.asarray(segmentation, dtype=np.int32).reshape(-1)
    target_ids = {
        int(record["segmentation_id"])
        for record in target_records
        if "segmentation_id" in record
    }
    object_by_id = {
        int(record["segmentation_id"]): record
        for record in object_records
        if "segmentation_id" in record
    }
    non_target = valid & ~target_flat
    table_top_z = estimate_table_top_z(cloud[non_target])
    skipped_records: list[dict[str, Any]] = []
    candidates: list[tuple[int, dict[str, Any], np.ndarray]] = []
    for segmentation_id, record in sorted(object_by_id.items()):
        instance_mask = valid & (segmentation_flat == int(segmentation_id))
        points = cloud[instance_mask]
        is_target = int(segmentation_id) in target_ids or bool(
            np.count_nonzero(instance_mask & target_flat)
        )
        if is_target:
            reason = "target_excluded"
        elif points.shape[0] < int(min_instance_points):
            reason = "too_few_points"
        else:
            candidates.append((int(segmentation_id), record, points))
            continue
        skipped_records.append(
            {
                "segmentation_id": int(segmentation_id),
                "actor_name": record.get("actor_name"),
                "reason": reason,
                "points": int(points.shape[0]),
            }
        )
    candidates.sort(key=lambda item: item[2].shape[0], reverse=True)
    return {
        "cloud": cloud,
        "valid": valid,
        "target_flat": target_flat,
        "non_target": non_target,
        "table_top_z": table_top_z,
        "object_by_id": object_by_id,
        "candidates": candidates[: int(max_obstacles)],
        "skipped_records": skipped_records,
        "workspace_bounds": workspace_bounds,
        "min_instance_points": int(min_instance_points),
        "max_obstacles": int(max_obstacles),
    }


def _oracle_scene_metadata(
    *,
    context: dict[str, Any],
    table_metadata: dict[str, Any],
    source: str,
    obstacle_records: list[dict[str, Any]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(table_metadata)
    metadata.update(
        {
            "source": source,
            "valid_points": int(np.count_nonzero(context["valid"])),
            "target_points": int(
                np.count_nonzero(context["valid"] & context["target_flat"])
            ),
            "non_target_points": int(np.count_nonzero(context["non_target"])),
            "n_object_records": len(context["object_by_id"]),
            "n_pointcloud_obstacles": len(obstacle_records),
            "n_obstacles": 1 + len(obstacle_records),
            "min_instance_points": int(context["min_instance_points"]),
            "max_obstacles": int(context["max_obstacles"]),
            "obstacle_records": obstacle_records,
            "skipped_instance_records": context["skipped_records"],
            "workspace_bounds": context["workspace_bounds"],
            "target_exclusion_applied": True,
            "uses_actor_collision_truth": False,
            "uses_oracle_segmentation_truth": True,
        }
    )
    metadata.update(extra)
    return metadata


def _voxelize_closed_instance(
    *,
    occupied: np.ndarray,
    axes: list[np.ndarray],
    points: np.ndarray,
    voxel_size: float,
) -> tuple[int, bool]:
    from scipy.spatial import Delaunay
    from scipy.spatial import QhullError

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    before = int(np.count_nonzero(occupied))
    mins = pts.min(axis=0) - 0.5 * voxel_size
    maxs = pts.max(axis=0) + 0.5 * voxel_size
    slices: list[slice] = []
    for axis_index, values in enumerate(axes):
        lo = max(0, int(np.searchsorted(values, mins[axis_index], side="left")))
        hi = min(values.shape[0], int(np.searchsorted(values, maxs[axis_index], side="right")))
        slices.append(slice(lo, hi))
    if any(part.start >= part.stop for part in slices):
        return 0, False
    gx, gy, gz = np.meshgrid(
        axes[0][slices[0]],
        axes[1][slices[1]],
        axes[2][slices[2]],
        indexing="ij",
    )
    query = np.column_stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)])
    used_fallback = False
    try:
        hull_test = Delaunay(pts, qhull_options="QJ")
        inside = hull_test.find_simplex(query) >= 0
    except (QhullError, ValueError):
        used_fallback = True
        obb = points_to_yaw_obb_cuboid(pts, padding=0.0, min_dimension=voxel_size)
        if obb is None:
            return 0, True
        pose = np.asarray(obb["pose"], dtype=np.float64)
        yaw = 2.0 * math.atan2(pose[6], pose[3])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        local_xy = (query[:, :2] - pose[:2]) @ rotation
        half_dims = 0.5 * np.asarray(obb["dims"], dtype=np.float64)
        inside = (
            (np.abs(local_xy[:, 0]) <= half_dims[0])
            & (np.abs(local_xy[:, 1]) <= half_dims[1])
            & (np.abs(query[:, 2] - pose[2]) <= half_dims[2])
        )
    local_occupied = occupied[tuple(slices)]
    local_occupied |= inside.reshape(local_occupied.shape)
    occupied[tuple(slices)] = local_occupied
    return int(np.count_nonzero(occupied) - before), used_fallback


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return pts @ mat[:3, :3].T + mat[:3, 3]


def _round_list(values: Any, ndigits: int = 8) -> list[float]:
    return [round(float(value), ndigits) for value in np.asarray(values).reshape(-1)]


def _safe_name(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum() or char in {"_", "-"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    text = "".join(cleaned).strip("_")
    return text[:80] or "object"
