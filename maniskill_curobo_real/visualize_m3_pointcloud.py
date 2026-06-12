#!/usr/bin/env python3
"""Visualize M3 point clouds and cuboids before running the full grasp stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from maniskill_curobo_real.scene_builder import (
    DEFAULT_WORKSPACE_BOUNDS,
    OPENCV_TO_SAPIEN_CAMERA,
    build_oracle_instance_cuboid_scene,
    build_pointcloud_cuboid_scene,
    cluster_xy_points,
    estimate_table_top_z,
    points_to_aabb_cuboid,
    rgbd_to_base_cloud,
    transform_points,
    valid_workspace_mask,
)


DEFAULT_CAMERA_EYE = (-0.20, 0.0, 0.27)
DEFAULT_CAMERA_TARGET = (0.05, 0.0, 0.08)
DEFAULT_SEEDS = (1, 9, 19, 50, 99)


PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
    ],
    dtype=np.uint8,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="PickClutterYCB-v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output-root",
        default="maniskill_curobo_real/runs/m3_pointcloud_debug_pickclutter",
    )
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    parser.add_argument("--mask-mode", default="task-target")
    parser.add_argument("--settle-before-export-steps", type=int, default=0)
    parser.add_argument("--max-points-per-view", type=int, default=80000)
    parser.add_argument("--min-instance-points", type=int, default=400)
    parser.add_argument("--oracle-instance-padding", type=float, default=0.0)
    parser.add_argument("--oracle-min-cuboid-dimension", type=float, default=0.005)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        print(f"visualizing seed {seed}", flush=True)
        summary = visualize_seed(args, seed, output_root / f"seed{seed:03d}")
        summaries.append(summary)
        print(
            f"  realistic_clusters={summary['realistic']['n_cluster_cuboids']} "
            f"oracle_instances={summary['oracle']['n_instance_cuboids']}",
            flush=True,
        )
    (output_root / "summary.json").write_text(
        json.dumps({"seeds": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def visualize_seed(args: argparse.Namespace, seed: int, output_dir: Path) -> dict[str, Any]:
    from maniskill_codex.zerograsp_inputs import (
        collect_mask_actor_records,
        extract_zerograsp_input,
    )
    from maniskill_curobo.scripts import execute_curobo_pick as execute

    output_dir.mkdir(parents=True, exist_ok=True)
    env_args = argparse.Namespace(
        env_id=args.env_id,
        seed=seed,
        camera=args.camera,
        width=args.width,
        height=args.height,
        render_width=args.render_width,
        render_height=args.render_height,
        camera_eye=args.camera_eye,
        camera_target=args.camera_target,
        gripper_open=1.0,
    )
    env = execute.build_env(env_args)
    try:
        obs, _ = env.reset(seed=seed)
        if args.settle_before_export_steps > 0:
            obs, _ = execute.settle_environment(
                env,
                steps=int(args.settle_before_export_steps),
                gripper=1.0,
            )
        bundle = extract_zerograsp_input(obs, env, args.camera, mask_mode=args.mask_mode)
        segmentation = raw_segmentation(obs, args.camera)
        camera_model = execute.camera_model_matrix(env, args.camera)
        world_from_base = execute.robot_base_matrix(env)
        all_object_records = collect_mask_actor_records(env, "all-objects")
        task_target_records = collect_mask_actor_records(env, "task-target")
    finally:
        env.close()

    save_input_images(output_dir, bundle.rgb, bundle.mask)
    cloud = rgbd_to_base_cloud(
        depth_m=bundle.depth,
        camera_matrix=bundle.camera_matrix,
        camera_model_matrix=camera_model,
        world_from_base_matrix=world_from_base,
    )
    valid = valid_workspace_mask(cloud, bundle.depth, DEFAULT_WORKSPACE_BOUNDS)
    target_mask = bundle.mask.reshape(-1) > 0
    non_target = valid & ~target_mask
    table_top_z = estimate_table_top_z(cloud[non_target])

    realistic = visualize_realistic_m3(
        output_dir=output_dir / "realistic_m3",
        cloud=cloud,
        valid=valid,
        target_mask=target_mask,
        table_top_z=table_top_z,
        max_points=int(args.max_points_per_view),
        rgb=bundle.rgb,
        camera_matrix=bundle.camera_matrix,
        camera_model_matrix=camera_model,
        world_from_base_matrix=world_from_base,
    )
    oracle = visualize_oracle_instances(
        output_dir=output_dir / "oracle_instances",
        cloud=cloud,
        valid=valid,
        target_mask=target_mask,
        segmentation=segmentation.reshape(-1),
        object_records=all_object_records,
        target_records=task_target_records,
        min_instance_points=int(args.min_instance_points),
        instance_padding=float(args.oracle_instance_padding),
        min_cuboid_dimension=float(args.oracle_min_cuboid_dimension),
        max_points=int(args.max_points_per_view),
        rgb=bundle.rgb,
        camera_matrix=bundle.camera_matrix,
        camera_model_matrix=camera_model,
        world_from_base_matrix=world_from_base,
    )

    scene_result = build_oracle_instance_cuboid_scene(
        depth_m=bundle.depth,
        mask=bundle.mask,
        segmentation=segmentation,
        object_records=all_object_records,
        target_records=task_target_records,
        camera_matrix=bundle.camera_matrix,
        camera_model_matrix=camera_model,
        world_from_base_matrix=world_from_base,
        min_instance_points=int(args.min_instance_points),
        instance_padding=float(args.oracle_instance_padding),
        min_cuboid_dimension=float(args.oracle_min_cuboid_dimension),
    )
    summary = {
        "seed": int(seed),
        "env_id": args.env_id,
        "camera": args.camera,
        "table_top_z": float(table_top_z),
        "valid_points": int(np.count_nonzero(valid)),
        "target_points": int(np.count_nonzero(valid & target_mask)),
        "non_target_points": int(np.count_nonzero(non_target)),
        "realistic": realistic,
        "oracle": oracle,
        "m3_scene_metadata": scene_result.metadata,
    }
    (output_dir / "debug_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def visualize_realistic_m3(
    *,
    output_dir: Path,
    cloud: np.ndarray,
    valid: np.ndarray,
    target_mask: np.ndarray,
    table_top_z: float,
    max_points: int,
    rgb: np.ndarray,
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    non_target = valid & ~target_mask
    obstacle_mask = (
        non_target
        & (cloud[:, 2] > table_top_z + 0.018)
        & (cloud[:, 2] < table_top_z + 0.22)
    )
    obstacle_points = cloud[obstacle_mask]
    clusters = cluster_xy_points(obstacle_points, grid_cell_m=0.025, min_points=120)

    point_sets = [
        PointSet("target", cloud[valid & target_mask], np.array([230, 25, 75], dtype=np.uint8)),
        PointSet("non_target_low", cloud[non_target & ~obstacle_mask], np.array([150, 150, 150], dtype=np.uint8)),
    ]
    cuboids = []
    cluster_records = []
    for index, points in enumerate(clusters[:20]):
        color = PALETTE[(index + 3) % len(PALETTE)]
        point_sets.append(PointSet(f"cluster_{index:02d}", points, color))
        cuboid = points_to_aabb_cuboid(points, padding=0.015)
        if cuboid is None:
            continue
        cuboids.append(CuboidRecord(f"cluster_{index:02d}", cuboid, color))
        cluster_records.append(
            {
                "name": f"cluster_{index:02d}",
                "points": int(points.shape[0]),
                "dims": cuboid["dims"],
                "pose": cuboid["pose"],
            }
        )

    write_colored_ply(output_dir / "points_target_non_target_clusters.ply", point_sets, max_points=max_points)
    write_cuboids_obj(output_dir / "cluster_cuboids.obj", cuboids)
    write_cuboids_json(output_dir / "cluster_cuboids.json", cuboids, extra={"clusters": cluster_records})
    plot_top_view(output_dir / "top_view.png", point_sets, cuboids, title="M3 realistic: target/non-target clusters", max_points=max_points)
    plot_3d_view(output_dir / "view_3d.png", point_sets, cuboids, title="M3 realistic: target/non-target clusters", max_points=max_points)
    plot_camera_overlay(
        output_dir / "camera_overlay.png",
        rgb,
        point_sets,
        cuboids,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
        title="M3 realistic projected on ZeroGrasp RGB",
        max_points=max_points,
    )
    return {
        "output_dir": str(output_dir),
        "obstacle_candidate_points": int(obstacle_points.shape[0]),
        "n_cluster_cuboids": len(cuboids),
        "cluster_cuboids": cluster_records,
    }


def visualize_oracle_instances(
    *,
    output_dir: Path,
    cloud: np.ndarray,
    valid: np.ndarray,
    target_mask: np.ndarray,
    segmentation: np.ndarray,
    object_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    min_instance_points: int,
    instance_padding: float,
    min_cuboid_dimension: float,
    max_points: int,
    rgb: np.ndarray,
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = {int(record["segmentation_id"]) for record in target_records if "segmentation_id" in record}
    object_by_id = {
        int(record["segmentation_id"]): record for record in object_records if "segmentation_id" in record
    }
    point_sets = []
    cuboids = []
    instance_records = []
    for index, segmentation_id in enumerate(sorted(object_by_id)):
        mask = valid & (segmentation == int(segmentation_id))
        points = cloud[mask]
        if points.shape[0] < min_instance_points:
            continue
        is_target = int(segmentation_id) in target_ids or bool(np.count_nonzero(mask & target_mask))
        color = np.array([230, 25, 75], dtype=np.uint8) if is_target else PALETTE[index % len(PALETTE)]
        name = sanitize_name(object_by_id[segmentation_id].get("actor_name") or f"seg_{segmentation_id}")
        label = f"{segmentation_id}_{name}"
        point_sets.append(PointSet(label, points, color))
        cuboid = points_to_aabb_cuboid(
            points,
            padding=float(instance_padding),
            min_dimension=float(min_cuboid_dimension),
        )
        if cuboid is None:
            continue
        cuboids.append(CuboidRecord(label, cuboid, color))
        instance_records.append(
            {
                "segmentation_id": int(segmentation_id),
                "name": name,
                "is_target": bool(is_target),
                "points": int(points.shape[0]),
                "dims": cuboid["dims"],
                "pose": cuboid["pose"],
            }
        )

    write_colored_ply(output_dir / "points_by_instance.ply", point_sets, max_points=max_points)
    write_cuboids_obj(output_dir / "instance_cuboids.obj", cuboids)
    write_cuboids_json(output_dir / "instance_cuboids.json", cuboids, extra={"instances": instance_records})
    plot_top_view(output_dir / "top_view.png", point_sets, cuboids, title="Oracle: ManiSkill instance point clouds", max_points=max_points)
    plot_3d_view(output_dir / "view_3d.png", point_sets, cuboids, title="Oracle: ManiSkill instance point clouds", max_points=max_points)
    plot_camera_overlay(
        output_dir / "camera_overlay.png",
        rgb,
        point_sets,
        cuboids,
        camera_matrix=camera_matrix,
        camera_model_matrix=camera_model_matrix,
        world_from_base_matrix=world_from_base_matrix,
        title="Oracle instances projected on ZeroGrasp RGB",
        max_points=max_points,
    )
    return {
        "output_dir": str(output_dir),
        "n_instance_cuboids": len(cuboids),
        "instances": instance_records,
    }


class PointSet:
    def __init__(self, name: str, points: np.ndarray, color: np.ndarray):
        self.name = str(name)
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.color = np.asarray(color, dtype=np.uint8).reshape(3)


class CuboidRecord:
    def __init__(self, name: str, cuboid: dict[str, Any], color: np.ndarray):
        self.name = str(name)
        self.dims = np.asarray(cuboid["dims"], dtype=np.float64).reshape(3)
        self.pose = np.asarray(cuboid["pose"], dtype=np.float64).reshape(7)
        self.color = np.asarray(color, dtype=np.uint8).reshape(3)


def raw_segmentation(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    sensor = obs["sensor_data"][camera_name]
    position_segmentation = to_numpy(sensor["PositionSegmentation"])[0].astype(np.float32)
    return position_segmentation[:, :, 3].astype(np.int32)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def write_colored_ply(path: Path, point_sets: list[PointSet], *, max_points: int) -> None:
    rows = []
    for point_set in point_sets:
        points = downsample_points(point_set.points, max_points=max_points // max(1, len(point_sets)))
        colors = np.repeat(point_set.color.reshape(1, 3), points.shape[0], axis=0)
        rows.append((points, colors))
    total = sum(points.shape[0] for points, _ in rows)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {total}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for points, colors in rows:
            for p, c in zip(points, colors):
                file.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_cuboids_obj(path: Path, cuboids: list[CuboidRecord]) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# Cuboid wireframes. Open together with the PLY point cloud.\n")
        vertex_offset = 1
        for cuboid in cuboids:
            vertices = cuboid_vertices(cuboid)
            file.write(f"o {cuboid.name}\n")
            for vertex in vertices:
                file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b in cuboid_edges():
                file.write(f"l {vertex_offset + a} {vertex_offset + b}\n")
            vertex_offset += 8


def write_cuboids_json(path: Path, cuboids: list[CuboidRecord], *, extra: dict[str, Any]) -> None:
    payload = {
        "cuboids": [
            {
                "name": cuboid.name,
                "dims": cuboid.dims.tolist(),
                "pose": cuboid.pose.tolist(),
                "color_rgb": cuboid.color.tolist(),
            }
            for cuboid in cuboids
        ],
        **extra,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plot_top_view(
    path: Path,
    point_sets: list[PointSet],
    cuboids: list[CuboidRecord],
    *,
    title: str,
    max_points: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    for point_set in point_sets:
        points = downsample_points(point_set.points, max_points=max_points // max(1, len(point_sets)))
        if points.size == 0:
            continue
        color = point_set.color / 255.0
        ax.scatter(points[:, 0], points[:, 1], s=0.4, color=color, label=point_set.name, alpha=0.65)
    for cuboid in cuboids:
        x, y, _ = cuboid.pose[:3]
        dx, dy, _ = cuboid.dims
        rect = Rectangle(
            (x - dx / 2.0, y - dy / 2.0),
            dx,
            dy,
            fill=False,
            edgecolor=cuboid.color / 255.0,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, y, cuboid.name[:18], fontsize=6, color=cuboid.color / 255.0)
    ax.set_title(title)
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.legend(markerscale=8, fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_3d_view(
    path: Path,
    point_sets: list[PointSet],
    cuboids: list[CuboidRecord],
    *,
    title: str,
    max_points: int,
) -> None:
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    for point_set in point_sets:
        points = downsample_points(point_set.points, max_points=max_points // max(1, len(point_sets)))
        if points.size == 0:
            continue
        color = point_set.color / 255.0
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25, color=color, label=point_set.name, alpha=0.55)
    for cuboid in cuboids:
        segments = [(cuboid_vertices(cuboid)[a], cuboid_vertices(cuboid)[b]) for a, b in cuboid_edges()]
        ax.add_collection3d(
            Line3DCollection(
                segments,
                colors=[cuboid.color / 255.0],
                linewidths=1.0,
            )
        )
    ax.set_title(title)
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.set_zlabel("base z (m)")
    ax.view_init(elev=28, azim=-63)
    set_equalish_axes(ax, point_sets)
    ax.legend(markerscale=8, fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_input_images(output_dir: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Save the exact RGB input image and a target-mask overlay for orientation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image = rgb_uint8(rgb)
    Image.fromarray(image).save(output_dir / "zerograsp_rgb.png")

    target = np.asarray(mask).reshape(image.shape[:2]) > 0
    overlay = image.astype(np.float32)
    red = np.array([255.0, 40.0, 80.0], dtype=np.float32)
    overlay[target] = 0.45 * overlay[target] + 0.55 * red
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        output_dir / "zerograsp_mask_overlay.png"
    )


def plot_camera_overlay(
    path: Path,
    rgb: np.ndarray,
    point_sets: list[PointSet],
    cuboids: list[CuboidRecord],
    *,
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    title: str,
    max_points: int,
) -> None:
    """Project base-frame point clouds/cuboids back to the ZeroGrasp RGB view."""

    image = rgb_uint8(rgb)
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ax.imshow(image)
    for point_set in point_sets:
        points = downsample_points(point_set.points, max_points=max_points // max(1, len(point_sets)))
        if points.size == 0:
            continue
        uv, visible = project_base_points_to_pixels(
            points,
            camera_matrix=camera_matrix,
            camera_model_matrix=camera_model_matrix,
            world_from_base_matrix=world_from_base_matrix,
            image_shape=(h, w),
        )
        if not np.any(visible):
            continue
        color = point_set.color / 255.0
        ax.scatter(
            uv[visible, 0],
            uv[visible, 1],
            s=0.7,
            color=color,
            alpha=0.62,
            label=point_set.name,
            linewidths=0,
        )

    for cuboid in cuboids:
        vertices = cuboid_vertices(cuboid)
        uv, visible = project_base_points_to_pixels(
            vertices,
            camera_matrix=camera_matrix,
            camera_model_matrix=camera_model_matrix,
            world_from_base_matrix=world_from_base_matrix,
            image_shape=(h, w),
        )
        segments = []
        for a, b in cuboid_edges():
            if visible[a] and visible[b]:
                segments.append([uv[a], uv[b]])
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=[cuboid.color / 255.0],
                    linewidths=2.0,
                    alpha=0.95,
                )
            )
        center_uv, center_visible = project_base_points_to_pixels(
            cuboid.pose[:3].reshape(1, 3),
            camera_matrix=camera_matrix,
            camera_model_matrix=camera_model_matrix,
            world_from_base_matrix=world_from_base_matrix,
            image_shape=(h, w),
        )
        if bool(center_visible[0]):
            ax.text(
                float(center_uv[0, 0]),
                float(center_uv[0, 1]),
                cuboid.name[:20],
                fontsize=6,
                color=cuboid.color / 255.0,
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.0, "edgecolor": "none"},
            )

    ax.set_title(title)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, markerscale=6, fontsize=6, loc="lower right")
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def project_base_points_to_pixels(
    points_base: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    camera_model_matrix: np.ndarray,
    world_from_base_matrix: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project robot-base points into the OpenCV pixel frame used by ZeroGrasp."""

    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

    points_world = transform_points(np.asarray(world_from_base_matrix), points)
    camera_from_world = np.linalg.inv(np.asarray(camera_model_matrix, dtype=np.float64).reshape(4, 4))
    points_camera = transform_points(camera_from_world, points_world)
    points_cv = points_camera @ OPENCV_TO_SAPIEN_CAMERA.T

    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    z = points_cv[:, 2]
    finite = np.isfinite(points_cv).all(axis=1) & (z > 1e-6)
    uv = np.zeros((points.shape[0], 2), dtype=np.float64)
    uv[:, 0] = k[0, 0] * points_cv[:, 0] / np.where(finite, z, 1.0) + k[0, 2]
    uv[:, 1] = k[1, 1] * points_cv[:, 1] / np.where(finite, z, 1.0) + k[1, 2]
    h, w = image_shape
    in_image = (
        finite
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < float(w))
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < float(h))
    )
    return uv, in_image


def cuboid_vertices(cuboid: CuboidRecord) -> np.ndarray:
    center = cuboid.pose[:3]
    dims = cuboid.dims
    offsets = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    return center.reshape(1, 3) + offsets * dims.reshape(1, 3) / 2.0


def cuboid_edges() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def downsample_points(points: np.ndarray, *, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if max_points <= 0 or pts.shape[0] <= max_points:
        return pts
    indices = np.linspace(0, pts.shape[0] - 1, int(max_points), dtype=np.int64)
    return pts[indices]


def set_equalish_axes(ax: Any, point_sets: list[PointSet]) -> None:
    points = np.concatenate(
        [downsample_points(point_set.points, max_points=5000) for point_set in point_sets if point_set.points.size],
        axis=0,
    )
    if points.size == 0:
        return
    mins = np.nanpercentile(points, 1, axis=0)
    maxs = np.nanpercentile(points, 99, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(center[2] - radius, -0.05), center[2] + radius)


def sanitize_name(value: Any) -> str:
    text = str(value)
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in ("_", "-", ".") else "_")
    return "".join(out)[:80]


def rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr[:, :, :3]
    if arr.size and float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
