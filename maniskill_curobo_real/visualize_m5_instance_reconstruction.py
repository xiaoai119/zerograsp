#!/usr/bin/env python3
"""Visualize M5 instance-aware multi-view reconstruction outputs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from maniskill_curobo_real.multiview_instance_scene import load_multiview_instances
from maniskill_curobo_real.scene_builder import DEFAULT_WORKSPACE_BOUNDS


DEFAULT_ROOT = Path("maniskill_curobo_real/runs/m5_multiview_curobo_mapper_pickclutter_seed2_21")
DEFAULT_STAGE = "m5_multiview_instance_voxel_esdf_no_table"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 10, 16, 21])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points-per-instance", type=int, default=1800)
    parser.add_argument("--max-voxels", type=int, default=4500)
    parser.add_argument("--surface-threshold", type=float, default=0.02)
    parser.add_argument(
        "--show-target",
        action="store_true",
        help="Show the excluded target instance as grey points for debugging.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "visualization_instance_reconstruction" / str(args.stage)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for seed in args.seeds:
        record = visualize_seed(
            root=root,
            stage=str(args.stage),
            seed=int(seed),
            output_dir=output_dir,
            max_points_per_instance=max(1, int(args.max_points_per_instance)),
            max_voxels=max(1, int(args.max_voxels)),
            surface_threshold=float(args.surface_threshold),
            show_target=bool(args.show_target),
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
    summary_path = output_dir / "m5_instance_reconstruction_visualization_summary.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "n": len(records)}, ensure_ascii=False, indent=2))
    return 0


def visualize_seed(
    *,
    root: Path,
    stage: str,
    seed: int,
    output_dir: Path,
    max_points_per_instance: int,
    max_voxels: int,
    surface_threshold: float,
    show_target: bool,
) -> dict:
    seed_name = f"seed{seed:03d}"
    input_dir = root / "multiview_inputs" / seed_name
    scene_dir = root / stage / seed_name / "real_scene"
    metadata_path = scene_dir / "curobo_scene_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    instances, input_metadata = load_multiview_instances(input_root=root / "multiview_inputs", seed=seed)
    views = load_rgb_views(input_dir)

    scene = {}
    voxel = None
    scene_path = scene_dir / "curobo_scene.yml"
    voxel_path = scene_dir / "curobo_scene_voxel.npz"
    if scene_path.is_file():
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
    if voxel_path.is_file():
        voxel = load_voxel(voxel_path, surface_threshold=surface_threshold, max_voxels=max_voxels)

    output_path = output_dir / f"{seed_name}_{stage}_reconstruction.png"
    render_panel(
        output_path=output_path,
        seed_name=seed_name,
        stage=stage,
        views=views,
        instances=instances,
        scene=scene,
        voxel=voxel,
        metadata=metadata,
        input_metadata=input_metadata,
        max_points_per_instance=max_points_per_instance,
        show_target=show_target,
    )
    return {
        "seed": int(seed),
        "stage": stage,
        "output": str(output_path),
        "metadata": str(metadata_path),
        "scene": str(scene_path if scene_path.is_file() else voxel_path),
        "n_views": len(views),
        "n_instances_seen": int(metadata.get("n_instances_seen", 0)),
        "n_pointcloud_obstacles": int(metadata.get("n_pointcloud_obstacles", 0)),
        "table_included": bool(metadata.get("table_included", False)),
        "occupied_voxels": int(metadata.get("occupied_voxels", -1)),
    }


def load_rgb_views(input_dir: Path) -> list[dict]:
    views = []
    for metadata_path in sorted(input_dir.glob("view_*/view_metadata.json")):
        view_dir = metadata_path.parent
        rgb_path = view_dir / "rgb.png"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if rgb_path.is_file():
            image = Image.open(rgb_path)
        else:
            with np.load(view_dir / "rgbd.npz") as payload:
                image = Image.fromarray(np.asarray(payload["rgb"], dtype=np.uint8))
        views.append(
            {
                "name": metadata.get("view", {}).get("name", view_dir.name),
                "role": metadata.get("view", {}).get("role", ""),
                "image": image,
            }
        )
    return views


def load_voxel(scene_path: Path, *, surface_threshold: float, max_voxels: int) -> dict:
    with np.load(scene_path) as payload:
        sdf = payload["feature_tensor"].astype(np.float32)
        center = payload["voxel_center"].astype(np.float64)
        dims = payload["voxel_dims"].astype(np.float64)
        voxel_size = float(payload["voxel_size"])
    axes = voxel_axes(center, voxel_size, sdf.shape)
    occupied_idx = np.argwhere(sdf <= 0.0)
    surface_idx = np.argwhere(sdf <= float(surface_threshold))
    return {
        "xyz_occupied": subsample(idx_to_xyz(occupied_idx, axes), max_voxels),
        "xyz_surface": subsample(idx_to_xyz(surface_idx, axes), max_voxels),
        "occupied_count": int(occupied_idx.shape[0]),
        "surface_count": int(surface_idx.shape[0]),
        "center": center,
        "dims": dims,
        "voxel_size": voxel_size,
    }


def render_panel(
    *,
    output_path: Path,
    seed_name: str,
    stage: str,
    views: list[dict],
    instances,
    scene: dict,
    voxel: dict | None,
    metadata: dict,
    input_metadata: dict,
    max_points_per_instance: int,
    show_target: bool,
) -> None:
    fig = plt.figure(figsize=(22, 12), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(0.85, 1.35, 0.36))
    for i in range(3):
        ax = fig.add_subplot(grid[0, i])
        if i < len(views):
            ax.imshow(views[i]["image"])
            ax.set_title(f"{views[i]['name']}\n{views[i]['role']}")
        ax.set_axis_off()

    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(1, len(instances))))
    ax_cloud = fig.add_subplot(grid[1, 0], projection="3d")
    ax_geom = fig.add_subplot(grid[1, 1], projection="3d")
    ax_top = fig.add_subplot(grid[1, 2])

    for idx, instance in enumerate(instances):
        if instance.is_task_target and not show_target:
            continue
        color = (0.45, 0.45, 0.45, 0.22) if instance.is_task_target else colors[idx % len(colors)]
        pts = subsample(instance.points_base, max_points_per_instance)
        label = f"{instance.actor_name[:18]} ({pts.shape[0]}/{instance.points_base.shape[0]})"
        ax_cloud.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, color=color, label=label)
        ax_top.scatter(pts[:, 0], pts[:, 1], s=2, color=color)

    cuboids = (scene.get("cuboid") or {}) if scene else {}
    for name, cuboid in cuboids.items():
        draw_cuboid_wire(ax_geom, cuboid["pose"], cuboid["dims"], color="darkorange", linewidth=1.4)
        draw_cuboid_wire_2d(ax_top, cuboid["pose"], cuboid["dims"], color="darkorange", linewidth=1.2)

    if voxel is not None:
        occ = voxel["xyz_occupied"]
        surf = voxel["xyz_surface"]
        if occ.size:
            ax_geom.scatter(occ[:, 0], occ[:, 1], occ[:, 2], s=4, c="crimson", alpha=0.5, label="occupied")
            ax_top.scatter(occ[:, 0], occ[:, 1], s=2, c="crimson", alpha=0.35)
        if surf.size:
            ax_geom.scatter(surf[:, 0], surf[:, 1], surf[:, 2], s=2, c="royalblue", alpha=0.18, label="surface band")

    ax_cloud.set_title("Merged instance point clouds\n(target excluded from reconstruction)")
    ax_geom.set_title("Reconstructed collision geometry\nOBB or voxel ESDF, no table")
    ax_top.set_title("Top-down reconstruction")
    for ax in (ax_cloud, ax_geom):
        set_3d_limits(ax)
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        ax.set_zlabel("base z (m)")
    set_2d_limits(ax_top)
    ax_top.set_xlabel("base x (m)")
    ax_top.set_ylabel("base y (m)")
    ax_top.set_aspect("equal", adjustable="box")
    if len(instances) <= 8:
        ax_cloud.legend(fontsize=6, loc="upper left")
    if cuboids or voxel is not None:
        ax_geom.legend(fontsize=7, loc="upper left")

    text_ax = fig.add_subplot(grid[2, :])
    text_ax.axis("off")
    summary = [
        f"{seed_name} {stage}",
        f"views={len(views)} instances_seen={metadata.get('n_instances_seen')} obstacles={metadata.get('n_pointcloud_obstacles')} table_included={metadata.get('table_included')}",
        f"valid_points_accumulated={input_metadata.get('valid_points_accumulated_over_views')} non_target_instance_points={input_metadata.get('non_target_instance_points_accumulated_over_views')}",
    ]
    if voxel is not None:
        summary.append(
            f"occupied_voxels={metadata.get('occupied_voxels')} shown_occ={voxel['xyz_occupied'].shape[0]} surface_voxels={voxel['surface_count']}"
        )
    obstacle_names = [str(item.get("actor_name")) for item in metadata.get("obstacle_records", [])]
    if obstacle_names:
        summary.append("obstacles: " + ", ".join(obstacle_names[:8]))
    text_ax.text(0.01, 0.9, "\n".join(summary), va="top", ha="left", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def draw_cuboid_wire(ax, pose, dims, *, color: str, linewidth: float) -> None:
    corners = cuboid_corners(np.asarray(pose, dtype=np.float64), np.asarray(dims, dtype=np.float64))
    for i, j in cuboid_edges():
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color,
            linewidth=linewidth,
        )


def draw_cuboid_wire_2d(ax, pose, dims, *, color: str, linewidth: float) -> None:
    corners = cuboid_corners(np.asarray(pose, dtype=np.float64), np.asarray(dims, dtype=np.float64))
    top = corners[[4, 5, 7, 6, 4], :]
    ax.plot(top[:, 0], top[:, 1], color=color, linewidth=linewidth)


def cuboid_corners(pose: np.ndarray, dims: np.ndarray) -> np.ndarray:
    center = pose[:3]
    rotation = quat_wxyz_to_matrix(pose[3:7])
    half = 0.5 * dims.reshape(3)
    local = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [-half[0], half[1], half[2]],
            [half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    return local @ rotation.T + center


def cuboid_edges() -> list[tuple[int, int]]:
    return [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def voxel_axes(center: np.ndarray, voxel_size: float, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = []
    for axis, count in enumerate(shape):
        axes.append(center[axis] + (np.arange(count, dtype=np.float64) - (count - 1) / 2.0) * voxel_size)
    return axes[0], axes[1], axes[2]


def idx_to_xyz(indices: np.ndarray, axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    x, y, z = axes
    return np.column_stack((x[indices[:, 0]], y[indices[:, 1]], z[indices[:, 2]]))


def subsample(points: np.ndarray, limit: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] <= limit:
        return pts
    step = int(np.ceil(pts.shape[0] / limit))
    return pts[::step]


def set_3d_limits(ax) -> None:
    bounds = np.asarray(DEFAULT_WORKSPACE_BOUNDS, dtype=np.float64)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    try:
        ax.set_box_aspect((bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0], bounds[2, 1] - bounds[2, 0]))
    except Exception:
        pass


def set_2d_limits(ax) -> None:
    bounds = np.asarray(DEFAULT_WORKSPACE_BOUNDS, dtype=np.float64)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])


if __name__ == "__main__":
    raise SystemExit(main())
