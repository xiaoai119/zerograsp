#!/usr/bin/env python3
"""Visualize M5 multi-view RGB-D fusion inputs and cuRobo mapper ESDF output."""

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

from maniskill_curobo_real.scene_builder import (
    DEFAULT_WORKSPACE_BOUNDS,
    depth_to_meters,
    rgbd_to_base_cloud,
)


DEFAULT_ROOT = Path("maniskill_curobo_real/runs/m5_multiview_curobo_mapper_seed1_20")
DEFAULT_STAGE = "m5_curobo_mapper_multiview_rgbd_esdf"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 9, 17])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--point-stride", type=int, default=12)
    parser.add_argument("--max-points-per-view", type=int, default=5000)
    parser.add_argument("--max-voxels", type=int, default=3500)
    parser.add_argument(
        "--surface-threshold",
        type=float,
        default=0.02,
        help="ESDF distance threshold for blue near-surface visualization voxels.",
    )
    parser.add_argument(
        "--show-target",
        action="store_true",
        help="Do not remove target mask pixels from the input point-cloud visualization.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "visualization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        record = visualize_seed(
            root=root,
            stage=str(args.stage),
            seed=int(seed),
            output_dir=output_dir,
            point_stride=max(1, int(args.point_stride)),
            max_points_per_view=max(1, int(args.max_points_per_view)),
            max_voxels=max(1, int(args.max_voxels)),
            surface_threshold=float(args.surface_threshold),
            exclude_target=not bool(args.show_target),
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
    summary_path = output_dir / "m5_multiview_visualization_summary.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "n": len(records)}, ensure_ascii=False, indent=2))
    return 0


def visualize_seed(
    *,
    root: Path,
    stage: str,
    seed: int,
    output_dir: Path,
    point_stride: int,
    max_points_per_view: int,
    max_voxels: int,
    surface_threshold: float,
    exclude_target: bool,
) -> dict:
    seed_name = f"seed{seed:03d}"
    input_dir = root / "multiview_inputs" / seed_name
    scene_dir = root / stage / seed_name / "real_scene"
    scene_path = scene_dir / "curobo_scene_voxel.npz"
    metadata_path = scene_dir / "curobo_scene_metadata.json"
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    views = load_view_clouds(
        input_dir,
        point_stride=point_stride,
        max_points_per_view=max_points_per_view,
        exclude_target=exclude_target,
    )
    voxel = load_voxel_scene(scene_path, surface_threshold=surface_threshold)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    output_path = output_dir / f"{seed_name}_m5_multiview_fusion_debug.png"
    render_seed_panel(
        output_path,
        seed_name=seed_name,
        views=views,
        voxel=voxel,
        metadata=metadata,
        max_voxels=max_voxels,
    )
    return {
        "seed": seed,
        "output": str(output_path),
        "scene": str(scene_path),
        "n_views": len(views),
        "input_points_shown": int(sum(view["points"].shape[0] for view in views)),
        "feature_min": float(np.nanmin(voxel["sdf"])),
        "feature_max": float(np.nanmax(voxel["sdf"])),
        "occupied_voxels": int(voxel["occupied_count"]),
        "surface_band_voxels": int(voxel["surface_count"]),
        "surface_threshold_m": float(surface_threshold),
        "metadata_occupied_voxels": int(metadata.get("occupied_voxels", -1)),
        "metadata_surface_band_voxels": int(metadata.get("surface_band_voxels", -1)),
        "all_default_esdf": bool(np.all(voxel["sdf"] >= 9999.0)),
    }


def load_view_clouds(
    input_dir: Path,
    *,
    point_stride: int,
    max_points_per_view: int,
    exclude_target: bool,
) -> list[dict]:
    views = []
    for metadata_path in sorted(input_dir.glob("view_*/view_metadata.json")):
        view_dir = metadata_path.parent
        rgbd_path = view_dir / "rgbd.npz"
        camera_path = view_dir / "camera.json"
        rgb_path = view_dir / "rgb.png"
        if not rgbd_path.is_file() or not camera_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        with np.load(rgbd_path) as payload:
            rgb = np.asarray(payload["rgb"], dtype=np.uint8)
            depth = depth_to_meters(np.asarray(payload["depth"], dtype=np.float32))
            mask = np.asarray(payload["mask"], dtype=np.uint8)
            cam_k = np.asarray(payload["cam_K"], dtype=np.float64)
        target_labels = {
            int(record["label"])
            for record in camera.get("objects", [])
            if bool(record.get("is_task_target", False))
        }
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 2.5)
        if exclude_target and target_labels:
            valid &= ~np.isin(mask, list(target_labels))
        sampled = np.zeros_like(valid, dtype=bool)
        sampled[::point_stride, ::point_stride] = True
        valid &= sampled
        points = rgbd_to_base_cloud(
            depth_m=depth,
            camera_matrix=cam_k,
            camera_model_matrix=np.asarray(metadata["world_from_camera"], dtype=np.float64),
            world_from_base_matrix=np.asarray(metadata["world_from_base"], dtype=np.float64),
        )
        valid_flat = valid.reshape(-1)
        points = points[valid_flat]
        points = points[in_workspace(points)]
        if points.shape[0] > max_points_per_view:
            step = int(np.ceil(points.shape[0] / max_points_per_view))
            points = points[::step]
        rgb_image = Image.open(rgb_path) if rgb_path.is_file() else Image.fromarray(rgb)
        views.append(
            {
                "name": metadata.get("view", {}).get("name", view_dir.name),
                "role": metadata.get("view", {}).get("role", ""),
                "rgb": rgb_image,
                "points": points.astype(np.float32, copy=False),
                "valid_pixels": int(np.count_nonzero(valid)),
                "target_labels": sorted(target_labels),
                "eye": metadata.get("view", {}).get("eye"),
                "target": metadata.get("view", {}).get("target"),
            }
        )
    return views


def in_workspace(points: np.ndarray) -> np.ndarray:
    bounds = np.asarray(DEFAULT_WORKSPACE_BOUNDS, dtype=np.float64)
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= bounds[0, 0])
        & (points[:, 0] <= bounds[0, 1])
        & (points[:, 1] >= bounds[1, 0])
        & (points[:, 1] <= bounds[1, 1])
        & (points[:, 2] >= bounds[2, 0])
        & (points[:, 2] <= bounds[2, 1])
    )


def load_voxel_scene(scene_path: Path, *, surface_threshold: float) -> dict:
    payload = np.load(scene_path)
    sdf = payload["feature_tensor"].astype(np.float32)
    center = payload["voxel_center"].astype(np.float64)
    dims = payload["voxel_dims"].astype(np.float64)
    voxel_size = float(payload["voxel_size"])
    x, y, z = voxel_axes(center, dims, voxel_size, sdf.shape)
    occupied_idx = np.argwhere(sdf <= 0.0)
    surface_idx = np.argwhere(sdf <= float(surface_threshold))
    return {
        "payload": payload,
        "sdf": sdf,
        "center": center,
        "dims": dims,
        "voxel_size": voxel_size,
        "surface_threshold": float(surface_threshold),
        "axes": (x, y, z),
        "occupied_idx": occupied_idx,
        "surface_idx": surface_idx,
        "occupied_count": int(occupied_idx.shape[0]),
        "surface_count": int(surface_idx.shape[0]),
    }


def voxel_axes(
    center: np.ndarray,
    dims: np.ndarray,
    voxel_size: float,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = []
    for axis, count in enumerate(shape):
        values = center[axis] + (np.arange(count, dtype=np.float64) - (count - 1) / 2.0) * voxel_size
        axes.append(values)
    return axes[0], axes[1], axes[2]


def idx_to_xyz(indices: np.ndarray, axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    x, y, z = axes
    return np.column_stack((x[indices[:, 0]], y[indices[:, 1]], z[indices[:, 2]]))


def subsample_xyz(xyz: np.ndarray, limit: int) -> np.ndarray:
    if xyz.shape[0] <= limit:
        return xyz
    step = int(np.ceil(xyz.shape[0] / limit))
    return xyz[::step]


def render_seed_panel(
    output_path: Path,
    *,
    seed_name: str,
    views: list[dict],
    voxel: dict,
    metadata: dict,
    max_voxels: int,
) -> None:
    fig = plt.figure(figsize=(20, 12), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(0.88, 1.25, 0.55))
    for i in range(3):
        ax = fig.add_subplot(grid[0, i])
        if i < len(views):
            img = views[i]["rgb"]
            ax.imshow(img)
            ax.set_title(f"{views[i]['name']}\n{views[i]['role']}")
        else:
            ax.set_title("missing view")
        ax.axis("off")

    ax_cloud = fig.add_subplot(grid[1, 0], projection="3d")
    draw_table(ax_cloud, voxel["payload"])
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for i, view in enumerate(views):
        pts = view["points"]
        if pts.size:
            ax_cloud.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=1.0,
                color=colors[i % len(colors)],
                alpha=0.42,
                label=view["name"],
            )
    setup_axes(ax_cloud, title="Input RGB-D points in robot base")
    ax_cloud.legend(loc="upper right", fontsize=7)

    ax_voxel = fig.add_subplot(grid[1, 1], projection="3d")
    draw_table(ax_voxel, voxel["payload"])
    occupied_xyz = idx_to_xyz(voxel["occupied_idx"], voxel["axes"])
    surface_xyz = idx_to_xyz(voxel["surface_idx"], voxel["axes"])
    if surface_xyz.shape[0] and occupied_xyz.shape[0] == 0:
        draw_voxels(ax_voxel, subsample_xyz(surface_xyz, max_voxels), voxel["voxel_size"], color=(0.1, 0.35, 1.0, 0.22))
    else:
        draw_voxels(ax_voxel, subsample_xyz(surface_xyz, max_voxels), voxel["voxel_size"], color=(0.15, 0.55, 1.0, 0.14))
        draw_voxels(ax_voxel, subsample_xyz(occupied_xyz, max_voxels), voxel["voxel_size"], color=(1.0, 0.1, 0.1, 0.35))
    setup_axes(ax_voxel, title="cuRobo mapper ESDF voxels\nred: occupied, blue: surface band")

    ax_top = fig.add_subplot(grid[1, 2])
    render_topdown_esdf(ax_top, voxel)

    ax_note = fig.add_subplot(grid[2, :])
    ax_note.axis("off")
    notes = [
        f"{seed_name} M5 multi-view fusion debug",
        f"feature min/max: {float(np.nanmin(voxel['sdf'])):.4g} / {float(np.nanmax(voxel['sdf'])):.4g}",
        f"occupied voxels: {voxel['occupied_count']} | near-surface voxels <= {voxel['surface_threshold']:.3f} m: {voxel['surface_count']}",
        f"metadata occupied/surface: {metadata.get('occupied_voxels')} / {metadata.get('surface_band_voxels')}",
        f"views: " + ", ".join(f"{v['name']} points={v['points'].shape[0]} valid={v['valid_pixels']}" for v in views),
        "If RGB-D points look normal but ESDF is empty/all 10000, the bug is in mapper integration/export rather than camera capture.",
    ]
    ax_note.text(0.0, 1.0, "\n".join(notes), va="top", ha="left", fontsize=11)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_topdown_esdf(ax: plt.Axes, voxel: dict) -> None:
    sdf = voxel["sdf"]
    x, y, _ = voxel["axes"]
    finite_like = np.where(sdf >= 9999.0, np.nan, sdf)
    if np.isfinite(finite_like).any():
        top = np.nanmin(finite_like, axis=2)
    else:
        top = np.full(sdf.shape[:2], np.nan, dtype=np.float32)
    image = ax.imshow(
        np.clip(top.T, -0.03, 0.12),
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        aspect="equal",
        cmap="coolwarm",
        vmin=-0.03,
        vmax=0.12,
        interpolation="nearest",
    )
    if np.isfinite(top).any() and float(np.nanmin(top)) <= 0.0 <= float(np.nanmax(top)):
        ax.contour(x, y, top.T, levels=[0.0], colors=["black"], linewidths=0.8)
    ax.set_title("Top-down min ESDF over z")
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    plt.colorbar(image, ax=ax, fraction=0.045, pad=0.03)


def draw_table(ax: plt.Axes, payload: np.lib.npyio.NpzFile) -> None:
    table_pose = payload["table_pose"].astype(np.float64)
    table_dims = payload["table_dims"].astype(np.float64)
    top_z = float(table_pose[2] + table_dims[2] / 2.0)
    bounds = np.asarray(DEFAULT_WORKSPACE_BOUNDS, dtype=np.float64)
    xx, yy = np.meshgrid(
        np.linspace(bounds[0, 0], bounds[0, 1], 2),
        np.linspace(bounds[1, 0], bounds[1, 1], 2),
    )
    ax.plot_surface(xx, yy, np.full_like(xx, top_z), color="#8b9098", alpha=0.18, shade=False)


def draw_voxels(ax: plt.Axes, xyz: np.ndarray, voxel_size: float, *, color: tuple[float, float, float, float]) -> None:
    if xyz.size == 0:
        return
    ax.bar3d(
        xyz[:, 0] - voxel_size / 2.0,
        xyz[:, 1] - voxel_size / 2.0,
        xyz[:, 2] - voxel_size / 2.0,
        voxel_size,
        voxel_size,
        voxel_size,
        color=color,
        edgecolor=(0.04, 0.04, 0.04, 0.08),
        linewidth=0.04,
        shade=True,
    )


def setup_axes(ax: plt.Axes, *, title: str) -> None:
    bounds = np.asarray(DEFAULT_WORKSPACE_BOUNDS, dtype=np.float64)
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_zlim(bounds[2, 0], bounds[2, 1])
    extent = bounds[:, 1] - bounds[:, 0]
    ax.set_box_aspect((extent[0], extent[1], extent[2] * 1.8))
    ax.view_init(elev=30, azim=-56)
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.set_zlabel("base z (m)")
    ax.set_title(title)


if __name__ == "__main__":
    raise SystemExit(main())
