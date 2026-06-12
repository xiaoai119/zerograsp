#!/usr/bin/env python3
"""Visualize an M4-C cuRobo voxel ESDF scene saved as NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="Path to curobo_scene_voxel.npz.")
    parser.add_argument("--output", help="Output PNG path.")
    parser.add_argument(
        "--free-distance-max",
        type=float,
        default=0.12,
        help="Maximum positive ESDF distance shown in heatmaps.",
    )
    return parser.parse_args(argv)


def voxel_axes(
    center: np.ndarray,
    dims: np.ndarray,
    voxel_size: float,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = []
    for axis, count in enumerate(shape):
        values = (
            center[axis]
            + (np.arange(count, dtype=np.float64) - (count - 1) / 2.0)
            * voxel_size
        )
        axes.append(values)
    return axes[0], axes[1], axes[2]


def find_rgb(scene_path: Path) -> Path | None:
    seed_dir = scene_path.parent.parent
    candidate = seed_dir / "zg_input" / "rgb.png"
    return candidate if candidate.is_file() else None


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    scene_path = Path(args.scene).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else scene_path.with_name("m4c_esdf_visualization.png")
    )

    payload = np.load(scene_path)
    sdf = payload["feature_tensor"].astype(np.float32)
    center = payload["voxel_center"].astype(np.float64)
    dims = payload["voxel_dims"].astype(np.float64)
    voxel_size = float(payload["voxel_size"])
    table_pose = payload["table_pose"].astype(np.float64)
    occupied = sdf <= 0.0
    x, y, z = voxel_axes(center, dims, voxel_size, sdf.shape)
    bounds = np.column_stack((center - dims / 2.0, center + dims / 2.0))
    occupied_indices = np.argwhere(occupied)
    occupied_xyz = np.column_stack(
        (
            x[occupied_indices[:, 0]],
            y[occupied_indices[:, 1]],
            z[occupied_indices[:, 2]],
        )
    )

    metadata_path = scene_path.with_name("curobo_scene_metadata.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    object_count = int(metadata.get("n_pointcloud_obstacles", 0))
    target_points = int(metadata.get("target_points", 0))

    fig = plt.figure(figsize=(16, 10), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0))

    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    if occupied_xyz.size:
        colors = plt.get_cmap("viridis")(
            Normalize(vmin=float(z.min()), vmax=max(float(z.max()), float(z.min()) + 1e-6))(
                occupied_xyz[:, 2]
            )
        )
        ax_3d.bar3d(
            occupied_xyz[:, 0] - voxel_size / 2.0,
            occupied_xyz[:, 1] - voxel_size / 2.0,
            occupied_xyz[:, 2] - voxel_size / 2.0,
            voxel_size,
            voxel_size,
            voxel_size,
            color=colors,
            edgecolor=(0.08, 0.08, 0.08, 0.22),
            linewidth=0.12,
            shade=True,
        )
    table_z = float(table_pose[2] + payload["table_dims"][2] / 2.0)
    if occupied_xyz.size:
        plot_min = occupied_xyz.min(axis=0) - np.array([0.08, 0.08, 0.015])
        plot_max = occupied_xyz.max(axis=0) + np.array([0.08, 0.08, 0.07])
        plot_min[:2] = np.maximum(plot_min[:2], bounds[:2, 0])
        plot_max[:2] = np.minimum(plot_max[:2], bounds[:2, 1])
        plot_min[2] = min(plot_min[2], table_z - 0.005)
    else:
        plot_min = bounds[:, 0]
        plot_max = bounds[:, 1]
    table_x, table_y = np.meshgrid(
        np.linspace(plot_min[0], plot_max[0], 2),
        np.linspace(plot_min[1], plot_max[1], 2),
    )
    ax_3d.plot_surface(
        table_x,
        table_y,
        np.full_like(table_x, table_z),
        color="#8b9098",
        alpha=0.22,
        shade=False,
    )
    ax_3d.set_title("M4-C occupied voxels used by cuRobo", fontsize=13)
    ax_3d.set_xlabel("base x (m)")
    ax_3d.set_ylabel("base y (m)")
    ax_3d.set_zlabel("base z (m)")
    ax_3d.view_init(elev=31, azim=-56)
    ax_3d.set_xlim(plot_min[0], plot_max[0])
    ax_3d.set_ylim(plot_min[1], plot_max[1])
    ax_3d.set_zlim(plot_min[2], plot_max[2])
    plot_size = np.maximum(plot_max - plot_min, 1e-3)
    ax_3d.set_box_aspect((plot_size[0], plot_size[1], plot_size[2] * 1.8))

    top_sdf = np.min(sdf, axis=2)
    ax_top = fig.add_subplot(grid[0, 1])
    image = ax_top.imshow(
        np.clip(top_sdf.T, -0.03, float(args.free_distance_max)),
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        aspect="equal",
        cmap="coolwarm",
        vmin=-0.03,
        vmax=float(args.free_distance_max),
        interpolation="nearest",
    )
    ax_top.contour(
        x,
        y,
        top_sdf.T,
        levels=[0.0],
        colors=["black"],
        linewidths=0.9,
    )
    ax_top.set_title("Top view: minimum signed distance over z")
    ax_top.set_xlabel("base x (m)")
    ax_top.set_ylabel("base y (m)")
    colorbar = fig.colorbar(image, ax=ax_top, fraction=0.046, pad=0.03)
    colorbar.set_label("ESDF distance (m)")

    occupancy_per_z = occupied.sum(axis=(0, 1))
    slice_index = int(np.argmax(occupancy_per_z))
    slice_z = float(z[slice_index])
    ax_slice = fig.add_subplot(grid[1, 1])
    slice_image = ax_slice.imshow(
        np.clip(sdf[:, :, slice_index].T, -0.03, float(args.free_distance_max)),
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        aspect="equal",
        cmap="coolwarm",
        vmin=-0.03,
        vmax=float(args.free_distance_max),
        interpolation="nearest",
    )
    ax_slice.contour(
        x,
        y,
        sdf[:, :, slice_index].T,
        levels=[0.0],
        colors=["black"],
        linewidths=0.9,
    )
    ax_slice.set_title(f"Horizontal ESDF slice at z={slice_z:.3f} m")
    ax_slice.set_xlabel("base x (m)")
    ax_slice.set_ylabel("base y (m)")
    fig.colorbar(slice_image, ax=ax_slice, fraction=0.046, pad=0.03)

    seed_name = scene_path.parent.parent.name
    fig.suptitle(
        f"{seed_name}: M4-C 10 mm voxel ESDF\n"
        f"{int(occupied.sum())} occupied voxels, {object_count} non-target instances, "
        f"{target_points} target points excluded",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    rgb_path = find_rgb(scene_path)
    summary = {
        "scene": str(scene_path),
        "output": str(output_path),
        "shape": list(sdf.shape),
        "voxel_size_m": voxel_size,
        "occupied_voxels": int(occupied.sum()),
        "horizontal_slice_z_m": slice_z,
        "rgb_reference": str(rgb_path) if rgb_path else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
