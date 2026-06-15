#!/usr/bin/env python3
"""Visualize M4-A yaw OBBs and M4-C voxel ESDF for the same scene."""

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


DEFAULT_ROOT = Path("maniskill_curobo_real/runs/pickclutter_full_depth_m4abc_seed1_200")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--m4a-stage",
        type=str,
        default=None,
        help="Optional M4-A stage directory name under root.",
    )
    parser.add_argument(
        "--m4c-stage",
        type=str,
        default=None,
        help="Optional M4-C stage directory name under root.",
    )
    parser.add_argument("--max-voxels", type=int, default=2500)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    seed_name = f"seed{int(args.seed):03d}"
    m4a_stage = resolve_stage_name(
        root,
        seed_name,
        override=args.m4a_stage,
        candidates=(
            "m4a_oracle_instance_yaw_obb",
            "m4a_zerograsp_reconstruction_yaw_obb",
        ),
        required_relative_path=Path("real_scene/curobo_scene.yml"),
    )
    m4c_stage = resolve_stage_name(
        root,
        seed_name,
        override=args.m4c_stage,
        candidates=(
            "m4c_oracle_instance_voxel_esdf",
            "m4c_zerograsp_reconstruction_voxel_esdf",
        ),
        required_relative_path=Path("real_scene/curobo_scene_voxel.npz"),
    )
    m4a_dir = root / m4a_stage / seed_name
    m4c_dir = root / m4c_stage / seed_name
    m4a_scene = m4a_dir / "real_scene" / "curobo_scene.yml"
    m4a_meta = m4a_dir / "real_scene" / "curobo_scene_metadata.json"
    m4c_scene = m4c_dir / "real_scene" / "curobo_scene_voxel.npz"
    m4c_meta = m4c_dir / "real_scene" / "curobo_scene_metadata.json"
    rgb_path = m4c_dir / "zg_input" / "rgb.png"
    for path in (m4a_scene, m4a_meta, m4c_scene, m4c_meta, rgb_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    comparison_output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "comparison" / f"{seed_name}_m4a_vs_m4c_same_scene.png"
    )
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    m4a_output = comparison_output.with_name(f"{comparison_output.stem}_m4a_only.png")
    m4c_output = comparison_output.with_name(f"{comparison_output.stem}_m4c_only.png")

    m4a_payload = json.loads(m4a_scene.read_text(encoding="utf-8"))
    m4a_metadata = json.loads(m4a_meta.read_text(encoding="utf-8"))
    m4c_metadata = json.loads(m4c_meta.read_text(encoding="utf-8"))
    voxel_payload = np.load(m4c_scene)
    sdf = voxel_payload["feature_tensor"].astype(np.float32)
    occupied = sdf <= 0.0
    center = voxel_payload["voxel_center"].astype(np.float64)
    dims = voxel_payload["voxel_dims"].astype(np.float64)
    voxel_size = float(voxel_payload["voxel_size"])
    x, y, z = voxel_axes(center, dims, voxel_size, sdf.shape)
    occ_idx = np.argwhere(occupied)
    if occ_idx.shape[0] > int(args.max_voxels):
        stride = int(np.ceil(occ_idx.shape[0] / int(args.max_voxels)))
        occ_idx = occ_idx[::stride]
    occ_xyz = np.column_stack((x[occ_idx[:, 0]], y[occ_idx[:, 1]], z[occ_idx[:, 2]]))

    plot_min, plot_max = plot_bounds_from_scene(m4a_payload, occ_xyz, voxel_payload)
    rgb_image = Image.open(rgb_path)
    render_side_by_side(
        comparison_output,
        seed_name=seed_name,
        m4a_payload=m4a_payload,
        voxel_payload=voxel_payload,
        occ_xyz=occ_xyz,
        voxel_size=voxel_size,
        plot_min=plot_min,
        plot_max=plot_max,
        rgb_image=rgb_image,
        m4a_metadata=m4a_metadata,
        m4c_metadata=m4c_metadata,
    )
    render_single_scene(
        m4a_output,
        title=f"{seed_name}: M4-A yaw OBB world",
        drawer=lambda ax: draw_m4a_boxes(ax, m4a_payload),
        voxel_payload=voxel_payload,
        plot_min=plot_min,
        plot_max=plot_max,
        notes=[
            "Orange wireframes = yaw OBB cuboids",
            f"Non-target obstacles: {m4a_metadata.get('n_pointcloud_obstacles')}",
            "Target object is excluded from the world model.",
        ],
    )
    render_single_scene(
        m4c_output,
        title=f"{seed_name}: M4-C voxel ESDF world",
        drawer=lambda ax: draw_m4c_voxels(ax, occ_xyz, voxel_size),
        voxel_payload=voxel_payload,
        plot_min=plot_min,
        plot_max=plot_max,
        notes=[
            "Blue blocks = occupied ESDF voxels",
            f"Non-target obstacles: {m4c_metadata.get('n_pointcloud_obstacles')}",
            f"Occupied voxels: {m4c_metadata.get('occupied_voxels')}",
            f"Voxel size: {voxel_size:.3f} m",
            "Target object is excluded from the world model.",
        ],
    )
    summary = {
        "seed": int(args.seed),
        "m4a_stage": m4a_stage,
        "m4c_stage": m4c_stage,
        "m4a_scene": str(m4a_scene),
        "m4c_scene": str(m4c_scene),
        "comparison_output": str(comparison_output),
        "m4a_output": str(m4a_output),
        "m4c_output": str(m4c_output),
        "shown_voxels": int(occ_idx.shape[0]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def resolve_stage_name(
    root: Path,
    seed_name: str,
    *,
    override: str | None,
    candidates: tuple[str, ...],
    required_relative_path: Path,
) -> str:
    if override:
        candidate_path = root / override / seed_name / required_relative_path
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        return override
    for candidate in candidates:
        candidate_path = root / candidate / seed_name / required_relative_path
        if candidate_path.is_file():
            return candidate
    checked = [str(root / candidate / seed_name / required_relative_path) for candidate in candidates]
    raise FileNotFoundError("Could not find a matching stage file. Checked:\n" + "\n".join(checked))


def setup_scene_axes(
    ax: plt.Axes,
    *,
    plot_min: np.ndarray,
    plot_max: np.ndarray,
    title: str,
) -> None:
    ax.set_xlim(plot_min[0], plot_max[0])
    ax.set_ylim(plot_min[1], plot_max[1])
    ax.set_zlim(plot_min[2], plot_max[2])
    extent = np.maximum(plot_max - plot_min, 1e-4)
    ax.set_box_aspect((extent[0], extent[1], extent[2] * 1.7))
    ax.view_init(elev=30, azim=-56)
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.set_zlabel("base z (m)")
    ax.set_title(title)


def render_side_by_side(
    output: Path,
    *,
    seed_name: str,
    m4a_payload: dict,
    voxel_payload: np.lib.npyio.NpzFile,
    occ_xyz: np.ndarray,
    voxel_size: float,
    plot_min: np.ndarray,
    plot_max: np.ndarray,
    rgb_image: Image.Image,
    m4a_metadata: dict,
    m4c_metadata: dict,
) -> None:
    fig = plt.figure(figsize=(18, 10), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.72))
    m4a_ax = fig.add_subplot(grid[0, 0], projection="3d")
    m4c_ax = fig.add_subplot(grid[0, 1], projection="3d")
    rgb_ax = fig.add_subplot(grid[0, 2])
    m4a_note_ax = fig.add_subplot(grid[1, 0])
    m4c_note_ax = fig.add_subplot(grid[1, 1])
    shared_note_ax = fig.add_subplot(grid[1, 2])

    draw_table(m4a_ax, voxel_payload)
    draw_m4a_boxes(m4a_ax, m4a_payload)
    setup_scene_axes(
        m4a_ax,
        plot_min=plot_min,
        plot_max=plot_max,
        title="M4-A only: yaw OBB cuboids",
    )

    draw_table(m4c_ax, voxel_payload)
    draw_m4c_voxels(m4c_ax, occ_xyz, voxel_size)
    setup_scene_axes(
        m4c_ax,
        plot_min=plot_min,
        plot_max=plot_max,
        title="M4-C only: voxel ESDF occupied cells",
    )

    rgb_ax.imshow(rgb_image)
    rgb_ax.set_title("Same RGB input")
    rgb_ax.axis("off")

    write_note(
        m4a_note_ax,
        [
            f"{seed_name} / M4-A",
            "",
            "Orange wireframes = per-instance yaw OBB cuboids",
            f"Non-target obstacles: {m4a_metadata.get('n_pointcloud_obstacles')}",
            "This is the compact box approximation used by cuRobo.",
        ],
    )
    write_note(
        m4c_note_ax,
        [
            f"{seed_name} / M4-C",
            "",
            "Blue blocks = occupied ESDF voxels",
            f"Non-target obstacles: {m4c_metadata.get('n_pointcloud_obstacles')}",
            f"Occupied voxels: {m4c_metadata.get('occupied_voxels')}",
            f"Voxel size: {voxel_size:.3f} m",
        ],
    )
    write_note(
        shared_note_ax,
        [
            "Both panels use the same scene, same plot bounds, and same camera angle.",
            "",
            "The target object is excluded from the world collision model in both stages.",
            "Gray plane = estimated table top.",
        ],
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_single_scene(
    output: Path,
    *,
    title: str,
    drawer,
    voxel_payload: np.lib.npyio.NpzFile,
    plot_min: np.ndarray,
    plot_max: np.ndarray,
    notes: list[str],
) -> None:
    fig = plt.figure(figsize=(12, 9), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.6, 0.7))
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    note_ax = fig.add_subplot(grid[0, 1])
    draw_table(ax, voxel_payload)
    drawer(ax)
    setup_scene_axes(ax, plot_min=plot_min, plot_max=plot_max, title=title)
    write_note(note_ax, notes)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_note(ax: plt.Axes, lines: list[str]) -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11)


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


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def cuboid_corners(dims: np.ndarray, pose: np.ndarray) -> np.ndarray:
    half = np.asarray(dims, dtype=np.float64).reshape(3) / 2.0
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    rotation = quat_wxyz_to_matrix(pose[3:])
    return local @ rotation.T + pose[:3]


def draw_box_edges(ax: plt.Axes, corners: np.ndarray, *, color: str, linewidth: float, alpha: float) -> None:
    edges = (
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
    )
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def draw_m4a_boxes(ax: plt.Axes, payload: dict) -> None:
    cuboids = payload.get("cuboid", {})
    for name, cuboid in cuboids.items():
        corners = cuboid_corners(np.asarray(cuboid["dims"]), np.asarray(cuboid["pose"]))
        if name == "real_table_static":
            continue
        draw_box_edges(ax, corners, color="#ff8c00", linewidth=1.7, alpha=0.95)


def draw_m4c_voxels(ax: plt.Axes, occ_xyz: np.ndarray, voxel_size: float) -> None:
    if occ_xyz.size == 0:
        return
    ax.bar3d(
        occ_xyz[:, 0] - voxel_size / 2.0,
        occ_xyz[:, 1] - voxel_size / 2.0,
        occ_xyz[:, 2] - voxel_size / 2.0,
        voxel_size,
        voxel_size,
        voxel_size,
        color=(0.12, 0.34, 0.95, 0.22),
        edgecolor=(0.08, 0.10, 0.20, 0.08),
        linewidth=0.08,
        shade=True,
    )


def draw_table(ax: plt.Axes, payload: np.lib.npyio.NpzFile) -> None:
    table_pose = payload["table_pose"].astype(np.float64)
    table_dims = payload["table_dims"].astype(np.float64)
    top_z = float(table_pose[2] + table_dims[2] / 2.0)
    x0, x1 = table_pose[0] - table_dims[0] / 2.0, table_pose[0] + table_dims[0] / 2.0
    y0, y1 = table_pose[1] - table_dims[1] / 2.0, table_pose[1] + table_dims[1] / 2.0
    xx, yy = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
    ax.plot_surface(xx, yy, np.full_like(xx, top_z), color="#8b9098", alpha=0.18, shade=False)


def plot_bounds_from_scene(
    m4a_payload: dict,
    occ_xyz: np.ndarray,
    voxel_payload: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for name, cuboid in m4a_payload.get("cuboid", {}).items():
        if name == "real_table_static":
            continue
        points.append(cuboid_corners(np.asarray(cuboid["dims"]), np.asarray(cuboid["pose"])))
    if occ_xyz.size:
        points.append(occ_xyz)
    if points:
        stacked = np.concatenate(points, axis=0)
        plot_min = stacked.min(axis=0) - np.array([0.08, 0.08, 0.02])
        plot_max = stacked.max(axis=0) + np.array([0.08, 0.08, 0.06])
    else:
        center = voxel_payload["voxel_center"].astype(np.float64)
        dims = voxel_payload["voxel_dims"].astype(np.float64)
        plot_min = center - dims / 2.0
        plot_max = center + dims / 2.0
    table_pose = voxel_payload["table_pose"].astype(np.float64)
    table_dims = voxel_payload["table_dims"].astype(np.float64)
    table_top = float(table_pose[2] + table_dims[2] / 2.0)
    plot_min[2] = min(plot_min[2], table_top - 0.01)
    return plot_min, plot_max


if __name__ == "__main__":
    raise SystemExit(main())
