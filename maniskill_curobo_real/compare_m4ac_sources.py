#!/usr/bin/env python3
"""Compare full-depth M4A/M4C with ZeroGrasp-reconstructed M4A/M4C."""

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

from maniskill_curobo_real.visualize_m4a_m4c_scene import (
    cuboid_corners,
    draw_m4a_boxes,
    draw_m4c_voxels,
    draw_table,
    setup_scene_axes,
    voxel_axes,
    write_note,
)


DEFAULT_FULL_ROOT = Path("maniskill_curobo_real/runs/pickclutter_full_depth_m4abc_seed1_200")
DEFAULT_ZG_ROOT = Path("maniskill_curobo_real/runs/pickclutter_zerograsp_reconstruction_m3_m4abc_seed1_200")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--zg-root", type=Path, default=DEFAULT_ZG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-voxels", type=int, default=2500)
    parser.add_argument("--max-diff-voxels", type=int, default=3000)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    seed_name = f"seed{int(args.seed):03d}"
    full_root = args.full_root.expanduser().resolve()
    zg_root = args.zg_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else zg_root / "comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    full = load_source(
        full_root,
        seed_name,
        m4a_stage="m4a_oracle_instance_yaw_obb",
        m4c_stage="m4c_oracle_instance_voxel_esdf",
        max_voxels=int(args.max_voxels),
    )
    zg = load_source(
        zg_root,
        seed_name,
        m4a_stage="m4a_zerograsp_reconstruction_yaw_obb",
        m4c_stage="m4c_zerograsp_reconstruction_voxel_esdf",
        max_voxels=int(args.max_voxels),
    )

    plot_min, plot_max = combined_bounds([full, zg])
    panel_output = output_dir / f"{seed_name}_full_depth_vs_zg_m4a_m4c.png"
    diff_output = output_dir / f"{seed_name}_full_depth_vs_zg_m4a_m4c_diff.png"
    topdown_output = output_dir / f"{seed_name}_full_depth_vs_zg_m4a_m4c_topdown.png"
    render_panel_compare(panel_output, seed_name=seed_name, full=full, zg=zg, plot_min=plot_min, plot_max=plot_max)
    diff_stats = render_diff_compare(
        diff_output,
        seed_name=seed_name,
        full=full,
        zg=zg,
        plot_min=plot_min,
        plot_max=plot_max,
        max_diff_voxels=int(args.max_diff_voxels),
    )
    render_topdown_compare(
        topdown_output,
        seed_name=seed_name,
        full=full,
        zg=zg,
        plot_min=plot_min,
        plot_max=plot_max,
        max_diff_voxels=int(args.max_diff_voxels),
    )

    summary = {
        "seed": int(args.seed),
        "full_root": str(full_root),
        "zg_root": str(zg_root),
        "panel_output": str(panel_output),
        "diff_output": str(diff_output),
        "topdown_output": str(topdown_output),
        "full_m4a_boxes": int(full["m4a_box_count"]),
        "zg_m4a_boxes": int(zg["m4a_box_count"]),
        "full_m4c_occupied_voxels": int(full["m4c_metadata"].get("occupied_voxels", -1)),
        "zg_m4c_occupied_voxels": int(zg["m4c_metadata"].get("occupied_voxels", -1)),
        **diff_stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_source(
    root: Path,
    seed_name: str,
    *,
    m4a_stage: str,
    m4c_stage: str,
    max_voxels: int,
) -> dict:
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

    m4a_payload = json.loads(m4a_scene.read_text(encoding="utf-8"))
    voxel_payload = np.load(m4c_scene)
    occupied = voxel_payload["feature_tensor"].astype(np.float32) <= 0.0
    x, y, z = voxel_axes(
        voxel_payload["voxel_center"].astype(np.float64),
        voxel_payload["voxel_dims"].astype(np.float64),
        float(voxel_payload["voxel_size"]),
        occupied.shape,
    )
    occ_idx = np.argwhere(occupied)
    occ_idx_shown = subsample_indices(occ_idx, max_voxels)
    occ_xyz = np.column_stack((x[occ_idx_shown[:, 0]], y[occ_idx_shown[:, 1]], z[occ_idx_shown[:, 2]]))
    occ_xyz_all = np.column_stack((x[occ_idx[:, 0]], y[occ_idx[:, 1]], z[occ_idx[:, 2]]))

    cuboids = {
        name: cuboid for name, cuboid in m4a_payload.get("cuboid", {}).items() if name != "real_table_static"
    }
    return {
        "root": root,
        "m4a_stage": m4a_stage,
        "m4c_stage": m4c_stage,
        "m4a_payload": m4a_payload,
        "m4a_metadata": json.loads(m4a_meta.read_text(encoding="utf-8")),
        "m4a_box_count": len(cuboids),
        "m4c_payload": voxel_payload,
        "m4c_metadata": json.loads(m4c_meta.read_text(encoding="utf-8")),
        "occ_xyz": occ_xyz,
        "occ_xyz_all": occ_xyz_all,
        "voxel_size": float(voxel_payload["voxel_size"]),
        "rgb": Image.open(rgb_path),
    }


def subsample_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
    if indices.shape[0] <= max_count:
        return indices
    stride = int(np.ceil(indices.shape[0] / max_count))
    return indices[::stride]


def combined_bounds(sources: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for source in sources:
        for name, cuboid in source["m4a_payload"].get("cuboid", {}).items():
            if name == "real_table_static":
                continue
            points.append(cuboid_corners(np.asarray(cuboid["dims"]), np.asarray(cuboid["pose"])))
        if source["occ_xyz_all"].size:
            points.append(source["occ_xyz_all"])
    if not points:
        center = sources[0]["m4c_payload"]["voxel_center"].astype(np.float64)
        dims = sources[0]["m4c_payload"]["voxel_dims"].astype(np.float64)
        return center - dims / 2.0, center + dims / 2.0
    stacked = np.concatenate(points, axis=0)
    plot_min = stacked.min(axis=0) - np.array([0.08, 0.08, 0.02])
    plot_max = stacked.max(axis=0) + np.array([0.08, 0.08, 0.06])
    table_pose = sources[0]["m4c_payload"]["table_pose"].astype(np.float64)
    table_dims = sources[0]["m4c_payload"]["table_dims"].astype(np.float64)
    table_top = float(table_pose[2] + table_dims[2] / 2.0)
    plot_min[2] = min(plot_min[2], table_top - 0.01)
    return plot_min, plot_max


def render_panel_compare(
    output: Path,
    *,
    seed_name: str,
    full: dict,
    zg: dict,
    plot_min: np.ndarray,
    plot_max: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(18, 12), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.55))
    axes = {
        "full_a": fig.add_subplot(grid[0, 0], projection="3d"),
        "zg_a": fig.add_subplot(grid[0, 1], projection="3d"),
        "rgb": fig.add_subplot(grid[0, 2]),
        "full_c": fig.add_subplot(grid[1, 0], projection="3d"),
        "zg_c": fig.add_subplot(grid[1, 1], projection="3d"),
        "note": fig.add_subplot(grid[1:, 2]),
        "full_note": fig.add_subplot(grid[2, 0]),
        "zg_note": fig.add_subplot(grid[2, 1]),
    }

    for key, title, source, drawer in (
        ("full_a", "Full-depth M4A: yaw OBB", full, lambda ax, src: draw_m4a_boxes(ax, src["m4a_payload"])),
        ("zg_a", "ZG-reconstruction M4A: yaw OBB", zg, lambda ax, src: draw_m4a_boxes(ax, src["m4a_payload"])),
        ("full_c", "Full-depth M4C: voxel ESDF", full, lambda ax, src: draw_m4c_voxels(ax, src["occ_xyz"], src["voxel_size"])),
        ("zg_c", "ZG-reconstruction M4C: voxel ESDF", zg, lambda ax, src: draw_m4c_voxels(ax, src["occ_xyz"], src["voxel_size"])),
    ):
        ax = axes[key]
        draw_table(ax, full["m4c_payload"])
        drawer(ax, source)
        setup_scene_axes(ax, plot_min=plot_min, plot_max=plot_max, title=title)

    axes["rgb"].imshow(full["rgb"])
    axes["rgb"].set_title("Same RGB input")
    axes["rgb"].axis("off")

    write_note(
        axes["full_note"],
        [
            "Full-depth source",
            f"M4A boxes: {full['m4a_box_count']}",
            f"M4C occupied voxels: {full['m4c_metadata'].get('occupied_voxels')}",
        ],
    )
    write_note(
        axes["zg_note"],
        [
            "ZeroGrasp reconstruction source",
            f"M4A boxes: {zg['m4a_box_count']}",
            f"M4C occupied voxels: {zg['m4c_metadata'].get('occupied_voxels')}",
        ],
    )
    write_note(
        axes["note"],
        [
            f"{seed_name}",
            "",
            "Top row compares M4A cuboid worlds.",
            "Bottom row compares M4C ESDF worlds.",
            "",
            "All panels use the same base-frame bounds and same camera angle.",
            "Target object is excluded from all world-collision models.",
        ],
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_diff_compare(
    output: Path,
    *,
    seed_name: str,
    full: dict,
    zg: dict,
    plot_min: np.ndarray,
    plot_max: np.ndarray,
    max_diff_voxels: int,
) -> dict:
    full_set = voxel_key_set(full["occ_xyz_all"])
    zg_set = voxel_key_set(zg["occ_xyz_all"])
    common = full_set & zg_set
    full_only = full_set - zg_set
    zg_only = zg_set - full_set

    common_xyz = keys_to_xyz(common)
    full_only_xyz = keys_to_xyz(full_only)
    zg_only_xyz = keys_to_xyz(zg_only)
    common_xyz = subsample_xyz(common_xyz, max_diff_voxels)
    full_only_xyz = subsample_xyz(full_only_xyz, max_diff_voxels)
    zg_only_xyz = subsample_xyz(zg_only_xyz, max_diff_voxels)

    fig = plt.figure(figsize=(18, 8), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.2, 1.2, 0.8))
    m4a_ax = fig.add_subplot(grid[0, 0], projection="3d")
    m4c_ax = fig.add_subplot(grid[0, 1], projection="3d")
    note_ax = fig.add_subplot(grid[0, 2])

    draw_table(m4a_ax, full["m4c_payload"])
    draw_m4a_overlay(m4a_ax, full["m4a_payload"], color="#ff8c00", linewidth=1.6, linestyle="-")
    draw_m4a_overlay(m4a_ax, zg["m4a_payload"], color="#b000ff", linewidth=1.4, linestyle="--")
    setup_scene_axes(
        m4a_ax,
        plot_min=plot_min,
        plot_max=plot_max,
        title="M4A diff: full-depth boxes vs ZG boxes",
    )

    draw_table(m4c_ax, full["m4c_payload"])
    draw_voxel_points(m4c_ax, common_xyz, full["voxel_size"], color=(0.12, 0.34, 0.95, 0.18))
    draw_voxel_points(m4c_ax, full_only_xyz, full["voxel_size"], color=(1.0, 0.48, 0.0, 0.28))
    draw_voxel_points(m4c_ax, zg_only_xyz, zg["voxel_size"], color=(0.68, 0.0, 1.0, 0.28))
    setup_scene_axes(
        m4c_ax,
        plot_min=plot_min,
        plot_max=plot_max,
        title="M4C diff: voxel occupancy overlap",
    )

    union_count = max(len(common) + len(full_only) + len(zg_only), 1)
    iou = len(common) / union_count
    write_note(
        note_ax,
        [
            f"{seed_name} visual difference",
            "",
            "M4A panel:",
            "Orange solid = full-depth yaw OBB",
            "Purple dashed = ZG-reconstruction yaw OBB",
            "",
            "M4C panel:",
            "Blue = common occupied voxels",
            "Orange = only full-depth has obstacle",
            "Purple = only ZG reconstruction has obstacle",
            "",
            f"Common voxels: {len(common)}",
            f"Full-depth only: {len(full_only)}",
            f"ZG only: {len(zg_only)}",
            f"Voxel IoU: {iou:.3f}",
        ],
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "m4c_common_voxels": int(len(common)),
        "m4c_full_only_voxels": int(len(full_only)),
        "m4c_zg_only_voxels": int(len(zg_only)),
        "m4c_voxel_iou": float(iou),
    }


def render_topdown_compare(
    output: Path,
    *,
    seed_name: str,
    full: dict,
    zg: dict,
    plot_min: np.ndarray,
    plot_max: np.ndarray,
    max_diff_voxels: int,
) -> None:
    full_set = voxel_key_set(full["occ_xyz_all"])
    zg_set = voxel_key_set(zg["occ_xyz_all"])
    common_xyz = subsample_xyz(keys_to_xyz(full_set & zg_set), max_diff_voxels)
    full_only_xyz = subsample_xyz(keys_to_xyz(full_set - zg_set), max_diff_voxels)
    zg_only_xyz = subsample_xyz(keys_to_xyz(zg_set - full_set), max_diff_voxels)

    fig = plt.figure(figsize=(16, 7), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.72))
    m4a_ax = fig.add_subplot(grid[0, 0])
    m4c_ax = fig.add_subplot(grid[0, 1])
    note_ax = fig.add_subplot(grid[0, 2])

    draw_table_topdown(m4a_ax, full["m4c_payload"])
    draw_m4a_topdown_overlay(m4a_ax, full["m4a_payload"], color="#ff8c00", linestyle="-", label="full-depth")
    draw_m4a_topdown_overlay(m4a_ax, zg["m4a_payload"], color="#b000ff", linestyle="--", label="ZG reconstruction")
    setup_topdown_axes(m4a_ax, plot_min=plot_min, plot_max=plot_max, title="M4A top-down footprint")
    m4a_ax.legend(loc="upper right")

    draw_table_topdown(m4c_ax, full["m4c_payload"])
    draw_voxel_topdown(m4c_ax, common_xyz, color="#1f5bd5", label="common", alpha=0.34)
    draw_voxel_topdown(m4c_ax, full_only_xyz, color="#ff8500", label="full-depth only", alpha=0.48)
    draw_voxel_topdown(m4c_ax, zg_only_xyz, color="#a000ff", label="ZG only", alpha=0.38)
    setup_topdown_axes(m4c_ax, plot_min=plot_min, plot_max=plot_max, title="M4C top-down voxel projection")
    m4c_ax.legend(loc="upper right")

    common_count = len(full_set & zg_set)
    full_only_count = len(full_set - zg_set)
    zg_only_count = len(zg_set - full_set)
    iou = common_count / max(common_count + full_only_count + zg_only_count, 1)
    write_note(
        note_ax,
        [
            f"{seed_name} top-down comparison",
            "",
            "Left: M4A cuboid footprint on x-y plane.",
            "Right: M4C occupied voxels projected to x-y.",
            "",
            "Orange = full-depth source",
            "Purple = ZeroGrasp reconstruction source",
            "Blue = overlap/common voxels",
            "",
            f"Common voxels: {common_count}",
            f"Full-depth only: {full_only_count}",
            f"ZG only: {zg_only_count}",
            f"Voxel IoU: {iou:.3f}",
        ],
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def draw_m4a_overlay(ax: plt.Axes, payload: dict, *, color: str, linewidth: float, linestyle: str) -> None:
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
    for name, cuboid in payload.get("cuboid", {}).items():
        if name == "real_table_static":
            continue
        corners = cuboid_corners(np.asarray(cuboid["dims"]), np.asarray(cuboid["pose"]))
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.9,
            )


def draw_m4a_topdown_overlay(
    ax: plt.Axes,
    payload: dict,
    *,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    first = True
    for name, cuboid in payload.get("cuboid", {}).items():
        if name == "real_table_static":
            continue
        corners = cuboid_corners(np.asarray(cuboid["dims"]), np.asarray(cuboid["pose"]))
        footprint = corners[[0, 1, 2, 3, 0], :2]
        ax.plot(
            footprint[:, 0],
            footprint[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            alpha=0.92,
            label=label if first else None,
        )
        first = False


def draw_table_topdown(ax: plt.Axes, payload: np.lib.npyio.NpzFile) -> None:
    table_pose = payload["table_pose"].astype(np.float64)
    table_dims = payload["table_dims"].astype(np.float64)
    x0, x1 = table_pose[0] - table_dims[0] / 2.0, table_pose[0] + table_dims[0] / 2.0
    y0, y1 = table_pose[1] - table_dims[1] / 2.0, table_pose[1] + table_dims[1] / 2.0
    ax.fill(
        [x0, x1, x1, x0],
        [y0, y0, y1, y1],
        color="#8b9098",
        alpha=0.12,
        edgecolor="#8b9098",
        linewidth=1.0,
    )


def draw_voxel_topdown(
    ax: plt.Axes,
    xyz: np.ndarray,
    *,
    color: str,
    label: str,
    alpha: float,
) -> None:
    if xyz.size == 0:
        return
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        s=10,
        marker="s",
        color=color,
        alpha=alpha,
        linewidths=0.0,
        label=label,
    )


def setup_topdown_axes(
    ax: plt.Axes,
    *,
    plot_min: np.ndarray,
    plot_max: np.ndarray,
    title: str,
) -> None:
    ax.set_xlim(plot_min[0], plot_max[0])
    ax.set_ylim(plot_min[1], plot_max[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.set_title(title)


def draw_voxel_points(ax: plt.Axes, xyz: np.ndarray, voxel_size: float, *, color: tuple[float, float, float, float]) -> None:
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
        edgecolor=(0.05, 0.05, 0.05, 0.04),
        linewidth=0.06,
        shade=True,
    )


def voxel_key_set(xyz: np.ndarray) -> set[tuple[int, int, int]]:
    if xyz.size == 0:
        return set()
    return {tuple(row) for row in np.rint(xyz * 1000.0).astype(np.int64)}


def keys_to_xyz(keys: set[tuple[int, int, int]]) -> np.ndarray:
    if not keys:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(sorted(keys), dtype=np.float64) / 1000.0


def subsample_xyz(xyz: np.ndarray, max_count: int) -> np.ndarray:
    if xyz.shape[0] <= max_count:
        return xyz
    stride = int(np.ceil(xyz.shape[0] / max_count))
    return xyz[::stride]


if __name__ == "__main__":
    raise SystemExit(main())
