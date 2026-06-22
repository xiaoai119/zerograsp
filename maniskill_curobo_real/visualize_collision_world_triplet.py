#!/usr/bin/env python3
"""Visualize M0, M4C and M5 collision worlds for the same seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from maniskill_curobo_real.scene_builder import DEFAULT_WORKSPACE_BOUNDS
from maniskill_curobo_real.visualize_m5_instance_reconstruction import (
    cuboid_edges,
    cuboid_corners,
    idx_to_xyz,
    subsample,
    voxel_axes,
)


DEFAULT_M0_ROOT = Path("maniskill_curobo_real/runs/pickclutter_full_depth_m0_seed1_200")
DEFAULT_M4C_ROOT = Path("maniskill_curobo_real/runs/pickclutter_full_depth_m4abc_seed1_200")
DEFAULT_M5_ROOT = Path("maniskill_curobo_real/runs/pickclutter_m5_instance_esdf_no_table_seed1_200")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0_ROOT)
    parser.add_argument("--m4c-root", type=Path, default=DEFAULT_M4C_ROOT)
    parser.add_argument("--m5-root", type=Path, default=DEFAULT_M5_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 9, 10, 16, 21])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cuboids", type=int, default=120)
    parser.add_argument("--max-voxels", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    m0_root = args.m0_root.expanduser().resolve()
    m4c_root = args.m4c_root.expanduser().resolve()
    m5_root = args.m5_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else m5_root / "visualization_collision_world_triplet"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        try:
            record = visualize_seed(
                seed=int(seed),
                m0_root=m0_root,
                m4c_root=m4c_root,
                m5_root=m5_root,
                output_dir=output_dir,
                max_cuboids=int(args.max_cuboids),
                max_voxels=int(args.max_voxels),
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, indent=2))
        except FileNotFoundError as exc:
            record = {"seed": int(seed), "status": "missing_input", "missing": str(exc)}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, indent=2))
    summary_path = output_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def visualize_seed(
    *,
    seed: int,
    m0_root: Path,
    m4c_root: Path,
    m5_root: Path,
    output_dir: Path,
    max_cuboids: int,
    max_voxels: int,
) -> dict:
    seed_name = f"seed{seed:03d}"
    m0_scene = m0_root / "m0_maniskill_truth" / seed_name / "curobo_scene.yml"
    m4c_scene = m4c_root / "m4c_oracle_instance_voxel_esdf" / seed_name / "real_scene" / "curobo_scene_voxel.npz"
    m5_scene = m5_root / "m5_multiview_instance_voxel_esdf_no_table" / seed_name / "real_scene" / "curobo_scene_voxel.npz"
    m4c_meta = m4c_scene.with_name("curobo_scene_metadata.json")
    m5_meta = m5_scene.with_name("curobo_scene_metadata.json")
    for path in (m0_scene, m4c_scene, m5_scene, m4c_meta, m5_meta):
        if not path.is_file():
            raise FileNotFoundError(path)
    m0 = load_cuboid_scene(m0_scene)
    m4c = load_voxel_scene(m4c_scene, max_voxels=max_voxels)
    m5 = load_voxel_scene(m5_scene, max_voxels=max_voxels)
    m4c_metadata = json.loads(m4c_meta.read_text(encoding="utf-8"))
    m5_metadata = json.loads(m5_meta.read_text(encoding="utf-8"))
    output_path = output_dir / f"{seed_name}_m0_m4c_m5_collision_worlds.png"
    render_triplet(
        output_path=output_path,
        seed_name=seed_name,
        m0=m0,
        m4c=m4c,
        m5=m5,
        m4c_metadata=m4c_metadata,
        m5_metadata=m5_metadata,
        max_cuboids=max_cuboids,
    )
    return {
        "seed": int(seed),
        "status": "ok",
        "output": str(output_path),
        "m0_cuboids": len(m0.get("cuboid", {})),
        "m4c_occupied_voxels": int(m4c["occupied_count"]),
        "m5_occupied_voxels": int(m5["occupied_count"]),
        "m4c_obstacles": int(m4c_metadata.get("n_pointcloud_obstacles", -1)),
        "m5_obstacles": int(m5_metadata.get("n_pointcloud_obstacles", -1)),
    }


def load_cuboid_scene(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_voxel_scene(path: Path, *, max_voxels: int) -> dict:
    with np.load(path) as payload:
        sdf = payload["feature_tensor"].astype(np.float32)
        center = payload["voxel_center"].astype(np.float64)
        voxel_size = float(payload["voxel_size"])
        dims = payload["voxel_dims"].astype(np.float64)
        table_pose = payload["table_pose"].astype(np.float64) if "table_pose" in payload else None
        table_dims = payload["table_dims"].astype(np.float64) if "table_dims" in payload else None
    axes = voxel_axes(center, voxel_size, sdf.shape)
    occ_idx = np.argwhere(sdf <= 0.0)
    occ_xyz = idx_to_xyz(occ_idx, axes)
    return {
        "occupied_xyz": subsample(occ_xyz, max_voxels),
        "occupied_count": int(occ_idx.shape[0]),
        "center": center,
        "dims": dims,
        "voxel_size": voxel_size,
        "table_pose": table_pose,
        "table_dims": table_dims,
    }


def render_triplet(
    *,
    output_path: Path,
    seed_name: str,
    m0: dict,
    m4c: dict,
    m5: dict,
    m4c_metadata: dict,
    m5_metadata: dict,
    max_cuboids: int,
) -> None:
    fig = plt.figure(figsize=(22, 8), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.22))
    axes = [fig.add_subplot(grid[0, i], projection="3d") for i in range(3)]
    notes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    draw_m0(axes[0], m0, max_cuboids=max_cuboids)
    draw_voxels(axes[1], m4c, title="M4C oracle single-view voxel ESDF")
    draw_voxels(axes[2], m5, title="M5 multiview instance voxel ESDF, no table")
    setup_3d_axis(axes[0], f"{seed_name}: M0 ManiSkill truth cuboids")
    setup_3d_axis(axes[1], f"{seed_name}: M4C")
    setup_3d_axis(axes[2], f"{seed_name}: M5 latest reconstruction")
    note_texts = [
        [
            f"Cuboids: {len(m0.get('cuboid', {}))}",
            "Uses ManiSkill truth collision export.",
        ],
        [
            f"Obstacles: {m4c_metadata.get('n_pointcloud_obstacles')}",
            f"Occupied voxels: {m4c['occupied_count']}",
            f"Table included: {'table_pose' in m4c and m4c['table_pose'] is not None}",
        ],
        [
            f"Obstacles: {m5_metadata.get('n_pointcloud_obstacles')}",
            f"Occupied voxels: {m5['occupied_count']}",
            f"Table included: {m5_metadata.get('table_included')}",
            "Input: 3-view RGB-D + instance IDs.",
        ],
    ]
    for ax, text in zip(notes, note_texts):
        ax.axis("off")
        ax.text(0.02, 0.9, "\n".join(text), ha="left", va="top", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def draw_m0(ax, scene: dict, *, max_cuboids: int) -> None:
    cuboids = scene.get("cuboid", {})
    for index, (name, cuboid) in enumerate(cuboids.items()):
        if index >= max_cuboids:
            break
        color = "#8a8a8a" if "table" in name else "#d95f02"
        linewidth = 0.8 if "table" in name else 1.0
        draw_cuboid(ax, np.asarray(cuboid["pose"], dtype=np.float64), np.asarray(cuboid["dims"], dtype=np.float64), color=color, linewidth=linewidth)


def draw_voxels(ax, scene: dict, *, title: str) -> None:
    if scene["table_pose"] is not None and scene["table_dims"] is not None:
        draw_cuboid(ax, scene["table_pose"], scene["table_dims"], color="#8a8a8a", linewidth=0.8)
    pts = scene["occupied_xyz"]
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, c="#1f77b4", alpha=0.45)
    ax.set_title(title)


def draw_cuboid(ax, pose: np.ndarray, dims: np.ndarray, *, color: str, linewidth: float) -> None:
    corners = cuboid_corners(pose, dims)
    for i, j in cuboid_edges():
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color,
            linewidth=linewidth,
        )


def setup_3d_axis(ax, title: str) -> None:
    bounds = np.asarray(DEFAULT_WORKSPACE_BOUNDS, dtype=np.float64)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.set_zlabel("base z (m)")
    ax.set_title(title)
    try:
        ax.set_box_aspect((bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0], bounds[2, 1] - bounds[2, 0]))
    except Exception:
        pass
    ax.view_init(elev=23, azim=-58)


if __name__ == "__main__":
    raise SystemExit(main())
