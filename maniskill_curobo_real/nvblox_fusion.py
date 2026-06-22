#!/usr/bin/env python3
"""Nvblox-style integration entry point for multi-view RGB-D fusion experiments.

The current repository can already capture the required RGB-D bundles and
camera poses. This module keeps the nvblox dependency boundary explicit so the
rest of the experiment remains testable when nvblox_torch is not installed.

When NVIDIA's nvblox Python bindings are unavailable, this file can use
cuRobo's built-in block-sparse TSDF/ESDF mapper as the local nvblox-style
backend. That keeps the experiment focused on RGB-D + pose fusion rather than
raw point-cloud concatenation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from maniskill_curobo_real.scene_builder import (
    DEFAULT_TABLE_CENTER_XY,
    DEFAULT_TABLE_DIMS,
    DEFAULT_TABLE_QUAT_WXYZ,
    DEFAULT_TABLE_TOP_Z,
    depth_to_meters,
)


NVBLOX_MODULE_CANDIDATES = ("nvblox_torch", "nvblox", "pynvblox")


@dataclass(frozen=True)
class NvbloxAvailability:
    """Installed nvblox Python backend status."""

    available: bool
    module_name: str | None
    module_origin: str | None
    checked_modules: tuple[str, ...]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="maniskill_curobo_real/runs/nvblox_multiview_rgbd",
        help="Root created by capture_multiview_rgbd.py.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("external-nvblox", "curobo-mapper"),
        default="curobo-mapper",
        help=(
            "external-nvblox requires nvblox_torch/nvblox. curobo-mapper uses "
            "cuRobo's local block-sparse TSDF/ESDF mapper."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for fused curobo_scene_voxel.npz and metadata.",
    )
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--esdf-voxel-size", type=float, default=0.01)
    parser.add_argument("--truncation-distance", type=float, default=0.04)
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-max", type=float, default=2.5)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for cuRobo mapper.",
    )
    parser.add_argument(
        "--exclude-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero out depth pixels whose mask label belongs to task target objects.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only print backend availability and input manifest status.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    availability = check_nvblox_availability()
    manifest_status = inspect_input_root(Path(args.input_root), seed=args.seed)
    payload = {
        "nvblox": asdict(availability),
        "input_root": str(Path(args.input_root).expanduser().resolve()),
        "input": manifest_status,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.check_only:
        if args.backend == "external-nvblox":
            return 0 if availability.available else 2
        return 0
    if args.backend == "external-nvblox":
        require_nvblox(availability)
        raise NotImplementedError(
            "External nvblox RGB-D integration is not implemented in this environment yet. "
            "Install nvblox_torch/nvblox first, then connect its depth integration "
            "API here using the saved view_metadata.json poses."
        )
    if args.seed is None:
        raise ValueError("--seed is required when running curobo-mapper fusion.")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(args.input_root).expanduser().resolve() / f"seed{int(args.seed):03d}" / "curobo_mapper_esdf"
    )
    result = fuse_seed_with_curobo_mapper(
        input_root=Path(args.input_root),
        seed=int(args.seed),
        output_dir=output_dir,
        voxel_size=float(args.voxel_size),
        esdf_voxel_size=float(args.esdf_voxel_size),
        truncation_distance=float(args.truncation_distance),
        depth_min=float(args.depth_min),
        depth_max=float(args.depth_max),
        device=str(args.device),
        exclude_target=bool(args.exclude_target),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def check_nvblox_availability() -> NvbloxAvailability:
    for module_name in NVBLOX_MODULE_CANDIDATES:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return NvbloxAvailability(
                available=True,
                module_name=module_name,
                module_origin=spec.origin,
                checked_modules=NVBLOX_MODULE_CANDIDATES,
            )
    return NvbloxAvailability(
        available=False,
        module_name=None,
        module_origin=None,
        checked_modules=NVBLOX_MODULE_CANDIDATES,
    )


def require_nvblox(availability: NvbloxAvailability | None = None) -> NvbloxAvailability:
    status = availability if availability is not None else check_nvblox_availability()
    if not status.available:
        checked = ", ".join(status.checked_modules)
        raise RuntimeError(
            "No nvblox Python backend found. Checked: "
            f"{checked}. Install nvblox_torch or expose the cuRobo nvblox "
            "Python bindings before running RGB-D fusion."
        )
    return status


def inspect_input_root(input_root: Path, *, seed: int | None = None) -> dict:
    root = input_root.expanduser().resolve()
    if seed is not None:
        seed_dirs = [root / f"seed{int(seed):03d}"]
    else:
        seed_dirs = sorted(path for path in root.glob("seed*") if path.is_dir())
    records = []
    for seed_dir in seed_dirs:
        manifest = seed_dir / "multiview_manifest.json"
        view_metadata = sorted(seed_dir.glob("view_*/view_metadata.json"))
        records.append(
            {
                "seed_dir": str(seed_dir),
                "manifest_exists": manifest.is_file(),
                "n_view_metadata": len(view_metadata),
                "views": [str(path.parent.name) for path in view_metadata],
            }
        )
    return {
        "exists": root.exists(),
        "n_seed_dirs": len(records),
        "records": records[:20],
        "records_truncated": len(records) > 20,
    }


def fuse_seed_with_curobo_mapper(
    *,
    input_root: Path,
    seed: int,
    output_dir: Path,
    voxel_size: float,
    esdf_voxel_size: float,
    truncation_distance: float,
    depth_min: float,
    depth_max: float,
    device: str,
    exclude_target: bool,
) -> dict:
    """Fuse saved multi-view RGB-D bundles into a cuRobo ESDF voxel grid."""

    import torch

    from curobo._src.perception.mapper.mapper import Mapper
    from curobo._src.perception.mapper.mapper_cfg import MapperCfg
    from curobo._src.types.camera import CameraObservation
    from curobo._src.types.pose import Pose
    from curobo._src.util.warp import init_warp

    started = time.time()
    seed_dir = input_root.expanduser().resolve() / f"seed{int(seed):03d}"
    views = load_multiview_observations(
        seed_dir=seed_dir,
        device=device,
        exclude_target=exclude_target,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    if not views:
        raise FileNotFoundError(f"No view_metadata.json files found under {seed_dir}")
    first = views[0]
    height, width = first["depth"].shape[-2:]
    center, dims = workspace_center_and_dims_from_views(views)
    init_warp()
    mapper = Mapper(
        MapperCfg(
            extent_meters_xyz=tuple(float(v) for v in dims),
            extent_esdf_meters_xyz=tuple(float(v) for v in dims),
            voxel_size=float(voxel_size),
            esdf_voxel_size=float(esdf_voxel_size),
            truncation_distance=float(truncation_distance),
            depth_minimum_distance=float(depth_min),
            depth_maximum_distance=float(depth_max),
            grid_center=torch.as_tensor(center, device=device, dtype=torch.float32),
            num_cameras=1,
            image_height=int(height),
            image_width=int(width),
            device=device,
        )
    )
    integration_records = []
    for view in views:
        observation = CameraObservation(
            depth_image=view["depth"],
            rgb_image=view["rgb"],
            pose=Pose.from_matrix(view["base_from_camera"]),
            intrinsics=view["intrinsics"],
        )
        mapper.integrate(observation)
        integration_records.append(view["record"])
    voxel_grid = mapper.compute_esdf(esdf_origin=torch.as_tensor(center, device=device, dtype=torch.float32))
    scene_path, metadata_path = write_curobo_mapper_voxel_grid(
        voxel_grid=voxel_grid,
        output_dir=output_dir,
        metadata={
            "source": "m5_curobo_mapper_multiview_rgbd_esdf",
            "seed": int(seed),
            "backend": "curobo_mapper",
            "n_views": len(views),
            "views": integration_records,
            "grid_center": [float(v) for v in center],
            "grid_dims": [float(v) for v in dims],
            "voxel_size": float(voxel_size),
            "esdf_voxel_size": float(esdf_voxel_size),
            "truncation_distance": float(truncation_distance),
            "depth_min": float(depth_min),
            "depth_max": float(depth_max),
            "exclude_target": bool(exclude_target),
            "runtime_sec": float(time.time() - started),
        },
    )
    return {
        "status": "ok",
        "seed": int(seed),
        "backend": "curobo_mapper",
        "scene_model": str(scene_path),
        "metadata": str(metadata_path),
        "n_views": len(views),
        "runtime_sec": float(time.time() - started),
    }


def load_multiview_observations(
    *,
    seed_dir: Path,
    device: str,
    exclude_target: bool,
    depth_min: float,
    depth_max: float,
) -> list[dict]:
    import torch

    observations: list[dict] = []
    for metadata_path in sorted(seed_dir.glob("view_*/view_metadata.json")):
        view_dir = metadata_path.parent
        rgbd_path = view_dir / "rgbd.npz"
        camera_path = view_dir / "camera.json"
        if not rgbd_path.is_file() or not camera_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        with np.load(rgbd_path) as payload:
            rgb = np.asarray(payload["rgb"], dtype=np.uint8)
            depth = depth_to_meters(np.asarray(payload["depth"], dtype=np.float32))
            mask = np.asarray(payload["mask"], dtype=np.uint8)
            intrinsics = np.asarray(payload["cam_K"], dtype=np.float32).reshape(3, 3)
        target_labels = {
            int(record["label"])
            for record in camera.get("objects", [])
            if bool(record.get("is_task_target", False))
        }
        if exclude_target and target_labels:
            target_mask = np.isin(mask, list(target_labels))
            depth = depth.copy()
            depth[target_mask] = 0.0
        valid_depth = np.isfinite(depth) & (depth >= float(depth_min)) & (depth <= float(depth_max))
        depth = np.where(valid_depth, depth, 0.0).astype(np.float32)
        base_from_camera = np.asarray(metadata["base_from_camera"], dtype=np.float32).reshape(4, 4)
        observations.append(
            {
                "depth": torch.as_tensor(depth, device=device, dtype=torch.float32).unsqueeze(0),
                "rgb": torch.as_tensor(rgb, device=device, dtype=torch.uint8).unsqueeze(0),
                "intrinsics": torch.as_tensor(intrinsics, device=device, dtype=torch.float32).unsqueeze(0),
                "base_from_camera": torch.as_tensor(base_from_camera, device=device, dtype=torch.float32),
                "record": {
                    "name": metadata.get("view", {}).get("name", view_dir.name),
                    "role": metadata.get("view", {}).get("role"),
                    "rgbd": str(rgbd_path),
                    "metadata": str(metadata_path),
                    "target_labels_excluded": sorted(target_labels),
                    "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
                },
            }
        )
    return observations


def workspace_center_and_dims_from_views(views: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Use the existing M4C workspace bounds used by scene_builder."""

    from maniskill_curobo_real.scene_builder import DEFAULT_WORKSPACE_BOUNDS

    bounds = np.asarray(DEFAULT_WORKSPACE_BOUNDS, dtype=np.float32)
    center = 0.5 * (bounds[:, 0] + bounds[:, 1])
    dims = bounds[:, 1] - bounds[:, 0]
    return center.astype(np.float32), dims.astype(np.float32)


def write_curobo_mapper_voxel_grid(
    *,
    voxel_grid,
    output_dir: Path,
    metadata: dict,
) -> tuple[Path, Path]:
    import torch

    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene_path = out / "curobo_scene_voxel.npz"
    metadata_path = out / "curobo_scene_metadata.json"
    feature = voxel_grid.feature_tensor
    if isinstance(feature, torch.Tensor):
        feature_np = feature.detach().cpu().numpy()
    else:
        feature_np = np.asarray(feature)
    pose = np.asarray(voxel_grid.pose, dtype=np.float64)
    center = pose[:3]
    dims = np.asarray(voxel_grid.dims, dtype=np.float64)
    table_pose = default_table_pose()
    np.savez_compressed(
        scene_path,
        feature_tensor=feature_np.astype(np.float16),
        voxel_center=center,
        voxel_dims=dims,
        voxel_size=np.asarray(float(voxel_grid.voxel_size), dtype=np.float64),
        table_pose=np.asarray(table_pose, dtype=np.float64),
        table_dims=np.asarray(DEFAULT_TABLE_DIMS, dtype=np.float64),
    )
    meta = dict(metadata)
    surface_band = feature_np <= float(voxel_grid.voxel_size)
    meta.update(
        {
            "scene_model": str(scene_path),
            "voxel_center": center.tolist(),
            "voxel_dims": dims.tolist(),
            "voxel_shape": list(feature_np.shape),
            "occupied_voxels": int(np.count_nonzero(feature_np <= 0.0)),
            "occupied_fraction": float(np.mean(feature_np <= 0.0)),
            "surface_band_voxels": int(np.count_nonzero(surface_band)),
            "surface_band_fraction": float(np.mean(surface_band)),
            "table_pose": table_pose,
            "table_dims": list(DEFAULT_TABLE_DIMS),
        }
    )
    metadata_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return scene_path, metadata_path


def default_table_pose() -> list[float]:
    dims = np.asarray(DEFAULT_TABLE_DIMS, dtype=np.float64).reshape(3)
    return [
        float(DEFAULT_TABLE_CENTER_XY[0]),
        float(DEFAULT_TABLE_CENTER_XY[1]),
        float(DEFAULT_TABLE_TOP_Z - dims[2] / 2.0),
        *[float(v) for v in DEFAULT_TABLE_QUAT_WXYZ],
    ]


if __name__ == "__main__":
    raise SystemExit(main())
