"""Official GraspNet baseline runtime wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import open3d as o3d
import torch

from maniskill_curobo_graspnet.graspnet_adapter import (
    filter_grasps_by_target_limits,
    grasp_to_standard_json,
    load_point_cloud_input,
)


class GraspNetBaselineRuntime:
    """Load official GraspNet baseline once and run repeated predictions."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        checkpoint_path: str | Path,
        num_point: int = 20000,
        num_view: int = 300,
        collision_thresh: float = 0.01,
        voxel_size: float = 0.01,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.num_point = int(num_point)
        self.num_view = int(num_view)
        self.collision_thresh = float(collision_thresh)
        self.voxel_size = float(voxel_size)
        validate_runtime_files(self.repo_root, self.checkpoint_path)
        add_baseline_paths(self.repo_root)
        from graspnet import GraspNet, pred_decode
        from collision_detector import ModelFreeCollisionDetector

        self.pred_decode = pred_decode
        self.collision_detector_cls = ModelFreeCollisionDetector
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GraspNet(
            input_feature_dim=0,
            num_view=self.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        self.model.to(self.device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.checkpoint_epoch = int(checkpoint.get("epoch", -1))
        self.model.eval()

    def predict(
        self,
        *,
        input_dir: str | Path,
        output_dir: str | Path,
        depth_unit: str = "auto",
        target_margin_m: float = 0.02,
        context_margin_m: float = 0.08,
        filter_to_target: bool = True,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        started = time.time()
        prepared = load_point_cloud_input(
            input_dir,
            depth_unit=depth_unit,
            target_margin_m=target_margin_m,
            context_margin_m=context_margin_m,
        )
        points, colors = sample_points(
            prepared.points,
            prepared.colors,
            num_point=self.num_point,
            random_seed=random_seed,
        )
        end_points = {
            "point_clouds": torch.from_numpy(points[None].astype(np.float32)).to(
                self.device
            ),
            "cloud_colors": colors.astype(np.float32),
        }
        with torch.no_grad():
            outputs = self.model(end_points)
            preds = self.pred_decode(outputs)

        from graspnetAPI import GraspGroup

        grasp_group = GraspGroup(preds[0].detach().cpu().numpy())
        raw_array = np.asarray(grasp_group.grasp_group_array, dtype=np.float64)
        raw_count = int(raw_array.shape[0]) if raw_array.ndim == 2 else 0

        target_filtered_count = None
        if filter_to_target:
            grasp_group = filter_grasps_by_target_limits(
                grasp_group, prepared.target_limits
            )
            target_filtered_count = int(len(grasp_group))
        if self.collision_thresh > 0 and len(grasp_group) > 0:
            detector = self.collision_detector_cls(
                prepared.points, voxel_size=self.voxel_size
            )
            collision_mask = detector.detect(
                grasp_group,
                approach_dist=0.05,
                collision_thresh=self.collision_thresh,
            )
            grasp_group = grasp_group[~collision_mask]
        if len(grasp_group) > 0:
            grasp_group = grasp_group.nms().sort_by_score()

        output = Path(output_dir).expanduser().resolve()
        raw_dir = output / "raw_outputs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        np.save(raw_dir / "graspnet.grasp.npy", raw_array)
        recommended = None
        if len(grasp_group) > 0:
            recommended = grasp_to_standard_json(
                grasp_group[0],
                object_id=prepared.target_label,
            )
            (output / "recommended_grasp_top1.json").write_text(
                json.dumps(recommended, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        report = {
            "model": "graspnet-baseline",
            "runtime_sec": float(time.time() - started),
            "checkpoint_epoch": self.checkpoint_epoch,
            "n_grasps_raw": raw_count,
            "n_grasps_target_filtered": target_filtered_count,
            "n_grasps_final": int(len(grasp_group)),
            "recommended_grasp": recommended,
            "input": {
                "input_dir": str(Path(input_dir).expanduser().resolve()),
                "target_label": prepared.target_label,
                "target_point_count": prepared.target_point_count,
                "context_point_count": prepared.context_point_count,
                "target_limits": prepared.target_limits,
                "depth_scale_to_m": prepared.depth_scale_to_m,
            },
            "settings": {
                "num_point": self.num_point,
                "num_view": self.num_view,
                "collision_thresh": self.collision_thresh,
                "voxel_size": self.voxel_size,
                "target_margin_m": float(target_margin_m),
                "context_margin_m": float(context_margin_m),
                "filter_to_target": bool(filter_to_target),
            },
        }
        (output / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report


def sample_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    num_point: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    rng = np.random.default_rng(int(random_seed))
    if len(points) >= num_point:
        idxs = rng.choice(len(points), num_point, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = rng.choice(len(points), num_point - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    return (
        np.ascontiguousarray(points[idxs], dtype=np.float32),
        np.ascontiguousarray(colors[idxs], dtype=np.float32),
    )


def validate_runtime_files(repo_root: Path, checkpoint_path: Path) -> None:
    if not repo_root.is_dir():
        raise FileNotFoundError(f"GraspNet baseline repo not found: {repo_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"GraspNet checkpoint not found: {checkpoint_path}. "
            "Download checkpoint-rs.tar or checkpoint-kn.tar from the official README."
        )


def add_baseline_paths(repo_root: Path) -> None:
    for path in (
        repo_root,
        repo_root / "models",
        repo_root / "dataset",
        repo_root / "utils",
        repo_root / "pointnet2",
        repo_root / "knn",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def point_cloud_to_open3d(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return cloud
