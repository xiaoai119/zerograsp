"""Programmatic API for ZeroGrasp grasp detection.

Provides a reusable pipeline that loads the model once and accepts images
as numpy arrays or file paths, returning structured grasp results in-memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch as th
import torch.nn.functional as F

from main import BaseTrainer
from zerograsp.utils.array_bridge import numpy_to_torch, torch_to_numpy
from zerograsp.utils.collision_detector import (
    GRASP_MAX_DEPTH,
    GRASP_MAX_WIDTH,
    ModelFreeCollisionDetector,
)
from zerograsp.utils.config import parse_config
from zerograsp.utils.dataset import fetch_data, make_batch_from_arrays
from zerograsp.utils.math import rotation_6d_to_matrix, unnormalize_pts
from zerograsp.nets.utils import get_xyz_from_octree


def _load_grasp_group_class():
    try:
        from graspnetAPI import GraspGroup as api_grasp_group

        return api_grasp_group
    except ModuleNotFoundError:
        spec = importlib.util.find_spec("graspnetAPI")
        if spec is None or not spec.submodule_search_locations:
            raise
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        package = types.ModuleType("graspnetAPI")
        package.__path__ = [str(package_dir)]
        package.__package__ = "graspnetAPI"
        sys.modules["graspnetAPI"] = package
        grasp_spec = importlib.util.spec_from_file_location(
            "graspnetAPI.grasp",
            package_dir / "grasp.py",
        )
        if grasp_spec is None or grasp_spec.loader is None:
            raise
        grasp_module = importlib.util.module_from_spec(grasp_spec)
        sys.modules["graspnetAPI.grasp"] = grasp_module
        grasp_spec.loader.exec_module(grasp_module)
        package.GraspGroup = grasp_module.GraspGroup
        return grasp_module.GraspGroup


GraspGroup = _load_grasp_group_class()


@dataclass
class GraspPrediction:
    """A single grasp prediction in camera frame."""

    score: float
    width: float
    height: float
    depth: float
    rotation_matrix: np.ndarray  # (3, 3) camera-to-gripper rotation
    translation: np.ndarray  # (3,) gripper center in camera frame (meters)
    object_id: int  # semantic label from the instance mask


@dataclass
class ObjectGraspResult:
    """All grasp results for a single detected object instance."""

    object_id: int
    object_index: int
    point_cloud: np.ndarray  # (N, 3) reconstructed points (mm)
    normals: np.ndarray  # (N, 3) surface normals
    grasp_group_array: np.ndarray  # (M, 17) raw GraspGroup array after NMS
    n_grasps_before_collision: int
    n_grasps_after_collision: int
    n_grasps_final: int

    @property
    def grasps(self) -> list[GraspPrediction]:
        return _decode_grasp_group_array(self.grasp_group_array, self.object_id)


@dataclass
class ZeroGraspOutput:
    """Top-level output for one predict() call."""

    objects: list[ObjectGraspResult]
    runtime_sec: float = 0.0

    def __len__(self) -> int:
        return len(self.objects)

    def top_k_grasps(self, k: int = 1) -> list[GraspPrediction]:
        all_grasps = [g for obj in self.objects for g in obj.grasps]
        all_grasps.sort(key=lambda g: g.score, reverse=True)
        return all_grasps[:k]

    def recommended_grasp(self) -> Optional[GraspPrediction]:
        top = self.top_k_grasps(1)
        return top[0] if top else None


def _decode_grasp_group_array(arr: np.ndarray, object_id: int) -> list[GraspPrediction]:
    results = []
    for i in range(arr.shape[0]):
        row = arr[i]
        results.append(
            GraspPrediction(
                score=float(row[0]),
                width=float(row[1]),
                height=float(row[2]),
                depth=float(row[3]),
                rotation_matrix=row[4:13].reshape(3, 3).copy(),
                translation=row[13:16].copy(),
                object_id=object_id,
            )
        )
    return results


class ZeroGraspPipeline:
    """Programmatic API for ZeroGrasp grasp detection.

    Model is loaded once at construction time and reused across calls.

    Usage::

        pipeline = ZeroGraspPipeline(
            checkpoint_path="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
            config_path="configs/demo.yaml",
        )
        result = pipeline.predict(rgb, depth, mask, K)
        best = result.recommended_grasp()
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        config: Optional[object] = None,
        device: Optional[str] = None,
    ):
        if config_path is None and config is None:
            raise ValueError("Either config_path or config must be provided.")
        if config_path is not None and config is not None:
            raise ValueError("Provide either config_path or config, not both.")

        if device is None:
            device = "cuda" if th.cuda.is_available() else "cpu"
        self._device = device

        if config_path is not None:
            self._config = parse_config(config_path)
        else:
            self._config = config

        self._config.update_octree = True
        if not self._config.predict_grasp:
            self._config.predict_grasp = True

        self._model = BaseTrainer.load_from_checkpoint(
            checkpoint_path,
            config=self._config,
            strict=False,
            map_location=self._device,
        )
        self._model.to(self._device)
        self._model.eval()

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        depth_scale: float = 1.0,
    ) -> ZeroGraspOutput:
        """Run grasp detection from in-memory numpy arrays.

        Args:
            rgb: (H, W, 3) uint8 RGB image.
            depth: (H, W) float32 depth map.
            mask: (H, W) int32 instance mask (0 = background).
            K: (3, 3) float32 camera intrinsics matrix.
            depth_scale: Scalar factor applied to depth values (default 1.0).

        Returns:
            ZeroGraspOutput with per-object grasp predictions.

        Raises:
            ValueError: If input dimensions do not match config.
        """
        H, W = rgb.shape[:2]
        if H != self._config.img_height or W != self._config.img_width:
            raise ValueError(
                f"Input image size ({H}, {W}) does not match config "
                f"img_height/img_width ({self._config.img_height}, "
                f"{self._config.img_width}). Resize inputs or update the config."
            )

        batch = make_batch_from_arrays(
            rgb, depth, mask, K, self._config,
            depth_scale=depth_scale, device=self._device,
        )
        return self._infer(batch)

    def predict_from_files(
        self,
        rgb_path: str,
        depth_path: str,
        mask_path: str,
        camera_path: str,
        depth_scale: float = 1.0,
    ) -> ZeroGraspOutput:
        """Run grasp detection from file paths.

        Args:
            rgb_path: Path to RGB image.
            depth_path: Path to depth image.
            mask_path: Path to instance mask image.
            camera_path: Path to camera info JSON/YAML file.
            depth_scale: Scalar factor applied to depth values (default 1.0).

        Returns:
            ZeroGraspOutput with per-object grasp predictions.
        """
        batch = fetch_data(
            rgb_path, depth_path, mask_path, camera_path,
            self._config, depth_scale, device=self._device,
        )
        return self._infer(batch)

    def _infer(self, batch: tuple) -> ZeroGraspOutput:
        """Run model inference and post-processing."""
        self._model.eval()
        started = time.perf_counter()

        with th.no_grad():
            output = self._model.model(batch)

        pts_3d_in = batch[3][0]
        rays_3d = batch[4][0]
        z_min = batch[-2][0]

        octrees_out = output["octrees_out"]
        pcd, batch_id = get_xyz_from_octree(
            octrees_out, self._config.max_lod, nempty=True, return_batch=True
        )
        grid_res = 1 << self._config.min_lod
        pcd = unnormalize_pts(pcd, z_min, self._config.grid_size, grid_res)
        normals = octrees_out.normals[self._config.max_lod]
        signal = octrees_out.features[self._config.max_lod]
        sdf = signal[:, :1]

        batch_id_np = torch_to_numpy(batch_id).reshape(-1)
        pcd_np = torch_to_numpy(pcd).reshape(-1, 3)
        normals_np = torch_to_numpy(F.normalize(normals, dim=-1)).reshape(-1, 3)
        sdf_np = torch_to_numpy(sdf).reshape(-1, 1)
        pcd_np = pcd_np - normals_np * sdf_np

        signal_np = torch_to_numpy(signal[:, 1:].clone())  # (N, C-1) where C depends on config

        obj_ids = th.unique(pts_3d_in.labels, sorted=True)
        depth_pcd = torch_to_numpy(rays_3d.reshape(-1, 3)[::5])

        object_results = []
        for i, oi in enumerate(obj_ids):
            mask = batch_id_np == i
            if not np.any(mask):
                continue

            masked_pcd = np.ascontiguousarray(pcd_np[mask], dtype=np.float32)
            masked_normals = np.ascontiguousarray(normals_np[mask], dtype=np.float32)
            masked_signal = signal_np[mask]

            quality = masked_signal[:, 0:1]
            tangent = th.from_numpy(masked_signal[:, 2:5])
            gnormal = th.from_numpy(masked_signal[:, 5:8])
            R_matrix = rotation_6d_to_matrix(th.cat([-gnormal, tangent], dim=-1))
            R_np = torch_to_numpy(R_matrix).reshape(-1, 9)
            grasp_depth = masked_signal[:, 8:9]
            grasp_width = masked_signal[:, 9:10]
            translation = masked_pcd.reshape(-1, 3) / 1000.0
            height = np.full((quality.shape[0], 1), 0.02, dtype=quality.dtype)

            grasp_preds = np.concatenate(
                [
                    quality,
                    np.clip(grasp_width * GRASP_MAX_WIDTH, 0.0, GRASP_MAX_WIDTH),
                    height,
                    np.clip(grasp_depth * GRASP_MAX_DEPTH, 0.0, GRASP_MAX_DEPTH),
                    R_np,
                    translation,
                    -1 * np.ones((quality.shape[0], 1), dtype=quality.dtype),
                ],
                axis=-1,
            )

            gg = GraspGroup(grasp_preds)
            gg = gg.sort_by_score()
            n_before = len(gg)

            if self._config.use_collision_detection:
                with th.no_grad():
                    cloud = numpy_to_torch(masked_pcd / 1000.0, device=self._device).float()
                    cloud_nrm = numpy_to_torch(masked_normals, device=self._device).float()
                    depth_cloud = numpy_to_torch(
                        depth_pcd / 1000.0, device=self._device
                    ).float()
                    mfcdetector = ModelFreeCollisionDetector(cloud, cloud_nrm, depth_cloud)
                    collision_mask, delta_width, refined_depth = mfcdetector.detect(gg)
                    gg.grasp_group_array[:, 1] = gg.grasp_group_array[:, 1] + delta_width
                    gg.grasp_group_array[:, 3] = refined_depth

                if (~collision_mask).sum() > 0:
                    gg = gg[~collision_mask]

            n_after_col = len(gg)
            gg = gg.nms(0.03, 30.0 / 180 * np.pi).sort_by_score()
            n_final = len(gg)

            object_results.append(
                ObjectGraspResult(
                    object_id=int(oi),
                    object_index=i,
                    point_cloud=masked_pcd.copy(),
                    normals=masked_normals.copy(),
                    grasp_group_array=gg.grasp_group_array.copy(),
                    n_grasps_before_collision=n_before,
                    n_grasps_after_collision=n_after_col,
                    n_grasps_final=n_final,
                )
            )

        runtime_sec = time.perf_counter() - started
        return ZeroGraspOutput(objects=object_results, runtime_sec=runtime_sec)
