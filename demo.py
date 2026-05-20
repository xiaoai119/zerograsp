import os

import torch as th
import open3d as o3d
import numpy as np
import torch.nn.functional as F
from graspnetAPI import GraspGroup

from main import BaseTrainer
from zerograsp.utils.dataset import fetch_data
from zerograsp.utils.array_bridge import numpy_to_torch, torch_to_numpy
from zerograsp.utils.math import unnormalize_pts, rotation_6d_to_matrix
from zerograsp.utils.config import parse_config
from zerograsp.nets.utils import get_xyz_from_octree
from zerograsp.utils.collision_detector import ModelFreeCollisionDetector

GRASP_MAX_WIDTH = 0.1
GRASP_MAX_DEPTH = 0.04


def main():
    config = parse_config()
    config.update_octree = True
    device = "cuda" if th.cuda.is_available() else "cpu"
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading a model...")
    model = BaseTrainer.load_from_checkpoint(
        config.checkpoint, config=config, strict=False, map_location=device
    )
    model.to(device)
    model.eval()

    img_id = os.path.basename(config.img_path).split(".")[0]

    print("Fetching data...")
    batch = fetch_data(
        config.img_path,
        config.depth_path,
        config.mask_path,
        config.camera_info_path,
        config,
        1.0,
        device=device,
    )
    grid_res = 1 << config.min_lod
    with th.no_grad():
        print("Running inference...")
        output = model.model(batch)
        z_min = batch[-2][0]
        pts_3d_in = batch[3][0]
        rays_3d = batch[4][0]

        octrees_out = output["octrees_out"]
        pcd, batch_id = get_xyz_from_octree(
            octrees_out, config.max_lod, nempty=True, return_batch=True
        )
        pcd = unnormalize_pts(pcd, z_min, config.grid_size, grid_res)
        normals = octrees_out.normals[config.max_lod]
        signal = octrees_out.features[config.max_lod]
        sdf = signal[:, :1]
        batch_id = torch_to_numpy(batch_id).reshape(-1)
        pcd = torch_to_numpy(pcd).reshape(-1, 3)
        normals = torch_to_numpy(F.normalize(normals, dim=-1)).reshape(-1, 3)
        sdf = torch_to_numpy(sdf).reshape(-1, 1)
        pcd = pcd - normals * sdf

        obj_ids = th.unique(pts_3d_in.labels, sorted=True)
        grasp_preds = []
        for i, oi in enumerate(obj_ids):
            mask = batch_id == i
            print(
                f"Processing object index {i} (label={int(oi)}) with {int(mask.sum())} reconstructed points.",
                flush=True,
            )
            if not np.any(mask):
                print(f"Skipping object index {i} because no reconstructed points were assigned.")
                continue
            masked_pcd = np.ascontiguousarray(pcd[mask], dtype=np.float64)
            masked_normals = np.ascontiguousarray(normals[mask], dtype=np.float64)
            masked_colors = np.ascontiguousarray(
                np.clip((masked_normals + 1.0) / 2.0, 0.0, 1.0),
                dtype=np.float64,
            )
            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(masked_pcd)
            pcd_vis.normals = o3d.utility.Vector3dVector(masked_normals)
            pcd_vis.colors = o3d.utility.Vector3dVector(masked_colors)
            new_pcd_path = os.path.join(output_dir, f"{img_id}_{i}.ply")
            o3d.io.write_point_cloud(new_pcd_path, pcd_vis)
            print("pcd is exported to", new_pcd_path)
            masked_pcd_m = np.ascontiguousarray(masked_pcd, dtype=np.float32)

            # Grasp Poses
            masked_signal = signal[mask, 1:]
            # quality = (masked_signal[valid, :1] * masked_signal[valid, 1:2]).cpu().numpy()
            # quality = (masked_signal[valid, :1] * masked_signal[valid, 1:2]).cpu().numpy()
            quality = torch_to_numpy(masked_signal[:, :1]).reshape(-1, 1)
            tangent = masked_signal[:, 2:5]
            gnormal = masked_signal[:, 5:8]
            R = torch_to_numpy(
                rotation_6d_to_matrix(th.cat([-gnormal, tangent], dim=-1))
            ).reshape(-1, 9)
            depth = torch_to_numpy(masked_signal[:, 8:9]).reshape(-1, 1)
            width = torch_to_numpy(masked_signal[:, 9:10]).reshape(-1, 1)
            translation = masked_pcd_m.reshape(-1, 3) / 1000.0
            height = np.full((quality.shape[0], 1), 0.02, dtype=quality.dtype)
            grasp_preds = np.concatenate(
                [
                    quality,
                    np.clip(width * GRASP_MAX_WIDTH, 0.0, GRASP_MAX_WIDTH),
                    height,
                    np.clip(depth * GRASP_MAX_DEPTH, 0.0, GRASP_MAX_DEPTH),
                    R,
                    translation,
                    -1 * np.ones((quality.shape[0], 1), dtype=quality.dtype),
                ],
                axis=-1,
            )

            gg = GraspGroup(grasp_preds)
            gg = gg.sort_by_score()
            depth_pcd = torch_to_numpy(rays_3d.reshape(-1, 3)[::5])

            print("Number of grasps before collision detection", len(gg), flush=True)

            with th.no_grad():
                cloud = numpy_to_torch(pcd / 1000.0, device=device).float()
                cloud_nrm = numpy_to_torch(normals, device=device).float()
                depth_cloud = numpy_to_torch(
                    depth_pcd / 1000.0, device=device
                ).float()
                mfcdetector = ModelFreeCollisionDetector(cloud, cloud_nrm, depth_cloud)
                collision_mask, delta_width, refined_depth = mfcdetector.detect(gg)
                gg.grasp_group_array[:, 1] = gg.grasp_group_array[:, 1] + delta_width
                gg.grasp_group_array[:, 3] = refined_depth

            if (~collision_mask).sum() > 0:
                gg = gg[~collision_mask]

            gg = gg.nms(0.03, 30.0 / 180 * np.pi).sort_by_score()

            print("Number of grasps after collision detection", len(gg), flush=True)
            gg.save_npy(os.path.join(output_dir, f"{img_id}_{i}.grasp.npy"))
            print("grasp pose is exported", flush=True)

        print("saved!", flush=True)


if __name__ == "__main__":
    main()
