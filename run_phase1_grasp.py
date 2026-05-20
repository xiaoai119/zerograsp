import argparse
import os
import sys
import json
import datetime
import gc
import numpy as np
import torch as th
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import robocasa
import robocasa.utils.gym_utils.gymnasium_groot
import gymnasium as gym
import cv2
import imageio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase1.scripts.generate_zerograsp_inputs import process_collected_data, generate_mask_from_depth, compute_camera_intrinsics
from phase1.scripts.zerograsp_inference import ZeroGraspInference, GraspPose
from phase1.scripts.rmpflow_planner import RMPflowPlanner, RMPConfig


def project_3d_to_2d(point_3d, K):
    """Project 3D point to 2D image coordinates using camera intrinsics."""
    if point_3d is None or K is None:
        return None
    K = np.array(K).reshape(3, 3) if isinstance(K, list) else K
    x, y, z = point_3d
    if z <= 0:
        return None
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    u = int(fx * x / z + cx)
    v = int(fy * y / z + cy)
    return (u, v)


def make_env_with_depth(env_name):
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    robots_name = "GR1ArmsAndWaistFourierHands"
    controller_configs = load_composite_controller_config(
        controller=None,
        robot=robots_name.split("_")[0],
    )
    
    controller_configs["type"] = "BASIC"
    controller_configs["composite_controller_specific_configs"] = {}
    controller_configs["control_delta"] = False

    camera_names = ["egoview", "robot0_agentview_center"]
    camera_widths = 1280
    camera_heights = 800

    raw_env_name = env_name.split("/")[-1].replace("_Env", "").replace("_GR1ArmsAndWaistFourierHands", "")
    env_kwargs = dict(
        env_name=raw_env_name,
        robots=robots_name.split("_"),
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=True,
    )

    env = robosuite.make(**env_kwargs)
    return env, camera_names, camera_widths, camera_heights


def get_segmentation_mask(env, cam_name="egoview", cam_width=1280, cam_height=800):
    """Get instance segmentation mask directly from MuJoCo simulation."""
    try:
        seg = env.sim.render(
            camera_name=cam_name,
            width=cam_width,
            height=cam_height,
            segmentation=True,
        )
        seg = seg[::-1, :, 1]
        return seg
    except Exception as e:
        print(f"[Phase1-Grasp] Failed to get segmentation: {e}")
        return None


def get_camera_pose(env, cam_name="egoview"):
    """Get camera position and rotation matrix in world coordinates."""
    try:
        cam_id = env.sim.model.camera_name2id(cam_name)
        
        cam_body_id = env.sim.model.cam_bodyid[cam_id]
        
        cam_pos_local = env.sim.model.cam_pos[cam_id].copy()
        cam_quat_local = env.sim.model.cam_quat[cam_id].copy()
        
        body_pos = env.sim.data.body_xpos[cam_body_id].copy()
        body_quat = env.sim.data.body_xquat[cam_body_id].copy()
        
        def quat_to_rotmat(q):
            w, x, y, z = q
            return np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])
        
        body_rot = quat_to_rotmat(body_quat)
        cam_rot_local = quat_to_rotmat(cam_quat_local)
        
        cam_pos_world = body_pos + body_rot @ cam_pos_local
        cam_rot_world = body_rot @ cam_rot_local
        
        return cam_pos_world, cam_rot_world
    except Exception as e:
        print(f"[Phase1-Grasp] Failed to get camera pose: {e}")
        return np.zeros(3), np.eye(3)


def execute_ik_trajectory(env, target_pos, n_steps=100,
                          capture_frames=False, frame_buffer=None,
                          global_frame_buffer=None, frame_skip=4):
    """Execute trajectory using online IK with RELATIVE action format (Isaac-GR00T compatible).

    Key fix: Use relative delta actions instead of absolute position control,
    matching the GR1 robot's expected action format from Isaac-GR00T.
    """
    robot = env.robots[0]
    right_ctrl = robot.part_controllers['right']

    frame_counter = 0
    last_obs = None

    for step in range(n_steps):
        current_pos = right_ctrl.origin_pos.copy()

        pos_error = target_pos - current_pos
        pos_error_norm = np.linalg.norm(pos_error)

        if pos_error_norm < 0.015:
            print(f"[IK] Converged at step {step+1}, error={pos_error_norm:.4f}")
            break

        J = right_ctrl.J_pos.copy()

        damping = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))

        gain = min(1.0, 0.1 / max(pos_error_norm, 0.01))
        dq = J_pinv @ (pos_error * gain)

        current_q = right_ctrl.joint_pos.copy()
        left_q = robot.part_controllers['left'].joint_pos.copy()
        torso_q = robot.part_controllers['torso'].joint_pos.copy()
        left_gripper = robot.part_controllers['left_gripper'].joint_pos.copy() if hasattr(robot.part_controllers, 'left_gripper') else np.array([-1.0])
        right_gripper = robot.part_controllers['right_gripper'].joint_pos.copy() if hasattr(robot.part_controllers, 'right_gripper') else np.array([-1.0])

        action = np.zeros(env.action_dim)
        action[:7] = dq  # ✅ FIX: Use relative delta action for right arm
        action[7:14] = np.zeros(7)  # Keep left arm stationary (delta=0)
        action[14:17] = np.zeros(3)  # Keep torso stationary (delta=0)
        if env.action_dim > 17:
            action[17] = 0.0  # Left gripper delta
            action[18] = 0.0  # Right gripper delta

        obs, reward, done, info = env.step(action)
        last_obs = obs

        if capture_frames:
            frame_counter += 1
            if frame_counter % frame_skip == 0:
                if frame_buffer is not None:
                    frame_rgb, _, _ = capture_frame(env, "egoview", obs=obs, get_seg=False)
                    if frame_rgb is not None:
                        small_frame = cv2.resize(frame_rgb, (640, 400))
                        frame_buffer.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))
                if global_frame_buffer is not None:
                    global_img, _, _ = capture_frame(env, "robot0_agentview_center", obs=obs, get_seg=False)
                    if global_img is not None:
                        small_frame = cv2.resize(global_img, (640, 400))
                        global_frame_buffer.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))

        if done:
            break

    return step + 1, last_obs


def close_gripper_online(env, grip_steps=30, capture_frames=False,
                         frame_buffer=None, global_frame_buffer=None, frame_skip=2):
    """Close gripper using relative action format."""
    robot = env.robots[0]
    frame_counter = 0
    last_obs = None

    for _ in range(grip_steps):
        action = np.zeros(env.action_dim)
        action[:7] = np.zeros(7)  # Right arm delta
        action[7:14] = np.zeros(7)  # Left arm delta
        action[14:17] = np.zeros(3)  # Torso delta
        if env.action_dim > 17:
            action[17] = 0.0  # Left gripper
            action[18] = -0.1  # ✅ FIX: Relative delta to close gripper

        obs, reward, done, info = env.step(action)
        last_obs = obs

        if capture_frames:
            frame_counter += 1
            if frame_counter % frame_skip == 0:
                if frame_buffer is not None:
                    frame_rgb, _, _ = capture_frame(env, "egoview", obs=obs, get_seg=False)
                    if frame_rgb is not None:
                        small_frame = cv2.resize(frame_rgb, (640, 400))
                        frame_buffer.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))
                if global_frame_buffer is not None:
                    global_img, _, _ = capture_frame(env, "robot0_agentview_center", obs=obs, get_seg=False)
                    if global_img is not None:
                        small_frame = cv2.resize(global_img, (640, 400))
                        global_frame_buffer.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))

    return last_obs


def execute_rmpflow_trajectory(env, trajectory, planner, capture_frames=False,
                               frame_buffer=None, global_frame_buffer=None, frame_skip=4):
    """Execute RMPflow-planned joint trajectory using relative actions.

    Args:
        env: Robosuite environment
        trajectory: List of joint positions from RMPflow planner
        planner: RMPflowPlanner instance (for EE pose updates)
        capture_frames: Whether to capture frames for GIF
        frame_buffer: Buffer to store ego-view frames
        global_frame_buffer: Buffer to store global-view frames
        frame_skip: Capture every Nth frame

    Returns:
        last_obs: Final observation after execution
    """
    robot = env.robots[0]
    frame_counter = 0
    last_obs = None

    print(f"[RMPflow] Executing trajectory with {len(trajectory)} steps")

    for i, target_joints in enumerate(trajectory):
        current_right_q = robot.part_controllers['right'].joint_pos.copy()
        current_left_q = robot.part_controllers['left'].joint_pos.copy()
        current_torso_q = robot.part_controllers['torso'].joint_pos.copy()

        target_right_q = target_joints[:7]
        target_left_q = target_joints[7:14] if len(target_joints) > 7 else current_left_q
        target_torso_q = target_joints[14:17] if len(target_joints) > 14 else current_torso_q

        delta_right = target_right_q - current_right_q
        delta_left = target_left_q - current_left_q
        delta_torso = target_torso_q - current_torso_q

        max_delta = 0.05
        delta_right = np.clip(delta_right, -max_delta, max_delta)
        delta_left = np.clip(delta_left, -max_delta, max_delta)
        delta_torso = np.clip(delta_torso, -max_delta, max_delta)

        action = np.zeros(env.action_dim)
        action[:7] = delta_right
        action[7:14] = delta_left
        action[14:17] = delta_torso
        if env.action_dim > 17:
            action[17] = 0.0
            action[18] = 0.0

        obs, reward, done, info = env.step(action)
        last_obs = obs

        if hasattr(planner, 'update_ee_pose_from_obs'):
            planner.update_ee_pose_from_obs(obs)

        if capture_frames:
            frame_counter += 1
            if frame_counter % frame_skip == 0:
                if frame_buffer is not None:
                    frame_rgb, _, _ = capture_frame(env, "egoview", obs=obs, get_seg=False)
                    if frame_rgb is not None:
                        small_frame = cv2.resize(frame_rgb, (640, 400))
                        frame_buffer.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))
                if global_frame_buffer is not None:
                    global_img, _, _ = capture_frame(env, "robot0_agentview_center", obs=obs, get_seg=False)
                    if global_img is not None:
                        small_frame = cv2.resize(global_img, (640, 400))
                        global_frame_buffer.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))

        if done:
            print(f"[RMPflow] Trajectory terminated early at step {i+1}")
            break

    print(f"[RMPflow] Executed {min(i+1, len(trajectory))} trajectory steps")
    return last_obs


def transform_grasp_to_world(grasp_pose, cam_pos, cam_rot):
    """Transform grasp pose from camera coordinates to world coordinates.
    
    ZeroGrasp uses OpenCV camera convention:
    - X: right
    - Y: down  
    - Z: forward (into the scene)
    
    MuJoCo camera convention:
    - X: right
    - Y: up
    - Z: backward (out of the scene)
    """
    zg_to_mj = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    pos_mj_cam = zg_to_mj @ grasp_pose.position
    world_pos = cam_rot @ pos_mj_cam + cam_pos
    
    rot_mj_cam = zg_to_mj @ grasp_pose.rotation
    world_rot = cam_rot @ rot_mj_cam
    
    approach_mj_cam = zg_to_mj @ grasp_pose.approach
    world_approach = cam_rot @ approach_mj_cam
    
    from phase1.scripts.zerograsp_inference import GraspPose
    world_grasp = GraspPose(
        position=world_pos,
        rotation=world_rot,
        width=grasp_pose.width,
        quality=grasp_pose.quality,
        depth=grasp_pose.depth if hasattr(grasp_pose, 'depth') else 0.02,
        object_id=grasp_pose.object_id if hasattr(grasp_pose, 'object_id') else -1,
    )
    world_grasp.approach = world_approach
    
    return world_grasp


def capture_frame(env, cam_name="egoview", cam_width=1280, cam_height=800, obs=None, get_seg=True):
    if obs is None:
        if hasattr(env, '_get_observations'):
            obs = env._get_observations(force_update=True)
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_get_observations'):
            obs = env.unwrapped._get_observations()
        elif hasattr(env, 'get_obs'):
            try:
                obs = env.get_obs()
            except:
                pass
    
    if obs is None:
        return None, None, None
    
    rgb_key = None
    depth_key = None
    
    if isinstance(obs, dict):
        for key in obs.keys():
            key_lower = key.lower()
            if cam_name in key:
                if 'depth' in key_lower:
                    depth_key = key
                elif 'video' in key_lower or 'image' in key_lower:
                    if rgb_key is None or 'pad' in key_lower:
                        rgb_key = key

    rgb_img = None
    depth_img = None
    seg_img = None

    if rgb_key and rgb_key in obs:
        img = obs[rgb_key]
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgb_img = np.copy(img[::-1, :, :])
        elif len(img.shape) == 2:
            rgb_img = np.copy(img[::-1, :])

    if depth_key and depth_key in obs:
        depth_img = np.copy(obs[depth_key][::-1, :])
    
    if get_seg and hasattr(env, 'sim'):
        seg_img = get_segmentation_mask(env, cam_name, cam_width, cam_height)

    return rgb_img, depth_img, seg_img


def check_grasp_success(env, obs=None):
    try:
        if obs is None:
            if hasattr(env, '_get_observations'):
                obs = env._get_observations(force_update=True)
            else:
                obs = env.unwrapped._get_observations()
        
        if isinstance(obs, dict):
            gripper_pos = obs.get("gripper_position", None)
            if gripper_pos is not None:
                return float(np.mean(gripper_pos)) < 0.03
            
            for key in obs.keys():
                if 'gripper' in key.lower() and 'pos' in key.lower():
                    return float(np.mean(obs[key])) < 0.03
    except Exception as e:
        print(f"[Phase1-Grasp] check_grasp_success error: {e}")
    return False


def run_phase1_grasp(
    env_name="gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    output_dir=None,
    checkpoint_path=None,
    config_path=None,
    n_episodes=3,
    max_episode_steps=720,
    use_zerograsp=True,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / f"phase1_grasp_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    default_ckpt = str(PROJECT_ROOT / "submodules" / "ZeroGrasp" / "checkpoints" / "mirage_cvpr2025" / "mirage" / "epoch=1-step=80000.ckpt")
    default_cfg = str(PROJECT_ROOT / "submodules" / "ZeroGrasp" / "configs" / "demo.yaml")

    if checkpoint_path is None:
        checkpoint_path = default_ckpt
    if config_path is None:
        config_path = default_cfg

    print(f"[Phase1-Grasp] Output: {output_dir}")
    print(f"[Phase1-Grasp] Env: {env_name}")
    print(f"[Phase1-Grasp] Checkpoint: {checkpoint_path}")
    print(f"[Phase1-Grasp] Use ZeroGrasp: {use_zerograsp}")

    rmp_config = RMPConfig()

    print("[Phase1-Grasp] Creating environment...")
    env, cam_names, cam_w, cam_h = make_env_with_depth(env_name)
    
    ego_cam_name = "egoview"
    global_cam_name = "robot0_agentview_center"

    results = []

    for ep_idx in range(n_episodes):
        ep_output = output_dir / f"episode_{ep_idx:04d}"
        ep_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[Phase1-Grasp] Episode {ep_idx + 1}/{n_episodes}")
        print(f"{'='*60}")

        obs_result = env.reset()
        if isinstance(obs_result, tuple):
            obs, info = obs_result
        else:
            obs = obs_result
        rgb_img, depth_img, _ = capture_frame(env, ego_cam_name, obs=obs, get_seg=False)
        
        cv2.imwrite(str(ep_output / "initial_rgb.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        rgb_path = str(ep_output / "initial_rgb.png")
        seg_img = get_segmentation_mask(env, ego_cam_name, cam_w, cam_h)

        if rgb_img is None:
            print("[Phase1-Grasp] WARNING: Failed to capture initial frame, skipping episode")
            continue

        has_depth = depth_img is not None

        depth_npy_path = str(ep_output / "initial_depth.npy")
        mask_path = str(ep_output / "initial_mask.png")
        cam_info_path = str(ep_output / "camera.json")
        depth_zg_path = str(ep_output / "initial_depth.png")
        
        if has_depth:
            np.save(depth_npy_path, depth_img)

            K = compute_camera_intrinsics(fov_deg=77, width=1280, height=800)
            K_flat = np.array(K).flatten().tolist()
            with open(cam_info_path, "w") as f:
                json.dump({"cam_K": K_flat}, f)

            depth_near = 0.02
            depth_far = 10.0
            depth_buf = np.squeeze(depth_img).astype(np.float64)
            depth_buf = np.clip(depth_buf, 0.0, 0.999999)
            depth_linear = (depth_near * depth_far) / (depth_far - depth_buf * (depth_far - depth_near))
            depth_metric = depth_linear.astype(np.float32)
            depth_mm = (depth_metric * 1000).astype(np.uint16)
            cv2.imwrite(depth_zg_path, depth_mm)
            
            depth_vis = ((depth_metric - depth_metric.min()) / (depth_metric.max() - depth_metric.min() + 1e-6) * 255).astype(np.uint8)
            cv2.imwrite(str(ep_output / "initial_depth_vis.png"), depth_vis)
            print(f"[Phase1-Grasp] Saved depth visualization: range=[{depth_metric.min():.3f}, {depth_metric.max():.3f}]m")
        else:
            print("[Phase1-Grasp] No depth data available, skipping depth processing")

        if seg_img is not None:
            raw_seg = np.squeeze(seg_img).astype(np.int32)
            
            unique_ids = np.unique(raw_seg)
            unique_ids = unique_ids[unique_ids > 0]
            
            print(f"[Phase1-Grasp] Raw segmentation: {len(unique_ids)} geom instances")
            print(f"[Phase1-Grasp] Unique geom IDs: {unique_ids[:10]}{'...' if len(unique_ids) > 10 else ''}")
            
            target_geom_ids = set()
            try:
                env_unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
                
                if hasattr(env_unwrapped, 'objects'):
                    print(f"[Phase1-Grasp] env.objects keys: {list(env_unwrapped.objects.keys())}")
                    if 'obj' in env_unwrapped.objects:
                        obj = env_unwrapped.objects['obj']
                        print(f"[Phase1-Grasp] Target object: name={obj.name}, root_body={getattr(obj, 'root_body', 'N/A')}")
                        if hasattr(obj, 'geoms') and obj.geoms is not None:
                            print(f"[Phase1-Grasp] Object geoms: {obj.geoms}")
                            for geom_name in obj.geoms:
                                geom_id = env.sim.model.geom_name2id(geom_name)
                                target_geom_ids.add(geom_id)
                            print(f"[Phase1-Grasp] Target geom IDs from geoms: {target_geom_ids}")
                        elif hasattr(obj, 'root_body'):
                            body_id = env.sim.model.body_name2id(obj.root_body)
                            body_geoms = env.sim.model.body_geomadr[body_id]
                            if body_geoms >= 0:
                                body_geom_num = env.sim.model.body_geomnum[body_id]
                                for i in range(body_geom_num):
                                    target_geom_ids.add(body_geoms + i)
                            print(f"[Phase1-Grasp] Target geom IDs from body: {target_geom_ids}")
                    else:
                        print(f"[Phase1-Grasp] 'obj' not in env.objects")
                else:
                    print(f"[Phase1-Grasp] env has no 'objects' attribute")
                    if hasattr(env_unwrapped, 'object'):
                        print(f"[Phase1-Grasp] env has 'object' attribute")
            except Exception as e:
                import traceback
                print(f"[Phase1-Grasp] Could not get target object geom IDs: {e}")
                traceback.print_exc()
            
            instance_mask = np.zeros((800, 1280), dtype=np.uint8)
            
            if len(target_geom_ids) > 0:
                obj_id_new = 1
                for geom_id in target_geom_ids:
                    instance_mask[raw_seg == geom_id] = obj_id_new
                n_objs = 1
                print(f"[Phase1-Grasp] Using target object geom IDs: {target_geom_ids}")
            else:
                instance_areas = []
                for obj_id in unique_ids:
                    count = (raw_seg == obj_id).sum()
                    instance_areas.append((count, obj_id))
                
                instance_areas.sort(reverse=True)
                
                min_obj_area = 10000
                max_obj_area = 80000
                
                filtered_instances = []
                for area, obj_id in instance_areas:
                    if min_obj_area <= area <= max_obj_area:
                        filtered_instances.append((area, obj_id))
                        percentage = area / raw_seg.size * 100
                        print(f"[Phase1-Grasp]   Keeping instance {obj_id}: {area} pixels ({percentage:.1f}%)")
                
                obj_id_new = 1
                for area, old_id in filtered_instances[:3]:
                    instance_mask[raw_seg == old_id] = obj_id_new
                    obj_id_new += 1
                n_objs = obj_id_new - 1
            
            print(f"[Phase1-Grasp] Final mask: {n_objs} object instance(s)")
        else:
            print("[Phase1-Grasp] WARNING: No segmentation data, creating dummy mask")
            instance_mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
            instance_mask[:, :] = 1
            n_objs = 1
        
        print(f"[Phase1-Grasp] Final mask: {n_objs} object(s)")
        
        cv2.imwrite(mask_path, instance_mask)
        
        mask_vis = (instance_mask * (255 // max(n_objs, 1))).astype(np.uint8)
        cv2.imwrite(str(ep_output / "initial_mask_vis.png"), mask_vis)
        print(f"[Phase1-Grasp] Saved mask visualization: {n_objs} objects detected")

        if has_depth:
            print(f"[Phase1-Grasp] Captured frame, depth range=[{depth_metric.min():.3f}, {depth_metric.max():.3f}]m, mask objects={n_objs}")
        else:
            print(f"[Phase1-Grasp] Captured frame (no depth data), mask objects={n_objs}")

        current_joint_pos = np.zeros(14)
        try:
            current_joint_pos[:env.action_dim] = env._joint_positions.copy()
        except Exception:
            pass

        try:
            obstacle_pcd = None
            obs_3d = env._get_observations(force_update=True)
            if "egoview_pointcloud" in obs_3d or "pointcloud" in obs_3d:
                pc_key = "egoview_pointcloud" if "egoview_pointcloud" in obs_3d else "pointcloud"
                pc_data = obs_3d[pc_key]
                if hasattr(pc_data, 'shape') and len(pc_data.shape) >= 2:
                    obstacle_pcd = pc_data.reshape(-1, 3)[:500].copy()
        except Exception:
            obstacle_pcd = None

        if obstacle_pcd is not None:
            pcd_path = str(ep_output / "obstacle_pcd.npy")
            np.save(pcd_path, obstacle_pcd)

        cam_pos, cam_rot = get_camera_pose(env, "egoview")
        cam_pose_path = str(ep_output / "camera_pose.npy")
        np.save(cam_pose_path, {"position": cam_pos, "rotation": cam_rot})
        print(f"[Phase1-Grasp] Camera pose: pos={cam_pos}, rot_diag={np.diag(cam_rot)}")

        grasp_poses = []
        if use_zerograsp and has_depth:
            zg_output = ep_output / "zerograsp_output"
            zg_output.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy(rgb_path, str(zg_output / "input_rgb.png"))
            shutil.copy(depth_zg_path, str(zg_output / "input_depth.png"))
            shutil.copy(mask_path, str(zg_output / "input_mask.png"))
            shutil.copy(cam_info_path, str(zg_output / "camera.json"))

            print("[Phase1-Grasp] Initializing ZeroGrasp...")
            zerograsp = ZeroGraspInference(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                img_height=800,
                img_width=1280,
                top_k=3,
            )

            print(f"[Phase1-Grasp] Running ZeroGrasp inference...")
            try:
                grasp_poses = zerograsp.infer(
                    rgb_path=rgb_path,
                    depth_path=depth_zg_path,
                    mask_path=mask_path,
                    camera_info_path=cam_info_path,
                    depth_scale=1.0,
                )
                
                if len(grasp_poses) > 0:
                    cam_pose_data = np.load(cam_pose_path, allow_pickle=True).item()
                    cam_pos = cam_pose_data["position"]
                    cam_rot = cam_pose_data["rotation"]
                    
                    print(f"[Phase1-Grasp] Transforming {len(grasp_poses)} grasps from camera to world coordinates...")
                    print(f"[Phase1-Grasp]   Camera position: {cam_pos}")
                    print(f"[Phase1-Grasp]   First grasp (camera): pos={grasp_poses[0].position}")
                    
                    camera_grasp_poses = [gp for gp in grasp_poses]
                    
                    world_grasp_poses = []
                    for gp in grasp_poses:
                        world_gp = transform_grasp_to_world(gp, cam_pos, cam_rot)
                        world_grasp_poses.append(world_gp)
                    
                    grasp_poses = world_grasp_poses
                    print(f"[Phase1-Grasp]   First grasp (world): pos={grasp_poses[0].position}")
                    
            except Exception as e:
                print(f"[Phase1-Grasp] ZeroGrasp inference failed: {e}")
                print("[Phase1-Grasp] Falling back to mock grasp pose...")
                from phase1.scripts.zerograsp_inference import GraspPose
                mock_grasp = GraspPose(
                    position=np.array([-0.3, 0.2, 0.6]),
                    rotation=np.eye(3),
                    width=0.08,
                    quality=0.5
                )
                mock_grasp.approach = np.array([0, 0, 1])
                grasp_poses = [mock_grasp]

            grasp_results = []
            for i, gp in enumerate(grasp_poses):
                grasp_results.append({
                    "rank": i + 1,
                    "position": gp.position.tolist(),
                    "rotation": gp.rotation.tolist(),
                    "width": float(gp.width),
                    "height": 0.02,
                    "depth": float(gp.depth) if hasattr(gp, 'depth') else 0.02,
                    "quality": float(gp.quality),
                    "approach": gp.approach.tolist() if hasattr(gp, 'approach') and gp.approach is not None else [0, 0, 1],
                    "object_id": int(gp.object_id) if hasattr(gp, 'object_id') else -1,
                })
            
            with open(str(ep_output / "grasp_poses.json"), "w") as f:
                json.dump(grasp_results, f, indent=2)
            with open(str(zg_output / "grasp_poses.json"), "w") as f:
                json.dump(grasp_results, f, indent=2)
            print(f"[Phase1-Grasp] Saved {len(grasp_poses)} grasp poses to grasp_poses.json")
            
            if len(grasp_poses) > 0:
                grasp_arrays = []
                for gp in grasp_poses:
                    grasp_arrays.append(gp.to_graspnet_format())
                grasp_npy = np.stack(grasp_arrays, axis=0)
                np.save(str(ep_output / "grasp_poses.npy"), grasp_npy)
                np.save(str(zg_output / "grasp_poses.npy"), grasp_npy)
                print(f"[Phase1-Grasp] Saved grasp poses in graspnet format to grasp_poses.npy")
            
            vis_grasp_poses = camera_grasp_poses if 'camera_grasp_poses' in dir() else grasp_poses
            if len(vis_grasp_poses) > 0:
                grasp_vis_img = rgb_img.copy()
                for i, gp in enumerate(vis_grasp_poses[:5]):
                    pos_2d = project_3d_to_2d(gp.position, K)
                    if pos_2d is not None:
                        color = (0, 255, 0) if i == 0 else (255, 255, 0)
                        
                        approach_vec = gp.approach if hasattr(gp, 'approach') else np.array([0, 0, 1])
                        approach_end_3d = gp.position + approach_vec * 0.08
                        approach_2d = project_3d_to_2d(approach_end_3d, K)
                        if approach_2d is not None:
                            cv2.arrowedLine(grasp_vis_img, pos_2d, approach_2d, color, 2, tipLength=0.3)
                    
                    if hasattr(gp, 'rotation') and gp.rotation is not None:
                        R = gp.rotation.reshape(3, 3)
                        binormal = R[:, 1]
                        width_half = gp.width / 2
                        
                        finger1_3d = gp.position + binormal * width_half
                        finger2_3d = gp.position - binormal * width_half
                        
                        finger1_2d = project_3d_to_2d(finger1_3d, K)
                        finger2_2d = project_3d_to_2d(finger2_3d, K)
                        
                        if finger1_2d is not None and finger2_2d is not None:
                            cv2.line(grasp_vis_img, finger1_2d, finger2_2d, color, 2)
                            
                            finger_size = 8
                            cv2.rectangle(grasp_vis_img, 
                                         (finger1_2d[0]-finger_size//2, finger1_2d[1]-finger_size//2),
                                         (finger1_2d[0]+finger_size//2, finger1_2d[1]+finger_size//2),
                                         color, -1)
                            cv2.rectangle(grasp_vis_img,
                                         (finger2_2d[0]-finger_size//2, finger2_2d[1]-finger_size//2),
                                         (finger2_2d[0]+finger_size//2, finger2_2d[1]+finger_size//2),
                                         color, -1)
                            
                            depth_len = getattr(gp, 'depth', 0.02) * 500
                            depth_end1_3d = finger1_3d - approach_vec * depth_len
                            depth_end2_3d = finger2_3d - approach_vec * depth_len
                            depth_end1_2d = project_3d_to_2d(depth_end1_3d, K)
                            depth_end2_2d = project_3d_to_2d(depth_end2_3d, K)
                            if depth_end1_2d is not None:
                                cv2.line(grasp_vis_img, finger1_2d, depth_end1_2d, color, 1)
                            if depth_end2_2d is not None:
                                cv2.line(grasp_vis_img, finger2_2d, depth_end2_2d, color, 1)
                    
                    cv2.circle(grasp_vis_img, pos_2d, 4, color, -1)
                    
                    label = f"#{i+1} q={gp.quality:.2f} w={gp.width*100:.1f}cm"
                    cv2.putText(grasp_vis_img, label, 
                               (pos_2d[0]+15, pos_2d[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imwrite(str(ep_output / "grasp_visualization.png"), cv2.cvtColor(grasp_vis_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(zg_output / "grasp_visualization.png"), cv2.cvtColor(grasp_vis_img, cv2.COLOR_RGB2BGR))
            print(f"[Phase1-Grasp] Saved grasp visualization to grasp_visualization.png")
            print(f"[Phase1-Grasp] ZeroGrasp outputs saved to: {zg_output}")

        if use_zerograsp and has_depth:
            del zerograsp
        gc.collect()
        th.cuda.empty_cache()

        best_grasp = None
        if use_zerograsp and len(grasp_poses) > 0:
            best_grasp = grasp_poses[0]
            print(f"[Phase1-Grasp] Best grasp: score={best_grasp.quality:.4f}, pos={best_grasp.position.tolist()}, width={best_grasp.width:.4f}")
        elif use_zerograsp:
            print("[Phase1-Grasp] No grasps found from ZeroGrasp")

        if best_grasp is None:
            print("[Phase1-Grasp] No grasp available, skipping episode")
            continue

        print(f"[Phase1-Grasp] Executing grasp trajectory...")

        robot = env.robots[0]
        print(f"[Phase1-Grasp] Controller type: {robot.composite_controller.name}")

        gif_frames = []
        global_gif_frames = []

        if rgb_img is not None:
            small_frame = cv2.resize(rgb_img, (640, 400))
            gif_frames.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))

        global_rgb, _, _ = capture_frame(env, global_cam_name, obs=obs, get_seg=False)
        if global_rgb is not None:
            small_frame = cv2.resize(global_rgb, (640, 400))
            global_gif_frames.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))

        use_rmpflow = True
        try:
            print("[Phase1-Grasp] Initializing RMPflow planner...")
            planner = RMPflowPlanner(env, rmp_config)
            planner.update_ee_pose_from_obs(obs)

            print(f"[Phase1-Grasp] Planning full grasp trajectory with RMPflow...")
            plan_result = planner.plan_full_grasp(
                best_grasp,
                current_joint_pos[:14],
                obstacle_pcd=obstacle_pcd
            )

            if plan_result and 'pregrasp_traj' in plan_result and len(plan_result['pregrasp_traj']) > 0:
                print(f"[Phase1-Grasp] ✅ RMPflow planned {len(plan_result['pregrasp_traj'])} pre-grasp steps")
                obs = execute_rmpflow_trajectory(env, plan_result['pregrasp_traj'], planner,
                                                capture_frames=True, frame_buffer=gif_frames,
                                                global_frame_buffer=global_gif_frames, frame_skip=4)

                print(f"[Phase1-Grasp] ✅ RMPflow executing grasp trajectory ({len(plan_result.get('grasp_traj', []))} steps)")
                if 'grasp_traj' in plan_result and len(plan_result['grasp_traj']) > 0:
                    obs = execute_rmpflow_trajectory(env, plan_result['grasp_traj'], planner,
                                                    capture_frames=True, frame_buffer=gif_frames,
                                                    global_frame_buffer=global_gif_frames, frame_skip=4)

                print("[Phase1-Grasp] Closing gripper...")
                obs = close_gripper_online(env, grip_steps=30, capture_frames=True,
                                     frame_buffer=gif_frames, global_frame_buffer=global_gif_frames)

                if 'lift_traj' in plan_result and len(plan_result['lift_traj']) > 0:
                    print(f"[Phase1-Grasp] ✅ RMPflow executing lift trajectory ({len(plan_result['lift_traj'])} steps)")
                    obs = execute_rmpflow_trajectory(env, plan_result['lift_traj'], planner,
                                                    capture_frames=True, frame_buffer=gif_frames,
                                                    global_frame_buffer=global_gif_frames, frame_skip=4)

                execution_mode = "rmpflow_planned"
                print("[Phase1-Grasp] ✅ RMPflow trajectory execution completed")
            else:
                print("[Phase1-Grasp] ⚠️ RMPflow planning failed, falling back to IK")
                use_rmpflow = False

        except Exception as e:
            print(f"[Phase1-Grasp] ⚠️ RMPflow execution error: {e}")
            print("[Phase1-Grasp] Falling back to IK-based execution...")
            use_rmpflow = False

        if not use_rmpflow:
            pregrasp_pos = best_grasp.position + np.array([0, 0, 0.15])
            print(f"[Phase1-Grasp] Moving to pre-grasp position: {pregrasp_pos}")
            steps, obs = execute_ik_trajectory(env, pregrasp_pos, n_steps=100,
                                       capture_frames=True, frame_buffer=gif_frames,
                                       global_frame_buffer=global_gif_frames, frame_skip=4)
            print(f"[Phase1-Grasp] Pre-grasp completed in {steps} steps")

            print(f"[Phase1-Grasp] Moving to grasp position: {best_grasp.position}")
            steps, obs = execute_ik_trajectory(env, best_grasp.position, n_steps=100,
                                       capture_frames=True, frame_buffer=gif_frames,
                                       global_frame_buffer=global_gif_frames, frame_skip=4)
            print(f"[Phase1-Grasp] Grasp completed in {steps} steps")

            print("[Phase1-Grasp] Closing gripper...")
            obs = close_gripper_online(env, grip_steps=30, capture_frames=True,
                                 frame_buffer=gif_frames, global_frame_buffer=global_gif_frames)

            lift_pos = best_grasp.position + np.array([0, 0, 0.25])
            print(f"[Phase1-Grasp] Lifting to: {lift_pos}")
            steps, obs = execute_ik_trajectory(env, lift_pos, n_steps=50,
                                       capture_frames=True, frame_buffer=gif_frames,
                                       global_frame_buffer=global_gif_frames, frame_skip=4)
            print(f"[Phase1-Grasp] Lift completed in {steps} steps")

            execution_mode = "online_ik_fallback"

        final_rgb, final_depth, _ = capture_frame(env, ego_cam_name, obs=obs, get_seg=False)
        if final_rgb is not None:
            cv2.imwrite(str(ep_output / "final_rgb.png"), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
            small_frame = cv2.resize(final_rgb, (640, 400))
            gif_frames.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))
        
        final_global, _, _ = capture_frame(env, global_cam_name, obs=obs, get_seg=False)
        if final_global is not None:
            cv2.imwrite(str(ep_output / "final_rgb_global.png"), cv2.cvtColor(final_global, cv2.COLOR_RGB2BGR))
            small_frame = cv2.resize(final_global, (640, 400))
            global_gif_frames.append(cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))

        success = check_grasp_success(env, obs=obs)
        print(f"[Phase1-Grasp] Grasp success: {success}")

        if len(gif_frames) > 0:
            gif_path = str(ep_output / "grasp_execution_egoview.gif")
            imageio.mimsave(gif_path, gif_frames, fps=5)
            print(f"[Phase1-Grasp] Ego-view GIF saved: {gif_path} ({len(gif_frames)} frames)")
        
        if len(global_gif_frames) > 0:
            gif_path = str(ep_output / "grasp_execution_global.gif")
            imageio.mimsave(gif_path, global_gif_frames, fps=5)
            print(f"[Phase1-Grasp] Global-view GIF saved: {gif_path} ({len(global_gif_frames)} frames)")

        gc.collect()
        th.cuda.empty_cache()

        ep_result = {
            "episode_id": ep_idx,
            "success": bool(success),
            "execution_mode": execution_mode if 'execution_mode' in dir() else "unknown",
            "best_grasp_score": float(best_grasp.quality),
            "grasp_position": best_grasp.position.tolist(),
            "grasp_rotation": best_grasp.rotation.tolist(),
            "grasp_width": float(best_grasp.width),
            "n_grasps_found": len(grasp_poses),
            "plan_pregrasp_steps": 100,
            "plan_grasp_steps": 100,
            "plan_lift_steps": 50,
        }
        results.append(ep_result)

        with open(str(ep_output / "result.json"), "w") as f:
            json.dump(ep_result, f, indent=2)

    env.close()
    del env
    gc.collect()
    th.cuda.empty_cache()

    summary = {
        "timestamp": timestamp,
        "env_name": env_name,
        "n_episodes": n_episodes,
        "n_success": sum(1 for r in results if r["success"]),
        "success_rate": sum(1 for r in results if r["success"]) / max(len(results), 1),
        "episodes": results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[Phase1-Grasp] === SUMMARY ===")
    print(f"[Phase1-Grasp] Episodes: {n_episodes}")
    print(f"[Phase1-Grasp] Success: {summary['n_success']}/{len(results)} ({summary['success_rate']*100:.1f}%)")
    print(f"[Phase1-Grasp] Results saved to: {output_dir}")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: ZeroGrasp Grasp Pipeline")
    parser.add_argument("--env_name", type=str,
                        default="gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
                        help="Environment name")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to ZeroGrasp checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to ZeroGrasp config YAML")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Number of episodes")
    parser.add_argument("--max_episode_steps", type=int, default=720,
                        help="Max steps per episode")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--use_zerograsp", action="store_true", default=True,
                        help="Use ZeroGrasp for grasp detection")
    parser.add_argument("--no_zerograsp", action="store_false", dest="use_zerograsp",
                        help="Do not use ZeroGrasp")
    args = parser.parse_args()

    run_phase1_grasp(
        env_name=args.env_name,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        use_zerograsp=args.use_zerograsp,
    )
