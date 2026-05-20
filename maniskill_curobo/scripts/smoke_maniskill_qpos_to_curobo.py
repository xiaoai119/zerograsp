#!/usr/bin/env python3
"""Plan with cuRobo from the current ManiSkill Panda arm qpos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def main() -> int:
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    output_dir = Path(__file__).resolve().parents[1] / "smoke_tests" / "bridge"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PickSingleYCB-v1",
        render_mode="rgb_array",
        control_mode="pd_joint_pos",
        robot_uids="panda",
        obs_mode="state",
        max_episode_steps=5,
    )
    try:
        env.reset(seed=1)
        robot = env.unwrapped.agent.robot
        active_joint_names = [joint.name for joint in robot.get_active_joints()]
        qpos = robot.get_qpos().detach().cpu().numpy().reshape(-1)
    finally:
        env.close()

    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)

    name_to_qpos = dict(zip(active_joint_names, qpos))
    arm_qpos = np.array([name_to_qpos[name] for name in planner.joint_names], dtype=np.float32)
    q_start = JointState.from_position(
        torch.as_tensor(arm_qpos, device="cuda", dtype=torch.float32).unsqueeze(0),
        joint_names=planner.joint_names,
    )

    goal_pose = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32),
    )
    result = planner.plan_pose(goal_pose, q_start)
    if result is None or result.success is None or not bool(result.success.any()):
        status = getattr(result, "status", "unknown")
        raise RuntimeError(f"cuRobo planning from ManiSkill qpos failed: {status}")

    interpolated = result.get_interpolated_plan()
    positions = interpolated.position.detach().cpu().numpy()
    out_path = output_dir / "maniskill_qpos_to_curobo_plan.npz"
    np.savez(
        out_path,
        active_joint_names=np.array(active_joint_names),
        curobo_joint_names=np.array(planner.joint_names),
        trajectory_joint_names=np.array(interpolated.joint_names),
        maniskill_qpos=qpos,
        arm_qpos=arm_qpos,
        trajectory=positions,
    )
    print(f"active_joint_names={active_joint_names}")
    print(f"curobo_joint_names={planner.joint_names}")
    print(f"trajectory_joint_names={interpolated.joint_names}")
    print(f"arm_qpos={np.round(arm_qpos, 4).tolist()}")
    print(f"trajectory_shape={positions.shape}")
    print(f"bridge_plan={out_path}")
    print("maniskill_qpos_to_curobo_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
