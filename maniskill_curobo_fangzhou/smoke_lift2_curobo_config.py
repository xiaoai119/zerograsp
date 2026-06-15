#!/usr/bin/env python3
"""Smoke-test the generated Lift2 right-arm cuRobo config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from .generate_lift2_curobo_config import DEFAULT_OUTPUT, build_config


DEFAULT_OUTPUT_ROOT = Path("maniskill_curobo_fangzhou/runs/lift2_curobo_config_smoke")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--write-config-if-missing", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if args.write_config_if_missing and not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(build_config(tcp_xyz=[0.155, 0.0, -0.020], tcp_quat_wxyz=[1, 0, 0, 0]), sort_keys=False),
            encoding="utf-8",
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = run_kinematics_smoke(config)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "lift2_right_arm_curobo_smoke.json"
    result["config"] = str(config_path)
    result["report"] = str(report_path)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_kinematics_smoke(config: dict[str, Any]) -> dict[str, Any]:
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    kin_cfg = KinematicsCfg.from_robot_yaml_file(config, tool_frames=["right_tcp"])
    robot = Kinematics(kin_cfg, compute_spheres=True)
    joint_names = list(robot.joint_names)
    default_position = robot.default_joint_state.position.detach().clone()
    if default_position.ndim == 1:
        default_position = default_position.unsqueeze(0)
    state = robot.compute_kinematics(
        JointState.from_position(default_position, joint_names=joint_names)
    )
    tcp_pose = state.tool_poses.get_link_pose("right_tcp")
    tcp_position = tcp_pose.position.detach().cpu().numpy().reshape(-1).tolist()
    tcp_quaternion = tcp_pose.quaternion.detach().cpu().numpy().reshape(-1).tolist()
    robot_spheres = getattr(state, "robot_spheres", None)
    sphere_shape = list(robot_spheres.shape) if robot_spheres is not None else None

    q_offset = default_position.clone()
    q_offset[0, -1] = q_offset[0, -1] + 0.15
    state_offset = robot.compute_kinematics(
        JointState.from_position(q_offset, joint_names=joint_names)
    )
    tcp_pose_offset = state_offset.tool_poses.get_link_pose("right_tcp")
    tcp_position_offset = tcp_pose_offset.position.detach().cpu().numpy().reshape(-1).tolist()

    torch.cuda.synchronize()
    return {
        "status": "ok",
        "joint_names": joint_names,
        "tool_frames": list(robot.tool_frames),
        "dof": len(joint_names),
        "default_joint_position": default_position.detach().cpu().numpy().reshape(-1).tolist(),
        "right_tcp_position": tcp_position,
        "right_tcp_quaternion_wxyz": tcp_quaternion,
        "right_tcp_position_after_last_joint_offset": tcp_position_offset,
        "robot_spheres_shape": sphere_shape,
    }


if __name__ == "__main__":
    raise SystemExit(main())
