#!/usr/bin/env python3
"""Generate a first cuRobo robot config for Lift2's right arm.

This is intentionally a minimal, right-arm-only configuration.  It keeps the
mobile base, left arm, and right gripper fingers locked while cuRobo plans the
lift joint plus the six right arm joints.  It can also generate a fixed-lift
variant that locks the lift joint and only plans the six revolute arm joints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml

from .lift2_constants import (
    LIFT2_CUROBO_SAFE_REST_QPOS,
    LIFT2_JOINT_NAMES,
    LIFT2_RIGHT_TCP_QUAT_WXYZ_RIGHT_LINK26,
    LIFT2_RIGHT_TCP_XYZ_RIGHT_LINK26,
)
from .urdf_adapter import ensure_visual_urdf, load_collision_spheres


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_ROOT / "config" / "lift2_right_arm_curobo.yml"
DEFAULT_MANIFEST = PACKAGE_ROOT / "config" / "lift2_right_arm_curobo_manifest.json"
DEFAULT_FIXED_LIFT_OUTPUT = PACKAGE_ROOT / "config" / "lift2_right_arm_fixed_lift_curobo.yml"
DEFAULT_FIXED_LIFT_MANIFEST = (
    PACKAGE_ROOT / "config" / "lift2_right_arm_fixed_lift_curobo_manifest.json"
)

RIGHT_ARM_JOINT_NAMES = [
    "joint4",
    "right_joint21",
    "right_joint22",
    "right_joint23",
    "right_joint24",
    "right_joint25",
    "right_joint26",
]
RIGHT_ARM_FIXED_LIFT_JOINT_NAMES = [
    "right_joint21",
    "right_joint22",
    "right_joint23",
    "right_joint24",
    "right_joint25",
    "right_joint26",
]
RIGHT_GRIPPER_JOINT_NAMES = ["right_joint27", "right_joint28"]
RIGHT_COLLISION_LINK_NAMES = [
    "right_link21",
    "right_link22",
    "right_link23",
    "right_link24",
    "right_link25",
    "right_link26",
    "right_link27",
    "right_link28",
]

# Midpoint between the two finger meshes 1 cm behind their tips, expressed in
# right_link26. This stays aligned with the fixed right_tcp link inserted into
# ManiSkill execution URDFs by urdf_adapter.py.
DEFAULT_TCP_XYZ_RIGHT_LINK26 = LIFT2_RIGHT_TCP_XYZ_RIGHT_LINK26.tolist()
DEFAULT_TCP_QUAT_WXYZ_RIGHT_LINK26 = LIFT2_RIGHT_TCP_QUAT_WXYZ_RIGHT_LINK26.tolist()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--fixed-lift-highest",
        action="store_true",
        help="Lock joint4 at the requested lift value and only plan the six revolute right-arm joints.",
    )
    parser.add_argument(
        "--lift-joint-value",
        type=float,
        default=0.46,
        help="Locked joint4 value for --fixed-lift-highest, in meters.",
    )
    parser.add_argument(
        "--tcp-xyz",
        type=float,
        nargs=3,
        default=list(DEFAULT_TCP_XYZ_RIGHT_LINK26),
        metavar=("X", "Y", "Z"),
        help="right_link26 -> right_tcp translation, in meters.",
    )
    parser.add_argument(
        "--tcp-quat-wxyz",
        type=float,
        nargs=4,
        default=list(DEFAULT_TCP_QUAT_WXYZ_RIGHT_LINK26),
        metavar=("W", "X", "Y", "Z"),
        help="right_link26 -> right_tcp rotation quaternion.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fixed_lift_highest:
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_FIXED_LIFT_OUTPUT
        if args.manifest == DEFAULT_MANIFEST:
            args.manifest = DEFAULT_FIXED_LIFT_MANIFEST
    config = build_config(
        tcp_xyz=[float(v) for v in args.tcp_xyz],
        tcp_quat_wxyz=[float(v) for v in args.tcp_quat_wxyz],
        fixed_lift=bool(args.fixed_lift_highest),
        lift_joint_value=float(args.lift_joint_value),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    manifest = build_manifest(config_path=output, config=config)
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"config": str(output), "manifest": str(manifest_path)}, indent=2))
    return 0


def build_config(
    *,
    tcp_xyz: list[float],
    tcp_quat_wxyz: list[float],
    fixed_lift: bool = False,
    lift_joint_value: float = 0.46,
) -> dict[str, object]:
    visual_urdf = ensure_visual_urdf().resolve()
    collision_spheres = load_collision_spheres()
    right_collision_spheres = {
        link_name: collision_spheres[link_name]
        for link_name in RIGHT_COLLISION_LINK_NAMES
        if link_name in collision_spheres
    }
    rest_by_joint = {
        name: float(value)
        for name, value in zip(LIFT2_JOINT_NAMES, LIFT2_CUROBO_SAFE_REST_QPOS)
    }
    planned_joint_names = (
        RIGHT_ARM_FIXED_LIFT_JOINT_NAMES if fixed_lift else RIGHT_ARM_JOINT_NAMES
    )
    default_joint_position = [rest_by_joint[name] for name in planned_joint_names]
    # cuRobo only needs locks for non-planned joints that are still on branches
    # used by collision links.  Wheels and the left arm are outside the
    # right-arm kinematic subtree used here, so adding them to lock_joints makes
    # the loader look for joints that are not present in the reduced graph.
    lock_joints = {name: rest_by_joint[name] for name in RIGHT_GRIPPER_JOINT_NAMES}
    if fixed_lift:
        lock_joints["joint4"] = float(lift_joint_value)

    return {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "urdf_path": str(visual_urdf),
                "asset_root_path": str((PACKAGE_ROOT / "urdf" / "lift2").resolve()),
                "base_link": "base_link",
                "tool_frames": ["right_tcp"],
                "grasp_contact_link_names": [
                    "right_link26",
                    "right_link27",
                    "right_link28",
                    "attached_object",
                ],
                "collision_link_names": RIGHT_COLLISION_LINK_NAMES,
                "collision_sphere_buffer": 0.0,
                "collision_spheres": right_collision_spheres,
                "extra_collision_spheres": {"attached_object": 4},
                "extra_links": {
                    "right_tcp": {
                        "parent_link_name": "right_link26",
                        "link_name": "right_tcp",
                        "joint_name": "right_tcp_fixed_joint",
                        "joint_type": "FIXED",
                        "fixed_transform": [*tcp_xyz, *tcp_quat_wxyz],
                    },
                    "attached_object": {
                        "parent_link_name": "right_tcp",
                        "link_name": "attached_object",
                        "joint_name": "right_attach_joint",
                        "joint_type": "FIXED",
                        "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    },
                },
                "lock_joints": lock_joints,
                "mesh_link_names": RIGHT_COLLISION_LINK_NAMES,
                "self_collision_buffer": {
                    link_name: 0.0 for link_name in [*RIGHT_COLLISION_LINK_NAMES, "attached_object"]
                },
                "self_collision_ignore": build_self_collision_ignore(),
                "cspace": {
                    "joint_names": planned_joint_names,
                    "default_joint_position": default_joint_position,
                    "null_space_weight": [1.0] * len(planned_joint_names),
                    "cspace_distance_weight": [1.0] * len(planned_joint_names),
                    "position_limit_clip": 0.01,
                    "max_acceleration": 8.0,
                    "max_jerk": 500.0,
                },
                "use_global_cumul": True,
            }
        }
    }


def build_self_collision_ignore() -> dict[str, list[str]]:
    return {
        "right_link21": ["right_link22"],
        "right_link22": ["right_link23"],
        # The coarse spheres for right_link23 and right_link25 overlap in the
        # default folded posture; the meshes are separated by right_link24, so
        # this is a sphere-model false positive rather than a useful collision.
        "right_link23": ["right_link24", "right_link25"],
        "right_link24": ["right_link25", "right_link26"],
        "right_link25": ["right_link26", "right_link27", "right_link28"],
        "right_link26": ["right_link27", "right_link28", "right_tcp", "attached_object"],
        "right_link27": ["right_link28", "right_tcp", "attached_object"],
        "right_link28": ["right_tcp", "attached_object"],
        "right_tcp": ["attached_object"],
    }


def build_manifest(*, config_path: Path, config: dict[str, object]) -> dict[str, object]:
    kinematics = config["robot_cfg"]["kinematics"]  # type: ignore[index]
    tcp_transform = kinematics["extra_links"]["right_tcp"]["fixed_transform"]  # type: ignore[index]
    planned_joint_names = list(kinematics["cspace"]["joint_names"])  # type: ignore[index]
    return {
        "config": str(config_path),
        "urdf_path": kinematics["urdf_path"],  # type: ignore[index]
        "base_link": kinematics["base_link"],  # type: ignore[index]
        "tool_frames": kinematics["tool_frames"],  # type: ignore[index]
        "planned_joint_names": planned_joint_names,
        "locked_joint_names": sorted(kinematics["lock_joints"].keys()),  # type: ignore[index, union-attr]
        "fixed_lift": "joint4" in kinematics["lock_joints"],  # type: ignore[operator]
        "fixed_lift_joint_value": kinematics["lock_joints"].get("joint4"),  # type: ignore[union-attr]
        "right_gripper_joint_names": RIGHT_GRIPPER_JOINT_NAMES,
        "collision_link_names": RIGHT_COLLISION_LINK_NAMES,
        "tcp": {
            "parent_link": "right_link26",
            "link": "right_tcp",
            "xyz": list(tcp_transform[:3]),
            "quat_wxyz": list(tcp_transform[3:7]),
            "note": "Midpoint of both finger meshes at the section 1 cm behind the tips.",
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
