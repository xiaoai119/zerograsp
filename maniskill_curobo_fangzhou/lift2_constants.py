"""Lift2 joint-order constants shared by ManiSkill and cuRobo bridge code."""

from __future__ import annotations

import numpy as np


LIFT2_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "left_joint11",
    "right_joint21",
    "left_joint12",
    "right_joint22",
    "left_joint13",
    "right_joint23",
    "left_joint14",
    "right_joint24",
    "left_joint15",
    "right_joint25",
    "left_joint16",
    "right_joint26",
    "left_joint17",
    "left_joint18",
    "right_joint27",
    "right_joint28",
)

# Wheels stay at zero, the lift is raised, both arms are mildly folded, and
# both grippers are open. Values follow the limits in lift2.urdf.
LIFT2_REST_QPOS = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.46,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.03,
        0.03,
        0.03,
        0.03,
    ],
    dtype=np.float32,
)

# cuRobo applies a 0.01 m position-limit clip to joint4 by default.  The raw
# rest pose puts joint4 exactly at the URDF upper limit (0.46), which makes the
# planner reject the start state.  Keep a cuRobo-safe variant for planning
# smoke tests and generated robot configs.
LIFT2_CUROBO_SAFE_REST_QPOS = LIFT2_REST_QPOS.copy()
LIFT2_CUROBO_SAFE_REST_QPOS[3] = 0.45

# Fixed tool frame used by cuRobo and by the ManiSkill execution URDFs.  The
# point is the midpoint between the two finger meshes at a cross-section 1 cm
# behind their tips.  The finger tips are at x=0.15497 m in the right_link26
# frame, so the grasp center is x=0.14497 m.  Keeping the same frame in both
# systems lets us measure actual execution error against the planned TCP target.
LIFT2_RIGHT_TCP_PARENT_LINK = "right_link26"
LIFT2_RIGHT_TCP_LINK = "right_tcp"
LIFT2_RIGHT_TCP_JOINT = "right_tcp_fixed_joint"
LIFT2_RIGHT_TCP_XYZ_RIGHT_LINK26 = np.array(
    [0.14497, 0.001786, 0.0],
    dtype=np.float32,
)
LIFT2_RIGHT_TCP_RPY_RIGHT_LINK26 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
LIFT2_RIGHT_TCP_QUAT_WXYZ_RIGHT_LINK26 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Virtual head RGB-D camera frame.  The source CAD has an RS01-like camera
# housing baked into base_link.STL rather than exported as a separate URDF
# link.  Add a fixed frame at the tuned local camera pose so ManiSkill sensors
# can be mounted to the robot instead of being standalone world cameras.
LIFT2_HEAD_CAMERA_PARENT_LINK = "base_link"
LIFT2_HEAD_CAMERA_LINK = "head_camera_link"
LIFT2_HEAD_CAMERA_JOINT = "head_camera_fixed_joint"
LIFT2_HEAD_CAMERA_EYE_BASE = np.array([0.300000, 0.0, 1.041600], dtype=np.float32)
LIFT2_HEAD_CAMERA_TARGET_BASE = np.array([0.656926, 0.0, 0.701600], dtype=np.float32)
