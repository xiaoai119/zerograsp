"""Constants for the Unitree H2 URDF models."""

from __future__ import annotations

from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
H2_DESCRIPTION_ROOT = PACKAGE_ROOT / "unitree_ros" / "robots" / "h2_description"
H2_STL_URDF = H2_DESCRIPTION_ROOT / "H2.urdf"
H2_DAE_URDF = H2_DESCRIPTION_ROOT / "H2_dae.urdf"

H2_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "head_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# A conservative display pose: legs are close to the URDF neutral pose while the
# arms are slightly bent and moved away from the torso for easier inspection.
# The order matches SAPIEN's active-joint order after loading H2.urdf.
H2_REST_QPOS = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.18,
        0.18,
        0.0,
        0.25,
        0.25,
        0.0,
        0.0,
        0.0,
        0.35,
        -0.35,
        -0.12,
        -0.12,
        0.0,
        0.0,
        0.75,
        0.75,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
