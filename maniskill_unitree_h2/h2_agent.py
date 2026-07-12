"""Minimal ManiSkill agent registration for Unitree H2."""

from __future__ import annotations

import math

import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

from .h2_constants import H2_DAE_URDF, H2_REST_QPOS, H2_STL_URDF


@register_agent()
class UnitreeH2STL(BaseAgent):
    """Unitree H2 loaded from the STL-visual URDF."""

    uid = "unitree_h2_stl"
    urdf_path = str(H2_STL_URDF)
    fix_root_link = True
    disable_self_collisions = True

    keyframes = {
        "rest": Keyframe(qpos=H2_REST_QPOS, pose=sapien.Pose()),
    }

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="head_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=math.pi / 2,
                near=0.01,
                far=100,
                entity_uid="head_yaw_link",
            )
        ]


@register_agent()
class UnitreeH2DAEVisual(UnitreeH2STL):
    """Unitree H2 loaded from the DAE-visual URDF."""

    uid = "unitree_h2_dae_visual"
    urdf_path = str(H2_DAE_URDF)

