"""Minimal ManiSkill agent registration for the Lift2 dual-arm robot."""

from __future__ import annotations

import math

import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

from .lift2_constants import LIFT2_HEAD_CAMERA_LINK, LIFT2_JOINT_NAMES, LIFT2_REST_QPOS
from .urdf_adapter import (
    ensure_collision_sphere_urdf,
    ensure_full_collision_urdf,
    ensure_visual_urdf,
)


@register_agent()
class Lift2Visual(BaseAgent):
    """Visual-only Lift2 agent used to validate URDF loading and placement."""

    uid = "lift2_visual"
    urdf_path = str(ensure_visual_urdf())
    fix_root_link = True
    disable_self_collisions = True

    keyframes = {
        "rest": Keyframe(qpos=LIFT2_REST_QPOS, pose=sapien.Pose()),
    }

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="base_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=math.pi / 2,
                near=0.01,
                far=100,
                entity_uid=LIFT2_HEAD_CAMERA_LINK,
            )
        ]


@register_agent()
class Lift2CollisionSpheres(Lift2Visual):
    """Lift2 with physical sphere collisions on both arms and grippers."""

    uid = "lift2_collision_spheres"
    urdf_path = str(ensure_collision_sphere_urdf())


@register_agent()
class Lift2FullCollision(Lift2Visual):
    """Lift2 with the original URDF mesh collisions for physics contacts."""

    uid = "lift2_full_collision"
    urdf_path = str(ensure_full_collision_urdf())


@register_agent()
class Lift2CollisionSpheresDebug(Lift2CollisionSpheres):
    """Collision-sphere agent with colored sphere visuals for calibration."""

    uid = "lift2_collision_spheres_debug"
    urdf_path = str(ensure_collision_sphere_urdf(show_spheres=True))
