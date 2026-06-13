"""Minimal ManiSkill agent registration for the Lift2 dual-arm robot."""

from __future__ import annotations

import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.registration import register_agent

from .urdf_adapter import ensure_collision_sphere_urdf, ensure_visual_urdf


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
        1.20,
        1.20,
        1.80,
        1.80,
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


@register_agent()
class Lift2CollisionSpheres(Lift2Visual):
    """Lift2 with physical sphere collisions on both arms and grippers."""

    uid = "lift2_collision_spheres"
    urdf_path = str(ensure_collision_sphere_urdf())


@register_agent()
class Lift2CollisionSpheresDebug(Lift2CollisionSpheres):
    """Collision-sphere agent with colored sphere visuals for calibration."""

    uid = "lift2_collision_spheres_debug"
    urdf_path = str(ensure_collision_sphere_urdf(show_spheres=True))
