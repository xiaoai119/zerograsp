# Lift2 ManiSkill integration

This directory contains the initial compatibility test for replacing the
Franka robot with `urdf/lift2/urdf/lift2.urdf`.

## Current status

- The exported ROS `package://lift2/...` mesh paths are converted to absolute
  paths in a generated URDF.
- The robot is registered in ManiSkill as `lift2_visual`.
- The mobile base is fixed on the floor on the opposite side of the table from
  Franka and rotated 180 degrees to face the workspace. Its root is placed at
  `x=0.7069260`, computed from the rotated base bounds and the positive-X table
  edge, leaving a 2 cm gap so the mobile base does not intersect the table.
  Its height is derived from the wheel joint and mesh bounds.
- A 20-DoF rest pose is supplied for the lift, both arms, and both grippers.
- Collision meshes are intentionally removed from this first generated URDF.
  The original STL files are detailed render meshes and are not yet suitable
  as efficient articulation collision geometry.
- Both arms and both grippers also have a generated collision-sphere model.
  The same `center` and `radius` values are written into ManiSkill URDF
  collision elements and a cuRobo-style YAML file.
- The current fit contains 50 physical sphere shapes across 16 moving arm and
  gripper links. A 3 mm conservative radius padding is applied.
- `PickClutterYCBLift2-v1` places the robot in a deterministic clutter seed and
  renders the complete scene.

Run:

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python \
  -m maniskill_curobo_fangzhou.render_lift2_seed --seed 1
```

The default robot UID is `lift2_collision_spheres_debug`: left-arm spheres are
blue and right-arm spheres are red. Each run writes both a complete scene image
and a closer robot calibration image. For physical collision geometry without
the colored overlays, use:

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python \
  -m maniskill_curobo_fangzhou.render_lift2_seed \
  --seed 1 --robot-uid lift2_collision_spheres
```

The sphere model currently covers the two arms and grippers, but not the
mobile base or lift body. Self-collision remains disabled during this first
calibration pass. Replacing Franka for actual grasp execution still requires
a right- or left-arm controller definition, TCP links, gripper mimic control,
and a complete matching cuRobo robot configuration.
