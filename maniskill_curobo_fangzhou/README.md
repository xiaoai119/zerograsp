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

The default robot UID is `lift2_collision_spheres_debug`: it uses the same
physical collision spheres as the normal collision-sphere robot, and only draws
the links whose YAML style has `visible: true`. By default every collision
sphere visual is hidden. Each run writes both a complete scene image and a
closer robot calibration image. For physical collision geometry without any
debug visual-sphere support, use:

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

## Adjusting collision spheres

The editable collision-sphere source is:

```text
maniskill_curobo_fangzhou/generated/lift2_collision_spheres.yml
```

Each entry is in the local coordinate frame of that URDF link, in metres:

```yaml
collision_spheres:
  left_link12:
  - center: [-0.0086847, 0.0002655, 0.0006897]
    radius: 0.0456612
```

For straight links, prefer the line-chain form. It guarantees that all generated
spheres lie on one straight line:

```yaml
collision_spheres:
  left_link12:
    line:
      start: [-0.01, 0.0, 0.0]
      end: [-0.25, 0.0, 0.0]
      count: 4
      radius: 0.046
```

`radius` can also be `[start_radius, end_radius]` to taper the chain, or a list
with one radius per sphere:

```yaml
collision_spheres:
  left_link12:
    line:
      start: [-0.01, 0.0, 0.0]
      end: [-0.25, 0.0, 0.0]
      count: 4
      radius: [0.048, 0.046, 0.044, 0.042]
```

For mildly bent links, use multiple straight segments:

```yaml
collision_spheres:
  left_link13:
    segments:
    - start: [0.02, 0.0, 0.02]
      end: [0.12, 0.0, 0.05]
      count: 3
      radius: 0.052
    - start: [0.12, 0.0, 0.05]
      end: [0.22, 0.0, 0.06]
      count: 3
      radius: 0.058
```

Use these rules when tuning:

- If a sphere is too large, reduce `radius`.
- If a sphere misses part of the link, increase `radius` or add another sphere.
- If a whole straight chain is offset, edit `start` and `end` instead of every
  generated sphere center.
- If one explicit sphere is in the wrong place, edit `center: [x, y, z]`.
- Coordinates are link-local, not world coordinates. For an unfamiliar link,
  change one coordinate by a visible amount such as `0.02`, render the debug
  view, and then you will know which local axis it moved along.
- Left arm links are `left_link11` through `left_link18`; right arm links are
  `right_link21` through `right_link28`.

After editing the YAML, regenerate or reload the generated URDFs. The Python
loader does this automatically when the YAML timestamp is newer than the URDF,
but the most explicit way is:

```bash
rm -f maniskill_curobo_fangzhou/generated/lift2_maniskill_collision_spheres*.urdf
PYTHONPATH=. python -m maniskill_curobo_fangzhou.render_lift2_seed \
  --seed 1 --robot-uid lift2_collision_spheres_debug
```

For a moving preview with the enabled debug spheres:

```bash
PYTHONPATH=. python -m maniskill_curobo_fangzhou.record_lift2_motion_video \
  --seed 1 --robot-uid lift2_collision_spheres_debug
```

That recording command is the normal "edit-and-preview" loop: edit
`lift2_collision_spheres.yml`, run the command above, and it will regenerate
the debug URDF if needed and write a fresh video.

The YAML also includes ten preset colors:

```yaml
color_palette:
  blue: [0.10, 0.45, 1.00, 0.38]
  red: [1.00, 0.25, 0.10, 0.38]
  green: [0.10, 0.85, 0.25, 0.38]
  yellow: [1.00, 0.85, 0.05, 0.38]
  purple: [0.65, 0.25, 1.00, 0.38]
  cyan: [0.00, 0.85, 1.00, 0.38]
  orange: [1.00, 0.55, 0.05, 0.38]
  pink: [1.00, 0.30, 0.75, 0.38]
  white: [1.00, 1.00, 1.00, 0.32]
  black: [0.05, 0.05, 0.05, 0.45]
```

Per-link color, human-readable labels, and visibility switches live in
`collision_sphere_styles`. For example:

```yaml
collision_sphere_styles:
  left_link12:
    label: left upper arm long link
    color: cyan
    visible: true
```

Set `color` to any palette name, or directly to an RGBA list like
`[0.2, 1.0, 0.6, 0.4]`. Set `visible: true` only for the links you are
currently tuning; leave it as `false` to hide that link's visual debug spheres.
This visibility flag only affects the rendered debug overlays. The physical
collision spheres are still written to the URDF and still participate in
planning/collision checks.

Do not rerun `fit_lift2_collision_spheres.py` after manual edits unless you
intend to overwrite the YAML. If a correction should survive refits, add it to
`MANUAL_SPHERE_OVERRIDES` in `fit_lift2_collision_spheres.py`.
