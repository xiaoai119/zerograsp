# Lift2 Right-Arm cuRobo Bringup

This note records the current minimal route for making cuRobo understand and
control the Lift2 right arm.

## Current Scope

The first cuRobo config is intentionally right-arm only:

- Planned joints:
  - `joint4`
  - `right_joint21`
  - `right_joint22`
  - `right_joint23`
  - `right_joint24`
  - `right_joint25`
  - `right_joint26`
- Locked gripper joints:
  - `right_joint27`
  - `right_joint28`
- Ignored for now:
  - wheels
  - left arm
  - dual-arm coordination

This gives cuRobo a 7-DoF planning problem: lift plus six right-arm joints.

## Generated Files

Generate the config:

```bash
PYTHONPATH=. python -m maniskill_curobo_fangzhou.generate_lift2_curobo_config
```

Outputs:

- `maniskill_curobo_fangzhou/config/lift2_right_arm_curobo.yml`
- `maniskill_curobo_fangzhou/config/lift2_right_arm_curobo_manifest.json`

The config uses:

- URDF: `maniskill_curobo_fangzhou/generated/lift2_maniskill_visual.urdf`
- collision spheres: `maniskill_curobo_fangzhou/generated/lift2_collision_spheres.yml`
- tool frame: `right_tcp`

## TCP Assumption

The current TCP is a geometric first guess:

```text
parent link: right_link26
tcp link:    right_tcp
translation: [0.155, 0.0, -0.020] m
rotation:    [1.0, 0.0, 0.0, 0.0] wxyz
```

Reasoning:

- `right_link26` is the gripper palm / wrist end.
- `right_link27` and `right_link28` are the two fingers.
- The finger meshes extend to about `x = 0.155 m` in the `right_link26` frame.
- The TCP is placed on the two-finger centerline near the finger tips.

This is not final calibration. Before grasp execution, we should visualize the
TCP marker and verify:

- the marker is between the two fingers,
- the approach axis matches the actual closing direction,
- the width axis matches the two-finger opening direction.

## Smoke Test

Run:

```bash
PYTHONPATH=. maniskill_curobo/envs/maniskill_curobo/bin/python \
  -m maniskill_curobo_fangzhou.smoke_lift2_curobo_config \
  --config maniskill_curobo_fangzhou/config/lift2_right_arm_curobo.yml
```

Current smoke result:

```text
status: ok
dof: 7
joint_names:
  joint4
  right_joint21
  right_joint22
  right_joint23
  right_joint24
  right_joint25
  right_joint26
tool_frames:
  right_tcp
robot_spheres_shape:
  [1, 1, 63, 4]
```

Report path:

```text
maniskill_curobo_fangzhou/runs/lift2_curobo_config_smoke/lift2_right_arm_curobo_smoke.json
```

## Controller Mapping

cuRobo outputs the 7 planned joints in this order:

```text
joint4,
right_joint21,
right_joint22,
right_joint23,
right_joint24,
right_joint25,
right_joint26
```

ManiSkill expects the full Lift2 action in `LIFT2_JOINT_NAMES` order. The helper
in `maniskill_curobo_fangzhou/lift2_curobo_bridge.py` maps cuRobo right-arm qpos
back into the full action vector:

```python
make_lift2_action_from_right_arm_qpos(right_arm_qpos, gripper_qpos=0.03)
```

This keeps non-right-arm joints at `LIFT2_REST_QPOS` unless a base action is
provided.

## cuRobo-Safe Rest Pose

The raw Lift2 rest pose sets `joint4 = 0.46`, exactly at the URDF upper limit.
cuRobo applies a `position_limit_clip = 0.01`, so its effective `joint4` range is
`[0.01, 0.45]`. Starting from `0.46` makes trajectory planning reject the start
state as invalid.

Use `LIFT2_CUROBO_SAFE_REST_QPOS` for cuRobo smoke tests and generated robot
configs:

```text
joint4: 0.45
```

## Reaching Smoke Status

Added:

```text
maniskill_curobo_fangzhou/smoke_lift2_right_arm_reach.py
```

This script tries to plan a tiny `right_tcp` motion and execute it through the
right-arm action mapping.

Current status:

```text
M0/M1 robot config load: passed
right_tcp FK: passed
MotionPlanner object creation: passed
right-arm action mapping: passed
same-pose IK: passed
right-arm reaching plan: passed
right-arm reaching execution video: passed
```

Report and video:

```text
maniskill_curobo_fangzhou/runs/lift2_right_arm_reach_smoke_seed001/report.json
maniskill_curobo_fangzhou/runs/lift2_right_arm_reach_smoke_seed001/right_arm_reach_seed001_960x540.mp4
```

The original failure had two concrete causes:

1. `right_link23` and `right_link25` had four overlapping collision-sphere
   pairs in the default folded posture. This was a false self-collision from
   the coarse sphere model, with max penetration around 13 mm. Adding
   `right_link23 -> right_link25` to `self_collision_ignore` fixed same-pose IK.
2. `joint4 = 0.46` was outside cuRobo's clipped planning limits. Using
   `joint4 = 0.45` for cuRobo-safe rest fixed trajectory planning.

This only proves the right-arm planning/control bridge can execute a tiny
motion. It does not yet prove grasp execution, M4C world collision integration,
or TCP calibration quality.

## M4C World Collision Smoke Status

Added:

```text
maniskill_curobo_fangzhou/smoke_lift2_m4c_right_arm_reach.py
```

This script loads a prebuilt M4C voxel ESDF scene, restores it into a cuRobo
`Scene(cuboid + VoxelGrid)`, creates the Lift2 right-arm planner with voxel
collision cache, and executes the same small `right_tcp` reach.

Current seed1 smoke result:

```text
status: ok
scene_has_cuboid: true
scene_has_voxel: true
voxel dims: [0.9, 1.5, 0.34]
voxel_size: 0.01
trajectory_steps: 21
```

Report and video:

```text
maniskill_curobo_fangzhou/runs/lift2_m4c_right_arm_reach_seed001/report.json
maniskill_curobo_fangzhou/runs/lift2_m4c_right_arm_reach_seed001/m4c_right_arm_reach_seed001_960x540.mp4
```

This proves the minimal right-arm planner can be instantiated with an M4C
world collision model. It still does not prove grasp execution; the next step
is converting ZeroGrasp / GraspNet poses into `right_tcp` targets and checking
approach/grasp/lift planning under the same world model.

## Next Steps

1. Visualize `right_tcp` in the ManiSkill scene and tune the TCP transform.
2. Compare cuRobo FK and ManiSkill FK for the same right-arm qpos.
3. Tune `right_tcp` against the real gripper center / closing axis.
4. Convert ZeroGrasp / GraspNet grasp poses into Lift2 `right_tcp` targets.
5. Run approach / grasp / lift planning under the M4C world collision model.
6. Only after right-arm grasp execution works, scale from seed1 smoke to seed
   batches.
