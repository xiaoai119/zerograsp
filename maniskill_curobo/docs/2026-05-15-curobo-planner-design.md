# cuRobo Planner Design

## Goal

Replace the current execution-stage IK chasing with cuRobo-based collision-aware joint trajectory generation, without modifying or polluting the existing `graduate` and `maniskill` environments.

## Constraints

- All new files, logs, cloned repos, and local conda environments for this experiment stay under `./maniskill_curobo/`.
- Do not install cuRobo into the existing `maniskill` environment.
- Keep ZeroGrasp inference in `graduate`.
- First implementation target is local smoke testing, not immediate replacement of the full pipeline.

## Route B: Preferred

Create an isolated Python 3.10 conda environment at:

```text
./maniskill_curobo/envs/maniskill_curobo
```

This environment should contain:

- PyTorch with CUDA support.
- ManiSkill 3.0.1.
- SAPIEN 3.0.3.
- cuRobo from source.
- CUDA compiler/toolkit pieces needed by cuRobo.

In this route, the future planner will read the current Panda `qpos` and target TCP pose directly inside ManiSkill, call cuRobo MotionGen, and execute the resulting joint trajectory in the simulator.

## Route C: Fallback

Use only if Route B fails due to incompatible Python, PyTorch, CUDA, or SAPIEN/cuRobo dependencies.

In Route C:

- cuRobo runs in a separate environment or process.
- The planner input is serialized as JSON/NPZ:
  - current joint positions,
  - target TCP pose,
  - robot/world collision config.
- The planner output is a joint trajectory file.
- ManiSkill loads and executes that trajectory.

## First Collision World

Start minimal:

- Panda robot model from cuRobo config.
- Table represented as a cuboid.
- No object mesh collision in the first smoke test.

Only after the arm can plan from current pose to pre-grasp without table collision should we add target and clutter objects.

## First Acceptance Gates

Route B is considered viable when all of these pass:

1. `torch.cuda.is_available()` returns `True` in the isolated environment.
2. `import mani_skill`, `import sapien`, and `import curobo` all succeed in the isolated environment.
3. A minimal cuRobo Franka/Panda MotionGen example can plan one collision-free trajectory.
4. A small ManiSkill environment can be created in the same Python process after cuRobo is imported.

## Local Smoke Test Result

The local Route B smoke test passed on 2026-05-15:

- Isolated environment path: `./maniskill_curobo/envs/maniskill_curobo`
- Python: 3.10.20
- PyTorch: 2.11.0+cu128
- CUDA visible to PyTorch: yes
- GPU: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0
- nvcc: 12.8.93 inside the isolated environment
- ManiSkill: 3.0.1
- SAPIEN: 3.0.3
- cuRobo: import succeeds
- cuRobo pose planning: succeeded with 61 waypoints
- cuRobo grasp planning: succeeded
- ManiSkill `PickSingleYCB-v1` with `pd_joint_pos`: created successfully after importing cuRobo
- Bridge smoke test: read ManiSkill Panda arm qpos, reordered it by cuRobo joint names, and planned from that state successfully
- Bridge trajectory joint names: cuRobo returns 9 active joints after interpolation, matching ManiSkill's 7 Panda arm joints plus 2 finger joints

One install issue was found and fixed: `pip install -e . --no-build-isolation` installs cuRobo itself, but the runtime CUDA Core backend also needs `cuda-core[cu12]>=0.7`.

## Non-Goals For The First Pass

- Do not rewrite `maniskill_codex/execute_zerograsp_pick.py` yet.
- Do not change ZeroGrasp inference.
- Do not tune grasp orientation conventions.
- Do not solve dense RGBD/nvblox collision worlds yet.
