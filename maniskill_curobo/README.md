# ManiSkill + cuRobo Isolated Experiment

This directory is the only workspace for the cuRobo integration experiment.

## Route Lock

- Primary route: **B** - create a new `maniskill_curobo` Python 3.10 conda environment that contains both ManiSkill and cuRobo.
- Fallback route: **C** - keep cuRobo in a separate process/environment that writes joint trajectories for ManiSkill to execute.
- Forbidden route: **A** - do not install cuRobo into the existing `maniskill` environment.

The existing `graduate` environment remains responsible for ZeroGrasp inference.

## Local Host Notes

Current local host discovered on 2026-05-15:

```text
GPU: NVIDIA GeForce RTX 5070 Ti
VRAM: 16GB
Driver: 595.58.03
PyTorch CUDA in graduate: 2.11.0+cu128
GPU compute capability: 12.0
System nvcc: not found
```

Because this machine is not an RTX 3090, CUDA extension builds here should use:

```bash
export TORCH_CUDA_ARCH_LIST=12.0
```

The future RTX 3090 host should use:

```bash
export TORCH_CUDA_ARCH_LIST=8.6
```

## Files

- `docs/2026-05-15-curobo-planner-design.md` - design and acceptance gates.
- `scripts/check_host.py` - host and Python environment diagnostics.
- `scripts/create_env_b.sh` - builds the isolated Python 3.10 `maniskill_curobo` conda environment under this directory.
- `scripts/smoke_imports.py` - verifies PyTorch, ManiSkill, SAPIEN, and cuRobo imports.
- `scripts/smoke_motion_planning.sh` - runs cuRobo pose and grasp planning examples.
- `scripts/smoke_maniskill_after_curobo.py` - verifies ManiSkill can create a Panda environment after cuRobo is imported.
- `scripts/smoke_maniskill_qpos_to_curobo.py` - reads ManiSkill Panda qpos and plans from that state with cuRobo.
- `scripts/run_full_pipeline.py` - full ManiSkill -> ZeroGrasp -> projection -> cuRobo execution pipeline.
- `scripts/execute_curobo_pick.py` - prototype executor that converts a ZeroGrasp grasp target to cuRobo pose goals, plans Panda joint trajectories, and executes them with ManiSkill `pd_joint_pos`.
- `scene_export.py` - converts ManiSkill table/object collision shapes into a per-run cuRobo scene model.
- `joint_trajectory_utils.py` - tested helpers for mapping cuRobo joint trajectories into ManiSkill joint-position actions.
- `tests/test_joint_trajectory_utils.py` - unit tests for joint-name ordering, trajectory shape handling, and action construction.

## First Commands

From the repository root:

```bash
python maniskill_curobo/scripts/check_host.py
bash maniskill_curobo/scripts/create_env_b.sh
```

Then activate the isolated environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/maniskill_curobo/envs/maniskill_curobo"
python maniskill_curobo/scripts/smoke_imports.py
bash maniskill_curobo/scripts/smoke_motion_planning.sh
python maniskill_curobo/scripts/smoke_maniskill_after_curobo.py
python maniskill_curobo/scripts/smoke_maniskill_qpos_to_curobo.py
```

## Full Pipeline

Run from the repository root. This uses the isolated ManiSkill + cuRobo Python
environment for ManiSkill/curobo steps and the existing `graduate` conda
environment for ZeroGrasp inference:

```bash
PYTHONPATH=. python maniskill_curobo/scripts/run_full_pipeline.py \
  --env-id PickSingleYCB-v1 \
  --seed 1 \
  --run-name picksingle_seed1_curobo_full
```

The run folder contains:

```text
zg_input/
zg_output/
grasp_projection.png
curobo_scene.yml
curobo_scene_metadata.json
curobo_plan_pre.npz
curobo_plan_grasp.npz
curobo_plan_lift.npz
execution.mp4
run_manifest.json
pipeline_manifest.json
logs/
```

`run_manifest.json` is written by the cuRobo executor. `pipeline_manifest.json`
records the full four-step pipeline and per-step logs.

## cuRobo Pick Executor Prototype

Run from the repository root after activating the isolated environment:

```bash
PYTHONPATH=. python maniskill_curobo/scripts/execute_curobo_pick.py \
  --zerograsp-output /path/to/zg_output_dir \
  --env-id PickSingleYCB-v1 \
  --seed 1 \
  --approach-axis positive-x \
  --camera-eye -0.30 0.0 0.55 \
  --camera-target 0.05 0.0 0.08 \
  --output-dir maniskill_curobo/runs/picksingle_seed1_curobo
```

The output directory contains:

```text
zg_input/rgb.png
zg_input/depth.png
zg_input/mask.png
zg_input/rgbd.npz
zg_input/camera.json
zg_input/scene.json
zg_output/recommended_grasp_top1.json
zg_output/raw_outputs/
grasp_projection.png
curobo_scene.yml
curobo_scene_metadata.json
curobo_plan_pre.npz
curobo_plan_grasp.npz
curobo_plan_lift.npz
execution.mp4
run_manifest.json
```

`zg_input` is exported from the exact ManiSkill reset used by the executor.
When `--zerograsp-output` is provided, that directory is copied into
`zg_output`, and the executor reads the copied output so the marker, trajectory,
projection image, and video can be audited against the saved ZeroGrasp
artifacts. `grasp_projection.png` uses the same drawing function as
`maniskill_codex.grasp_projection`. The default camera view matches
`maniskill_codex/camera_views.py`.

By default, the executor exports the current ManiSkill scene to
`curobo_scene.yml` and gives that file to cuRobo instead of using the fixed
`collision_test.yml`. The exporter reads the reset scene's table and object
collision shapes, converts boxes and convex meshes into conservative cuboid
obstacles in the Franka base frame, and writes audit metadata to
`curobo_scene_metadata.json`. The selected ZeroGrasp target object is excluded
from the obstacle set by default so the final grasp pose is not automatically
inside a forbidden obstacle. Use these switches when debugging:

```bash
--scene-source fixed              # use --scene-model, e.g. collision_test.yml
--scene-include-target-object     # keep the target object as an obstacle
--scene-min-cuboid-dimension 0.005
```

By default, the saved ZeroGrasp input images (`zg_input/rgb.png`,
`zg_input/depth.png`, `zg_input/mask.png`, and `zg_input/rgbd.npz`) use the same
sensor resolution as `maniskill_codex`: `1280x1024`. `execution.mp4` also uses
the same video settings: `1280x1024` render frames at `20` FPS. Override
`--width`, `--height`, `--render-width`, `--render-height`, and `--video-fps`
only for quick low-resolution smoke tests.

The video also draws the selected grasp pose with the same marker semantics as
`maniskill_codex`: green sphere for the grasp center, red arrow/bar for the
approach direction, and blue bar for the gripper width direction. Pass
`--no-grasp-marker` to disable the marker.

For a planner/executor smoke test without a ZeroGrasp output file:

```bash
PYTHONPATH=. python maniskill_curobo/scripts/execute_curobo_pick.py \
  --target-base 0.50 0.00 0.20 \
  --target-quat-wxyz 1 0 0 0 \
  --target-approach 0 0 -1 \
  --lift-offset 0.10 \
  --output-dir maniskill_curobo/runs/debug_target_seed1_curobo
```

Current limitation: object geometry is exported as cuboid approximations of
ManiSkill collision shapes. This is enough for a first real scene-aware planner
pass, but precise gripper/contact reasoning still needs a stricter Franka TCP
mapping and stage-specific target-object handling.

## Current Local Status

As of 2026-05-15, Route B is viable on this local machine:

```text
torch cuda: ok
mani_skill + sapien import: ok
curobo import: ok
cuRobo pose planning: ok
cuRobo grasp planning: ok
ManiSkill PickSingleYCB-v1 after cuRobo import: ok
ManiSkill qpos -> cuRobo plan bridge: ok
```

Smoke outputs:

```text
maniskill_curobo/smoke_tests/motion_planning/motion_plan.pdf
maniskill_curobo/smoke_tests/motion_planning/grasp_plan.pdf
maniskill_curobo/smoke_tests/bridge/maniskill_qpos_to_curobo_plan.npz
```
