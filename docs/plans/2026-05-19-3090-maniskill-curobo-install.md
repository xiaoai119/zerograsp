# 3090 ManiSkill cuRobo Installation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install and verify the `maniskill_curobo` pipeline on an RTX 3090 host while keeping ZeroGrasp inference in the existing `graduate` conda environment.

**Architecture:** Use the current Route B layout: `graduate` runs ZeroGrasp inference, and `maniskill_curobo/envs/maniskill_curobo` is an isolated Python 3.10 conda environment that contains ManiSkill, SAPIEN, PyTorch CUDA wheels, and cuRobo. The full pipeline script switches between the two environments and writes one self-contained output folder per run.

**Tech Stack:** Ubuntu, NVIDIA driver, conda, Python 3.10 for `maniskill_curobo`, Python 3.11 for `graduate`, PyTorch CUDA 12.8 wheels, ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.3.0, cuRobo, CUDA nvcc from conda when system `nvcc` is missing.

---

## Known Working Versions

The local `maniskill_curobo` environment currently reports:

```text
python 3.10.20
torch 2.11.0+cu128
mani_skill 3.0.1
sapien 3.0.3
gymnasium 1.3.0
imageio 2.37.3
opencv-python 4.13.0
numpy 2.2.6
scipy 1.15.3
pyyaml 6.0.3
curobo import ok
```

The tested local cuRobo checkout is:

```text
NVlabs/curobo commit 8726021
```

The RTX 3090 compute capability is `8.6`, so CUDA extension builds should use:

```bash
export TORCH_CUDA_ARCH_LIST=8.6
```

## Environment Layout

Expected project layout after installation:

```text
$PROJECT_ROOT/
  checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt
  maniskill_curobo/
    envs/maniskill_curobo/
    external/curobo/
    scripts/run_full_pipeline.py
    scripts/execute_curobo_pick.py
```

The two Python environments are:

```text
graduate:
  Purpose: ZeroGrasp inference
  Used by: maniskill_codex.run_zerograsp_inference

maniskill_curobo/envs/maniskill_curobo:
  Purpose: ManiSkill simulation, scene export, cuRobo planning, trajectory execution, video recording
  Used by: maniskill_curobo/scripts/run_full_pipeline.py through --maniskill-python
```

Do not install cuRobo into the old `maniskill` conda environment. This project intentionally keeps cuRobo isolated under `./maniskill_curobo/envs/maniskill_curobo`.

## Task 1: Prepare The 3090 Host

**Files:**
- No project files are modified in this task.

- [ ] **Step 1: Confirm GPU, driver, and conda**

Run:

```bash
nvidia-smi
conda --version
git --version
```

Expected:

```text
nvidia-smi shows NVIDIA GeForce RTX 3090
conda prints a version string
git prints a version string
```

- [ ] **Step 2: Install OS packages**

Run:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build git git-lfs ffmpeg \
  libegl1 libgl1 libglib2.0-0 libglvnd0 libvulkan1 \
  libsm6 libx11-6 libxext6 libxrender1 \
  mesa-vulkan-drivers vulkan-tools
```

Expected:

```text
apt finishes without package errors
```

- [ ] **Step 3: Enable Git LFS**

Run:

```bash
git lfs install
```

Expected:

```text
Git LFS initialized.
```

- [ ] **Step 4: Set the project root variable**

Use the final project location on the 3090 host. The examples below assume:

```bash
export PROJECT_ROOT="$HOME/zerograsp_mainline_minimal"
mkdir -p "$(dirname "$PROJECT_ROOT")"
```

Expected:

```text
No output
```

- [ ] **Step 5: Copy the project into `PROJECT_ROOT`**

Before continuing, copy the complete repository folder from the current development machine to:

```text
$HOME/zerograsp_mainline_minimal
```

Then run on the 3090 host:

```bash
export PROJECT_ROOT="$HOME/zerograsp_mainline_minimal"
test -d "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
pwd
```

Expected:

```text
The printed path ends with zerograsp_mainline_minimal
```

- [ ] **Step 6: Confirm the ZeroGrasp checkpoint exists**

Run:

```bash
test -f checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt
echo "checkpoint ok"
```

Expected:

```text
checkpoint ok
```

- [ ] **Step 7: Export common environment variables**

Run before installation and before smoke tests:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Expected:

```text
No output
```

## Task 2: Prepare Or Verify The ZeroGrasp Environment `graduate`

**Files:**
- Uses: `docs/superpowers/plans/2026-05-13-3090-conda-dual-env-install.md`
- Uses: `checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt`
- No project files are modified in this task.

- [ ] **Step 1: Check whether `graduate` already exists**

Run:

```bash
conda env list
```

Expected:

```text
The list contains an environment named graduate
```

- [ ] **Step 2: Install `graduate` if it is missing**

If `graduate` is missing, follow Task 2 in:

```text
docs/superpowers/plans/2026-05-13-3090-conda-dual-env-install.md
```

Use these 3090-specific exports before building ZeroGrasp CUDA extensions:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Expected:

```text
graduate installs torch, ocnn, dwconv, ofe, and ZeroGrasp dependencies
```

- [ ] **Step 3: Verify `graduate` imports**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graduate
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
python - <<'PY'
import torch
import ocnn
import dwconv
import ofe

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("zerograsp deps ok")
PY
```

Expected:

```text
cuda True
gpu NVIDIA GeForce RTX 3090
zerograsp deps ok
```

## Task 3: Create The Isolated `maniskill_curobo` Environment

**Files:**
- Uses: `maniskill_curobo/scripts/check_host.py`
- Uses: `maniskill_curobo/scripts/create_env_b.sh`
- Creates: `maniskill_curobo/envs/maniskill_curobo/`
- Creates: `maniskill_curobo/external/curobo/`
- Creates: `maniskill_curobo/logs/create_env_b.log`

- [ ] **Step 1: Run host diagnostics**

Run:

```bash
cd $PROJECT_ROOT
export TORCH_CUDA_ARCH_LIST=8.6
python maniskill_curobo/scripts/check_host.py
```

Expected:

```text
nvidia_smi_query shows NVIDIA GeForce RTX 3090
TORCH_CUDA_ARCH_LIST=8.6
disk and memory information are printed
```

- [ ] **Step 2: Pre-clone cuRobo at the tested commit**

Run:

```bash
cd $PROJECT_ROOT
mkdir -p maniskill_curobo/external
git clone https://github.com/NVlabs/curobo.git maniskill_curobo/external/curobo
cd maniskill_curobo/external/curobo
git checkout 8726021
git lfs pull
cd $PROJECT_ROOT
```

Expected:

```text
HEAD is now at 8726021...
Git LFS downloads cuRobo assets without errors
```

- [ ] **Step 3: Build the isolated environment**

Run:

```bash
cd $PROJECT_ROOT
export TORCH_CUDA_ARCH_LIST=8.6
bash maniskill_curobo/scripts/create_env_b.sh
```

Expected:

```text
maniskill_curobo/envs/maniskill_curobo is created
torch 2.11.0+cu128 is installed
mani_skill 3.0.1 is installed
curobo is installed editable from maniskill_curobo/external/curobo
maniskill_curobo/logs/create_env_b.log is written
```

- [ ] **Step 4: Activate the isolated environment**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $PROJECT_ROOT/maniskill_curobo/envs/maniskill_curobo
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Expected:

```text
The shell prompt shows the path-based maniskill_curobo environment
```

## Task 4: Verify ManiSkill, SAPIEN, And cuRobo Imports

**Files:**
- Uses: `maniskill_curobo/scripts/smoke_imports.py`

- [ ] **Step 1: Run project smoke imports**

Run:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
python maniskill_curobo/scripts/smoke_imports.py
```

Expected:

```text
torch cuda: ok
mani_skill + sapien import: ok
curobo import: ok
```

- [ ] **Step 2: Print exact package versions**

Run:

```bash
python - <<'PY'
import sys
import torch
import mani_skill
import sapien
import gymnasium
import imageio
import cv2
import numpy
import scipy
import yaml
import curobo

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mani_skill", mani_skill.__version__)
print("sapien", sapien.__version__)
print("gymnasium", gymnasium.__version__)
print("imageio", imageio.__version__)
print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pyyaml", yaml.__version__)
print("curobo import ok")
PY
```

Expected:

```text
python 3.10.x
torch 2.11.0+cu128
cuda True
gpu NVIDIA GeForce RTX 3090
mani_skill 3.0.1
sapien 3.0.3
gymnasium 1.3.0
curobo import ok
```

## Task 5: Run cuRobo And ManiSkill Smoke Tests

**Files:**
- Uses: `maniskill_curobo/scripts/smoke_motion_planning.sh`
- Uses: `maniskill_curobo/scripts/smoke_maniskill_after_curobo.py`
- Uses: `maniskill_curobo/scripts/smoke_maniskill_qpos_to_curobo.py`
- Outputs: `maniskill_curobo/smoke_tests/`

- [ ] **Step 1: Run cuRobo motion planning smoke tests**

Run:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
bash maniskill_curobo/scripts/smoke_motion_planning.sh
```

Expected:

```text
maniskill_curobo/smoke_tests/motion_planning/motion_plan.pdf is written
maniskill_curobo/smoke_tests/motion_planning/grasp_plan.pdf is written
```

- [ ] **Step 2: Verify ManiSkill after cuRobo import**

Run:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
python maniskill_curobo/scripts/smoke_maniskill_after_curobo.py
```

Expected:

```text
ManiSkill can create a Panda task after cuRobo has been imported
```

- [ ] **Step 3: Verify ManiSkill qpos to cuRobo bridge**

Run:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
python maniskill_curobo/scripts/smoke_maniskill_qpos_to_curobo.py
```

Expected:

```text
maniskill_curobo/smoke_tests/bridge/maniskill_qpos_to_curobo_plan.npz is written
```

## Task 6: Run The Real ZeroGrasp + PickSingle + cuRobo Pipeline

**Files:**
- Uses: `maniskill_curobo/scripts/run_full_pipeline.py`
- Outputs: `maniskill_curobo/runs/install_smoke_picksingle_curobo/`

- [ ] **Step 1: Run PickSingle smoke pipeline**

Run:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python maniskill_curobo/scripts/run_full_pipeline.py \
  --env-id PickSingleYCB-v1 \
  --seed 1 \
  --run-name install_smoke_picksingle_curobo \
  --mask-mode task-target \
  --approach-axis positive-x \
  --camera-eye -0.30 0.0 0.55 \
  --camera-target 0.05 0.0 0.08 \
  --zerograsp-env-name graduate \
  --maniskill-python $PROJECT_ROOT/maniskill_curobo/envs/maniskill_curobo/bin/python
```

Expected:

```text
Run folder: .../maniskill_curobo/runs/install_smoke_picksingle_curobo
Recommended grasp: .../zg_output/recommended_grasp_top1.json
Projection: .../grasp_projection.png
Video: .../execution.mp4
```

- [ ] **Step 2: Inspect PickSingle output files**

Run:

```bash
cd $PROJECT_ROOT
find maniskill_curobo/runs/install_smoke_picksingle_curobo -maxdepth 2 -type f | sort | sed -n '1,120p'
```

Expected:

```text
zg_input/rgb.png
zg_input/depth.png
zg_input/mask.png
zg_input/camera.json
zg_input/scene.json
zg_output/recommended_grasp_top1.json
grasp_projection.png
curobo_scene.yml
curobo_scene_metadata.json
planning_diagnostics.json
execution.mp4
pipeline_manifest.json
run_manifest.json
```

- [ ] **Step 3: Inspect planning diagnostics**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("maniskill_curobo/runs/install_smoke_picksingle_curobo/planning_diagnostics.json")
data = json.loads(path.read_text())
for item in data:
    print(item.get("stage"), item.get("success"), item.get("status"), item.get("failure_reason"))
PY
```

Expected:

```text
Each planned stage prints a stage name and success/status fields
```

## Task 7: Run The Real ZeroGrasp + PickClutter + cuRobo Pipeline

**Files:**
- Uses: `maniskill_curobo/scripts/run_full_pipeline.py`
- Outputs: `maniskill_curobo/runs/install_smoke_pickclutter_curobo/`

- [ ] **Step 1: Run PickClutter smoke pipeline**

Run:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python maniskill_curobo/scripts/run_full_pipeline.py \
  --env-id PickClutterYCB-v1 \
  --seed 1 \
  --run-name install_smoke_pickclutter_curobo \
  --mask-mode task-target \
  --approach-axis positive-x \
  --camera-eye -0.20 0.0 0.27 \
  --camera-target 0.05 0.0 0.08 \
  --zerograsp-env-name graduate \
  --maniskill-python $PROJECT_ROOT/maniskill_curobo/envs/maniskill_curobo/bin/python
```

Expected:

```text
Run folder: .../maniskill_curobo/runs/install_smoke_pickclutter_curobo
ZeroGrasp output, grasp projection, cuRobo scene, diagnostics, and execution video are written
```

- [ ] **Step 2: Inspect PickClutter output files**

Run:

```bash
cd $PROJECT_ROOT
find maniskill_curobo/runs/install_smoke_pickclutter_curobo -maxdepth 2 -type f | sort | sed -n '1,120p'
```

Expected:

```text
zg_input/rgb.png
zg_input/depth.png
zg_input/mask.png
zg_output/recommended_grasp_top1.json
grasp_projection.png
curobo_scene.yml
curobo_scene_metadata.json
planning_diagnostics.json
execution.mp4
pipeline_manifest.json
run_manifest.json
```

## Task 8: Run Unit Tests

**Files:**
- Uses: `maniskill_curobo/tests/`

- [ ] **Step 1: Run the `maniskill_curobo` test suite**

Run in the activated `maniskill_curobo` environment:

```bash
cd $PROJECT_ROOT
export PYTHONPATH="$PWD"
python -m unittest discover maniskill_curobo/tests -v
```

Expected:

```text
All tests in maniskill_curobo/tests pass
```

- [ ] **Step 2: Compile main scripts**

Run:

```bash
python -m py_compile \
  maniskill_curobo/scripts/run_full_pipeline.py \
  maniskill_curobo/scripts/execute_curobo_pick.py \
  maniskill_curobo/scene_export.py \
  maniskill_curobo/joint_trajectory_utils.py
```

Expected:

```text
No output
```

## Troubleshooting

### `create_env_b.sh` Builds For The Wrong GPU Architecture

`create_env_b.sh` defaults to `TORCH_CUDA_ARCH_LIST=12.0` when the variable is unset, because the original local machine was an RTX 5070 Ti. On RTX 3090, always export:

```bash
export TORCH_CUDA_ARCH_LIST=8.6
```

Then rebuild cuRobo:

```bash
cd $PROJECT_ROOT
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $PROJECT_ROOT/maniskill_curobo/envs/maniskill_curobo
export TORCH_CUDA_ARCH_LIST=8.6
cd maniskill_curobo/external/curobo
python -m pip install -e . --no-build-isolation --force-reinstall
```

### `torch.cuda.is_available()` Is False

Check the active environment and driver:

```bash
which python
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
nvidia-smi
```

Expected:

```text
which python points inside maniskill_curobo/envs/maniskill_curobo
torch includes +cu128
nvidia-smi works
```

If the NVIDIA driver is too old for CUDA 12.8 wheels, upgrade the driver first. Prefer upgrading the driver over downgrading this environment, because cuRobo and the existing project have been tested with CUDA 12.8 wheels.

### cuRobo Fails To Clone Or Missing Large Files

Run:

```bash
cd $PROJECT_ROOT/maniskill_curobo/external/curobo
git lfs install
git lfs pull
python -m pip install -e . --no-build-isolation
```

Expected:

```text
Git LFS downloads missing assets
cuRobo editable install finishes
```

### SAPIEN Or ManiSkill Rendering Fails

Reinstall Vulkan and GL runtime packages:

```bash
sudo apt install -y \
  libegl1 libgl1 libglib2.0-0 libglvnd0 libvulkan1 \
  mesa-vulkan-drivers vulkan-tools
```

Then test:

```bash
vulkaninfo | sed -n '1,40p'
```

Expected:

```text
Vulkan instance information is printed
```

### Full Pipeline Cannot Find `graduate`

Confirm the env name:

```bash
conda env list
```

If the ZeroGrasp environment has another name, pass it explicitly:

```bash
export ZEROGRASP_ENV_NAME=graduate
python maniskill_curobo/scripts/run_full_pipeline.py \
  --zerograsp-env-name $ZEROGRASP_ENV_NAME \
  --maniskill-python $PROJECT_ROOT/maniskill_curobo/envs/maniskill_curobo/bin/python
```

### cuRobo Planning Fails During A Smoke Run

The smoke run can still produce useful diagnostics. Inspect:

```bash
cat maniskill_curobo/runs/install_smoke_picksingle_curobo/planning_diagnostics.json
cat maniskill_curobo/runs/install_smoke_picksingle_curobo/logs/execute.stderr.log
```

Planning can fail because the sampled ZeroGrasp target is unreachable, collides with the exported ManiSkill scene, has an awkward approach orientation, or cannot be connected from the current robot state with the current cuRobo seed budget. Installation is considered valid if imports, smoke planners, scene export, and at least one simple pipeline run produce the expected artifacts; grasp success is a separate algorithmic evaluation.

## Final Acceptance Checklist

- [ ] `conda activate graduate` works.
- [ ] In `graduate`, `import torch, ocnn, dwconv, ofe` works and `torch.cuda.is_available()` prints `True`.
- [ ] `conda activate $PROJECT_ROOT/maniskill_curobo/envs/maniskill_curobo` works.
- [ ] In `maniskill_curobo`, `import torch, mani_skill, sapien, curobo` works and `torch.cuda.is_available()` prints `True`.
- [ ] `bash maniskill_curobo/scripts/smoke_motion_planning.sh` writes smoke planning artifacts.
- [ ] `python maniskill_curobo/scripts/smoke_maniskill_after_curobo.py` succeeds.
- [ ] `python maniskill_curobo/scripts/smoke_maniskill_qpos_to_curobo.py` writes a bridge `.npz`.
- [ ] `maniskill_curobo/runs/install_smoke_picksingle_curobo/zg_output/recommended_grasp_top1.json` exists.
- [ ] `maniskill_curobo/runs/install_smoke_picksingle_curobo/grasp_projection.png` exists.
- [ ] `maniskill_curobo/runs/install_smoke_picksingle_curobo/curobo_scene.yml` exists.
- [ ] `maniskill_curobo/runs/install_smoke_picksingle_curobo/planning_diagnostics.json` exists.
- [ ] `maniskill_curobo/runs/install_smoke_picksingle_curobo/execution.mp4` exists.
