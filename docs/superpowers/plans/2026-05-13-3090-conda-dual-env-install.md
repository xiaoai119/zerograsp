# 3090 Conda Dual Environment Installation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install this ZeroGrasp + ManiSkill project on an RTX 3090 host using the same two-conda-environment layout as the current working machine.

**Architecture:** Keep ZeroGrasp inference and ManiSkill simulation separated. The `graduate` conda environment owns ZeroGrasp, CUDA PyTorch extensions, ocnn/dwconv, and octree feature extraction; the `maniskill` conda environment owns ManiSkill/SAPIEN simulation, RGBD export, grasp execution, and video recording. The project scripts switch between these environments automatically through `maniskill_codex.run_full_pipeline`.

**Tech Stack:** Ubuntu, NVIDIA driver, conda, Python 3.11, PyTorch CUDA wheels, torch-scatter, torch-cluster, ocnn, dwconv, ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.3.0.

---

## Known Working Versions

The current project has been checked against this local environment shape:

```text
graduate:
  Python 3.11
  torch 2.11.0+cu128
  torchvision 0.26.0+cu128
  torch-scatter 2.1.2+pt211cu128
  torch-cluster 1.6.3+pt211cu128
  ocnn 2.3.1
  dwconv 1.1.0

maniskill:
  Python 3.11
  torch 2.11.0+cu128
  mani_skill 3.0.1
  sapien 3.0.3
  gymnasium 1.3.0
  imageio 2.37.3
  imageio-ffmpeg 0.6.0
```

Use CUDA 12.8 wheels if the 3090 host driver supports them. If the driver is older, use the fallback CUDA 12.1 commands in the final section.

## Task 1: Prepare The Host

**Files:**
- No project files are modified in this task.

- [ ] **Step 1: Confirm GPU and driver**

Run:

```bash
nvidia-smi
conda --version
```

Expected:

```text
NVIDIA-SMI prints RTX 3090 or another NVIDIA GPU
conda prints a version string
```

- [ ] **Step 2: Install OS packages**

Run:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build git ffmpeg \
  libegl1 libgl1 libglib2.0-0 libglvnd0 libvulkan1 \
  libsm6 libx11-6 libxext6 libxrender1 \
  mesa-vulkan-drivers vulkan-tools
```

Expected:

```text
apt finishes without package errors
```

- [ ] **Step 3: Copy the project to the host**

Put the project on the target machine, then enter the project root:

```bash
cd /path/to/zerograsp_mainline_minimal
```

Confirm the checkpoint exists:

```bash
test -f checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt && echo "checkpoint ok"
```

Expected:

```text
checkpoint ok
```

- [ ] **Step 4: Set common environment variables**

Run this before building CUDA extensions and before executing the pipeline:

```bash
cd /path/to/zerograsp_mainline_minimal
export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Expected:

```text
No output
```

## Task 2: Install ZeroGrasp Environment `graduate`

**Files:**
- Uses: `requirements.upstream.txt`
- Uses: `third_party/octree_feature_extractor/setup.py`
- No project files are modified in this task.

- [ ] **Step 1: Create and activate the environment**

Run:

```bash
conda create -n graduate python=3.11 -y
conda activate graduate
```

Expected:

```text
The shell prompt shows the graduate environment
```

- [ ] **Step 2: Install build tools**

Run:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install cmake ninja
```

Expected:

```text
pip finishes without errors
```

- [ ] **Step 3: Install PyTorch CUDA 12.8 wheels**

Run:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
```

Expected:

```text
Successfully installed torch torchvision
```

- [ ] **Step 4: Install PyG CUDA extension wheels**

Run:

```bash
python -m pip install \
  torch-scatter \
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

python -m pip install \
  torch-cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

Expected:

```text
Successfully installed torch-scatter
Successfully installed torch-cluster
```

- [ ] **Step 5: Install dwconv and ocnn**

Run from the project root with `TORCH_CUDA_ARCH_LIST=8.6` already exported:

```bash
cd /path/to/zerograsp_mainline_minimal
export TORCH_CUDA_ARCH_LIST=8.6

python -m pip install --no-build-isolation \
  "dwconv @ git+https://github.com/octree-nn/dwconv.git@ae53057eaf36dab01aa2727fcc93a749fd995af5#egg=dwconv" \
  "ocnn @ git+https://github.com/octree-nn/ocnn-pytorch.git"
```

Expected:

```text
Successfully installed dwconv ocnn
```

- [ ] **Step 6: Install upstream ZeroGrasp dependencies**

Run:

```bash
cd /path/to/zerograsp_mainline_minimal
grep -Ev '^(dwconv|ocnn) @ ' requirements.upstream.txt > /tmp/requirements.upstream.filtered.txt
python -m pip install --no-build-isolation -r /tmp/requirements.upstream.filtered.txt
python -m pip install graspnetAPI --no-deps
python -m pip install transforms3d==0.4.2 autolab_core cvxopt grasp_nms
```

Expected:

```text
pip finishes without dependency build errors
```

- [ ] **Step 7: Install octree feature extractor**

Run:

```bash
cd /path/to/zerograsp_mainline_minimal
export TORCH_CUDA_ARCH_LIST=8.6
cd third_party/octree_feature_extractor
python setup.py install
cd ../..
```

Expected:

```text
Finished processing dependencies for ofe-pytorch
```

- [ ] **Step 8: Verify `graduate`**

Run:

```bash
cd /path/to/zerograsp_mainline_minimal
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
torch 2.11.0+cu128
cuda True
gpu NVIDIA GeForce RTX 3090
zerograsp deps ok
```

## Task 3: Install ManiSkill Environment `maniskill`

**Files:**
- No project files are modified in this task.

- [ ] **Step 1: Create and activate the environment**

Run:

```bash
conda create -n maniskill python=3.11 -y
conda activate maniskill
```

Expected:

```text
The shell prompt shows the maniskill environment
```

- [ ] **Step 2: Install PyTorch CUDA 12.8 wheels**

Run:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
```

Expected:

```text
Successfully installed torch torchvision
```

- [ ] **Step 3: Install ManiSkill and video dependencies**

Run:

```bash
python -m pip install \
  mani_skill==3.0.1 \
  sapien==3.0.3 \
  gymnasium==1.3.0 \
  imageio==2.37.3 \
  imageio-ffmpeg==0.6.0 \
  pillow \
  opencv-python-headless
```

Expected:

```text
pip finishes without errors
```

- [ ] **Step 4: Verify `maniskill`**

Run:

```bash
python - <<'PY'
import torch
import mani_skill
import sapien
import gymnasium
import imageio

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mani_skill ok")
print("sapien ok")
print("gymnasium ok")
print("imageio ok")
PY
```

Expected:

```text
torch 2.11.0+cu128 cuda True
mani_skill ok
sapien ok
gymnasium ok
imageio ok
```

## Task 4: Verify The Full Pipeline

**Files:**
- Uses: `maniskill_codex/run_full_pipeline.py`
- Outputs: `maniskill_codex/runs/install_smoke_picksingle/`
- Outputs: `maniskill_codex/runs/install_smoke_pickclutter/`

- [ ] **Step 1: Run PickSingle smoke test**

Run:

```bash
cd /path/to/zerograsp_mainline_minimal
conda activate maniskill

export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. python -m maniskill_codex.run_full_pipeline \
  --run-name install_smoke_picksingle \
  --env-id PickSingleYCB-v1 \
  --seed 1 \
  --mask-mode task-target \
  --approach-axis positive-x \
  --maniskill-env-name maniskill \
  --zerograsp-env-name graduate
```

Expected:

```text
run_manifest.json is written under maniskill_codex/runs/install_smoke_picksingle
execution.mp4 is written under maniskill_codex/runs/install_smoke_picksingle
```

- [ ] **Step 2: Run PickClutter smoke test with the current preferred camera**

Run:

```bash
cd /path/to/zerograsp_mainline_minimal
conda activate maniskill

export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. python -m maniskill_codex.run_full_pipeline \
  --run-name install_smoke_pickclutter \
  --env-id PickClutterYCB-v1 \
  --seed 1 \
  --mask-mode task-target \
  --camera-eye -0.2 0.0 0.27 \
  --camera-target 0.05 0.0 0.08 \
  --approach-axis positive-x \
  --pregrasp-max-steps 200 \
  --descend-settle-pos-tolerance 0.02 \
  --maniskill-env-name maniskill \
  --zerograsp-env-name graduate
```

Expected:

```text
run_manifest.json is written under maniskill_codex/runs/install_smoke_pickclutter
zg_input/rgb.png is written
zg_input/mask.png is written
zg_output/recommended_grasp_top1.json is written
grasp_projection.png is written
execution.mp4 is written
```

- [ ] **Step 3: Inspect outputs**

Run:

```bash
ls -R maniskill_codex/runs/install_smoke_pickclutter | sed -n '1,80p'
python - <<'PY'
import json
from pathlib import Path

manifest = Path("maniskill_codex/runs/install_smoke_pickclutter/run_manifest.json")
data = json.loads(manifest.read_text())
print(json.dumps({
    "run_dir": str(manifest.parent),
    "env_id": data.get("env_id"),
    "seed": data.get("seed"),
    "status": data.get("status"),
}, indent=2))
PY
```

Expected:

```text
The output listing includes zg_input, zg_output, grasp_projection.png, execution.mp4, logs, and run_manifest.json
```

## Task 5: Optional Batch Verification

**Files:**
- Uses: `maniskill_codex/run_seed_batch.py`
- Outputs: `maniskill_codex/batch_runs/install_pickclutter_1_3/`

- [ ] **Step 1: Run a small PickClutter batch**

Run:

```bash
cd /path/to/zerograsp_mainline_minimal
conda activate maniskill

export PYTHONPATH="$PWD"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. python -m maniskill_codex.run_seed_batch \
  --seed-range 1-3 \
  --output-root maniskill_codex/batch_runs/install_pickclutter_1_3 \
  --batch-name install_pickclutter_1_3 \
  --env-id PickClutterYCB-v1 \
  --mask-mode task-target \
  --camera-eye -0.2 0.0 0.27 \
  --camera-target 0.05 0.0 0.08 \
  --approach-axis positive-x \
  --pregrasp-max-steps 200 \
  --descend-settle-pos-tolerance 0.02 \
  --maniskill-env-name maniskill \
  --zerograsp-env-name graduate
```

Expected:

```text
install_pickclutter_1_3_summary.json is written
install_pickclutter_1_3_summary.tsv is written
Each seed has its own run directory
```

## Troubleshooting

### Driver Too Old For CUDA 12.8 Wheels

If `torch.cuda.is_available()` is false, or CUDA initialization fails because the NVIDIA driver is too old, reinstall both environments with CUDA 12.1 PyTorch wheels instead.

For both `graduate` and `maniskill`, replace the PyTorch install command with:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.0 torchvision==0.17.0
```

In `graduate`, replace the PyG extension commands with:

```bash
python -m pip install \
  torch-scatter \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

python -m pip install \
  torch-cluster \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

Then rebuild `dwconv`, `ocnn`, and `third_party/octree_feature_extractor` with:

```bash
export TORCH_CUDA_ARCH_LIST=8.6
python -m pip install --force-reinstall --no-build-isolation \
  "dwconv @ git+https://github.com/octree-nn/dwconv.git@ae53057eaf36dab01aa2727fcc93a749fd995af5#egg=dwconv" \
  "ocnn @ git+https://github.com/octree-nn/ocnn-pytorch.git"

cd /path/to/zerograsp_mainline_minimal/third_party/octree_feature_extractor
python setup.py install
```

### CUDA Extension Build Fails

Check that this was exported before installing `dwconv`, `ocnn`, or OFE:

```bash
export TORCH_CUDA_ARCH_LIST=8.6
```

Also check that the active environment has the intended PyTorch:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

### `torch-scatter` Or `torch-cluster` Import Fails

The wheel must match the PyTorch version and CUDA wheel. For PyTorch `2.11.0+cu128`, use:

```text
https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

For PyTorch `2.2.0+cu121`, use:

```text
https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### SAPIEN Or ManiSkill Rendering Fails

Reinstall the graphics libraries:

```bash
sudo apt install -y \
  libegl1 libgl1 libglib2.0-0 libglvnd0 libvulkan1 \
  mesa-vulkan-drivers vulkan-tools
```

Then test Vulkan:

```bash
vulkaninfo | sed -n '1,40p'
```

### Pipeline Cannot Find The Conda Environments

Make sure the environment names match exactly:

```bash
conda env list
```

The pipeline defaults are:

```text
--maniskill-env-name maniskill
--zerograsp-env-name graduate
```

If custom names are used, pass them explicitly to `run_full_pipeline.py` and `run_seed_batch.py`.

## Final Acceptance Checklist

- [ ] `conda run -n graduate python -c "import torch, ocnn, dwconv, ofe; print(torch.cuda.is_available())"` prints `True`.
- [ ] `conda run -n maniskill python -c "import mani_skill, sapien, gymnasium; print('ok')"` prints `ok`.
- [ ] `maniskill_codex/runs/install_smoke_picksingle/run_manifest.json` exists.
- [ ] `maniskill_codex/runs/install_smoke_picksingle/execution.mp4` exists.
- [ ] `maniskill_codex/runs/install_smoke_pickclutter/zg_output/recommended_grasp_top1.json` exists.
- [ ] `maniskill_codex/runs/install_smoke_pickclutter/grasp_projection.png` exists.
- [ ] `maniskill_codex/runs/install_smoke_pickclutter/execution.mp4` exists.
