#!/usr/bin/env bash

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
ENV_DIR="${ROOT}/envs/maniskill_curobo"
LOG_DIR="${ROOT}/logs"
EXTERNAL_DIR="${ROOT}/external"
CUROBO_DIR="${EXTERNAL_DIR}/curobo"

mkdir -p "${LOG_DIR}" "${EXTERNAL_DIR}"

exec > >(tee "${LOG_DIR}/create_env_b.log") 2>&1

echo "[create_env_b] root=${ROOT}"
echo "[create_env_b] env=${ENV_DIR}"
echo "[create_env_b] repo=${REPO_ROOT}"
echo "[create_env_b] started=$(date -Is)"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "${ENV_DIR}" ]; then
  conda create -p "${ENV_DIR}" python=3.10 -y
else
  echo "[create_env_b] existing env found, reusing ${ENV_DIR}"
fi

conda activate "${ENV_DIR}"

python -m pip install --upgrade pip wheel "setuptools<82"

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0

python -m pip install \
  mani_skill==3.0.1 \
  sapien==3.0.3 \
  gymnasium==1.3.0 \
  imageio==2.37.3 \
  imageio-ffmpeg==0.6.0 \
  pillow \
  opencv-python-headless \
  numpy \
  scipy \
  pyyaml

python -m pip install 'cuda-core[cu12]>=0.7'

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[create_env_b] nvcc not found after Python package install; installing CUDA nvcc into conda env"
  conda install -y -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev cuda-cccl
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
echo "[create_env_b] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

if [ ! -d "${CUROBO_DIR}/.git" ]; then
  git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
else
  echo "[create_env_b] existing cuRobo clone found, reusing ${CUROBO_DIR}"
fi

cd "${CUROBO_DIR}"
git lfs install || true
git lfs pull || true
python -m pip install -e . --no-build-isolation

cd "${REPO_ROOT}"
python "${ROOT}/scripts/smoke_imports.py"

echo "[create_env_b] finished=$(date -Is)"
