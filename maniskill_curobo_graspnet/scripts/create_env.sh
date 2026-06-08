#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${ROOT}/maniskill_curobo_graspnet/envs/graspnet"
BASELINE_DIR="${ROOT}/maniskill_curobo_graspnet/external/graspnet-baseline"
CONDA="${CONDA:-/home/openclaw-server/miniconda3/bin/conda}"
ANYGRASP_ENV="${ROOT}/maniskill_curobo_anygrasp/envs/anygrasp"
SOURCE_ENV="${SOURCE_ENV:-${ANYGRASP_ENV}}"
GRADUATE_ENV="/home/openclaw-server/miniconda3/envs/graduate"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
BASELINE_COMMIT="${BASELINE_COMMIT:-280c215129f759ed8649cb4e89fc5dfee55f4f80}"

if [[ ! -x "${SOURCE_ENV}/bin/python" ]]; then
  SOURCE_ENV="${GRADUATE_ENV}"
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${CONDA}" create -y -p "${ENV_DIR}" --clone "${SOURCE_ENV}"
fi

if [[ ! -d "${BASELINE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${BASELINE_DIR}")"
  git clone https://github.com/graspnet/graspnet-baseline.git "${BASELINE_DIR}"
  git -C "${BASELINE_DIR}" checkout "${BASELINE_COMMIT}"
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

"${ENV_DIR}/bin/python" -m pip install -r "${BASELINE_DIR}/requirements.txt"
"${ENV_DIR}/bin/python" -m pip install "gdown"
# graspnetAPI pins very old numeric dependencies. Install its package code only,
# then restore the versions already known to work with this Python/PyTorch stack.
"${ENV_DIR}/bin/python" -m pip install "graspnetAPI" --no-deps
"${ENV_DIR}/bin/python" -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  "transforms3d==0.4.2" \
  "packaging<25"

(
  cd "${BASELINE_DIR}/pointnet2"
  "${ENV_DIR}/bin/python" setup.py install
)
if ! "${ENV_DIR}/bin/python" -c "import knn_pytorch" >/dev/null 2>&1; then
  (
    cd "${BASELINE_DIR}/knn"
    "${ENV_DIR}/bin/python" setup.py install
  )
fi

PYTHONPATH="${ROOT}" "${ENV_DIR}/bin/python" -m \
  maniskill_curobo_graspnet.scripts.check_runtime || true
