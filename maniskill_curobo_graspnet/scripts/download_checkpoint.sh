#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PYTHON="${ROOT}/maniskill_curobo_graspnet/envs/graspnet/bin/python"
OUT_DIR="${ROOT}/maniskill_curobo_graspnet/checkpoints"
CHECKPOINT="${CHECKPOINT:-rs}"

mkdir -p "${OUT_DIR}"

if [[ "${CHECKPOINT}" == "rs" ]]; then
  file_id="1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk"
  output="${OUT_DIR}/checkpoint-rs.tar"
elif [[ "${CHECKPOINT}" == "kn" ]]; then
  file_id="1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX"
  output="${OUT_DIR}/checkpoint-kn.tar"
else
  echo "CHECKPOINT must be either rs or kn." >&2
  exit 1
fi

if [[ -f "${output}" ]]; then
  echo "${output}"
  exit 0
fi

"${ENV_PYTHON}" -m pip install gdown
"${ENV_PYTHON}" -m gdown "https://drive.google.com/uc?id=${file_id}" -O "${output}"
echo "${output}"
