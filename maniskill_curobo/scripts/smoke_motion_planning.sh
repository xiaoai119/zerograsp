#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT}/smoke_tests/motion_planning"

mkdir -p "${OUTPUT_DIR}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

python -m curobo.examples.getting_started.motion_planning \
  --mode pose \
  --output-dir "${OUTPUT_DIR}"

python -m curobo.examples.getting_started.motion_planning \
  --mode grasp \
  --output-dir "${OUTPUT_DIR}"
