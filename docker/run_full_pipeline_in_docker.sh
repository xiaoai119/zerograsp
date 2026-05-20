#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${ZERO_GRASP_FULL_IMAGE_NAME:-zerograsp-maniskill:3090}"
OUTPUT_ROOT="$(realpath "${ZERO_GRASP_OUTPUT_ROOT:-./docker_runs}")"
HOST_WORKDIR="$(realpath "${ZERO_GRASP_HOST_WORKDIR:-$PWD}")"

mkdir -p "${OUTPUT_ROOT}"

docker run --rm --gpus all --ipc=host --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${OUTPUT_ROOT}:/workspace/output" \
  -v "${HOST_WORKDIR}:/workspace/host:ro" \
  "${IMAGE_NAME}" \
  --no-conda \
  --output-root /workspace/output \
  "$@"
