#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${ZERO_GRASP_IMAGE_NAME:-zerograsp-mainline:minimal}"

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <rgb_path> <depth_path> <mask_path> <camera_info_path> <output_dir> [extra run_mainline args...]"
  exit 1
fi

RGB_PATH="$(realpath "$1")"
DEPTH_PATH="$(realpath "$2")"
MASK_PATH="$(realpath "$3")"
CAMERA_INFO_PATH="$(realpath "$4")"
OUTPUT_DIR="$(realpath "$5")"
shift 5

mkdir -p "${OUTPUT_DIR}"

docker run --rm --gpus all \
  -v "${RGB_PATH%/*}:/mnt/rgb:ro" \
  -v "${DEPTH_PATH%/*}:/mnt/depth:ro" \
  -v "${MASK_PATH%/*}:/mnt/mask:ro" \
  -v "${CAMERA_INFO_PATH%/*}:/mnt/camera:ro" \
  -v "${OUTPUT_DIR}:/outputs" \
  "${IMAGE_NAME}" \
  --img_path "/mnt/rgb/${RGB_PATH##*/}" \
  --depth_path "/mnt/depth/${DEPTH_PATH##*/}" \
  --mask_path "/mnt/mask/${MASK_PATH##*/}" \
  --camera_info_path "/mnt/camera/${CAMERA_INFO_PATH##*/}" \
  --output_dir /outputs \
  "$@"
