#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${ZERO_GRASP_FULL_IMAGE_NAME:-zerograsp-maniskill:3090}"

docker build --rm -f docker/Dockerfile.full -t "${IMAGE_NAME}" .
