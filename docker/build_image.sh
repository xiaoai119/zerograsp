#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${ZERO_GRASP_IMAGE_NAME:-zerograsp-mainline:minimal}"
docker build --rm -f docker/Dockerfile -t "${IMAGE_NAME}" .
