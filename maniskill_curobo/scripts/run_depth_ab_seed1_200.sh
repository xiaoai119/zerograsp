#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/openclaw-server/miniconda3/envs/graduate/bin/python}"
PICKSINGLE_CANDIDATES="maniskill_collect_data/zerograsp_sparse_sft_plan/runs/topk_label_mining/keepgood_seed1_300_collision_top1"
PICKSINGLE_OUTPUT="maniskill_curobo/runs/depth_ab_picksingle_seed1_200"
PICKCLUTTER_OUTPUT="maniskill_curobo/runs/depth_ab_pickclutter_seed1_200"

COMMON_ARGS=(
  --seed-start 1
  --seed-end 200
  --baseline-depth-scale 0.0
  --depth-scale 1.0
  --depth-auto-fallback
  --grasp-depth-max-offset 0.04
  --workspace-z-min 0.01
  --close-steps 20
  --settle-steps 50
  --reuse-existing
)

echo "[1/2] PickSingleYCB-v1 seed1-200"
PYTHONPATH=. "${PYTHON_BIN}" \
  -m maniskill_curobo.scripts.run_zerograsp_depth_ab_batch \
  --env-id PickSingleYCB-v1 \
  --output-root "${PICKSINGLE_OUTPUT}" \
  --reuse-candidate-root "${PICKSINGLE_CANDIDATES}" \
  --persistent-worker \
  "${COMMON_ARGS[@]}"

echo "[2/2] PickClutterYCB-v1 seed1-200"
PYTHONPATH=. "${PYTHON_BIN}" \
  -m maniskill_curobo.scripts.run_zerograsp_depth_ab_batch \
  --env-id PickClutterYCB-v1 \
  --output-root "${PICKCLUTTER_OUTPUT}" \
  --camera-eye -0.2 0.0 0.27 \
  --camera-target 0.05 0.0 0.08 \
  "${COMMON_ARGS[@]}"
