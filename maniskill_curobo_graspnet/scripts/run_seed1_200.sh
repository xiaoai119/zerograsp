#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GRASPNET_PYTHON="${GRASPNET_PYTHON:-${ROOT}/maniskill_curobo_graspnet/envs/graspnet/bin/python}"
MANISKILL_PYTHON="${MANISKILL_PYTHON:-${ROOT}/maniskill_curobo/envs/maniskill_curobo/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/maniskill_curobo/runs/depth_corrected_settle20_seed1_200_rerun}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/maniskill_curobo_graspnet/runs/seed1_200}"
ZERO_SUMMARY="${ZERO_SUMMARY:-${SOURCE_ROOT}/summary.json}"
CUROBO_VIDEO="${CUROBO_VIDEO:-0}"

PYTHONPATH=. "${GRASPNET_PYTHON}" -m \
  maniskill_curobo_graspnet.scripts.run_graspnet_batch_inference \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --seed-start 1 \
  --seed-end 200 \
  --reuse-existing

curobo_args=(
  -m maniskill_curobo_graspnet.scripts.run_curobo_batch
  --output-root "${OUTPUT_ROOT}"
  --seed-start 1
  --seed-end 200
  --reuse-existing
)
if [[ "${CUROBO_VIDEO}" != "1" ]]; then
  curobo_args+=(--no-video)
fi

PYTHONPATH=. "${MANISKILL_PYTHON}" "${curobo_args[@]}"

PYTHONPATH=. "${MANISKILL_PYTHON}" -m \
  maniskill_curobo_graspnet.scripts.summarize_comparison \
  --graspnet-root "${OUTPUT_ROOT}" \
  --zerograsp-summary "${ZERO_SUMMARY}" \
  --output-dir "${OUTPUT_ROOT}/comparison" \
  --seed-start 1 \
  --seed-end 200
