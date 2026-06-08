#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MANISKILL_PYTHON="${MANISKILL_PYTHON:-maniskill_curobo/envs/maniskill_curobo/bin/python}"
UTILITY_PYTHON="${UTILITY_PYTHON:-/home/openclaw-server/miniconda3/envs/graduate/bin/python}"
RECORDS="${RECORDS:-maniskill_collect_data/zerograsp_sparse_sft_plan/runs/planning_gate/phase19_final_test_seed401_500/gate_records.jsonl}"
RUN_ROOT="${RUN_ROOT:-maniskill_curobo/runs/zerograsp_depth_ab_seed401_406_425_430_405_420}"
DESKTOP_ROOT="${DESKTOP_ROOT:-/home/openclaw-server/Desktop/maniskill_curobo_depth_ab}"
SEEDS=(401 406 425 430 405 420)

candidate_for_seed() {
  "${UTILITY_PYTHON}" - "${RECORDS}" "$1" <<'PY'
import json
import sys
from pathlib import Path

records_path = Path(sys.argv[1])
seed = int(sys.argv[2])
for line in records_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    if int(record["seed"]) == seed:
        print(record["selection"]["candidate_json"])
        raise SystemExit(0)
raise SystemExit(f"Seed {seed} is missing from {records_path}")
PY
}

run_variant() {
  local seed="$1"
  local label="$2"
  local depth_scale="$3"
  local candidate_json="$4"
  local output_dir="${RUN_ROOT}/seed${seed}/${label}"

  rm -rf "${output_dir}"
  mkdir -p "${output_dir}/zg_output"
  cp "${candidate_json}" "${output_dir}/zg_output/recommended_grasp_top1.json"

  set +e
  PYTHONPATH=. "${MANISKILL_PYTHON}" \
    -m maniskill_curobo.scripts.execute_curobo_pick \
    --env-id PickSingleYCB-v1 \
    --zerograsp-output "${output_dir}/zg_output" \
    --seed "${seed}" \
    --camera base_camera \
    --width 1280 \
    --height 1024 \
    --mask-mode task-target \
    --camera-eye -0.3 0.0 0.55 \
    --camera-target 0.05 0.0 0.08 \
    --render-width 1280 \
    --render-height 1024 \
    --pregrasp-offset 0.1 \
    --lift-offset 0.15 \
    --workspace-z-min 0.01 \
    --grasp-depth-scale "${depth_scale}" \
    --grasp-depth-max-offset 0.04 \
    --close-steps 20 \
    --settle-steps 50 \
    --settle-before-export-steps 0 \
    --action-repeat 2 \
    --max-waypoints-per-stage 80 \
    --robot-config franka.yml \
    --scene-source maniskill \
    --scene-min-cuboid-dimension 0.005 \
    --scene-model collision_test.yml \
    --warmup-iterations 2 \
    --video-fps 20 \
    --video-out "${output_dir}/execution.mp4" \
    --output-dir "${output_dir}" \
    --approach-axis positive-x \
    --no-grasp-marker \
    >"${output_dir}/stdout.log" \
    2>"${output_dir}/stderr.log"
  local exit_code=$?
  set -e
  echo "${exit_code}" >"${output_dir}/exit_code.txt"
}

if [[ "${ORGANIZE_ONLY:-false}" != "true" ]]; then
  rm -rf "${RUN_ROOT}"
  mkdir -p "${RUN_ROOT}"
  for seed in "${SEEDS[@]}"; do
    candidate_json="$(candidate_for_seed "${seed}")"
    echo "[seed ${seed}] translation-only baseline"
    run_variant "${seed}" baseline 0.0 "${candidate_json}"
    echo "[seed ${seed}] full predicted depth"
    run_variant "${seed}" depth 1.0 "${candidate_json}"
  done
fi

rm -rf "${DESKTOP_ROOT}"
mkdir -p "${DESKTOP_ROOT}"

"${UTILITY_PYTHON}" - "${RUN_ROOT}" "${DESKTOP_ROOT}" "${SEEDS[@]}" <<'PY'
import json
import shutil
import subprocess
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
desktop_root = Path(sys.argv[2])
seeds = [int(value) for value in sys.argv[3:]]


def outcome(manifest):
    if not manifest:
        return "missing_manifest"
    execution_failure = manifest.get("failure_reason")
    if execution_failure:
        if "stage=pre" in execution_failure:
            return "planning_failed_pre"
        if "stage=grasp" in execution_failure:
            return "planning_failed_grasp"
        if "stage=lift" in execution_failure:
            return "planning_failed_lift"
        return "execution_failed"
    metrics = manifest.get("object_lift_metrics") or {}
    if metrics.get("object_lift_success"):
        return "success"
    return metrics.get("failure_reason") or "failed"


rows = []
for seed in seeds:
    seed_out = desktop_root / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    row = {"seed": seed}
    videos = {}
    for label in ("baseline", "depth"):
        run_dir = run_root / f"seed{seed}" / label
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        grasp = manifest.get("grasp") or {}
        row[label] = {
            "outcome": outcome(manifest),
            "object_lift_success": bool((manifest.get("object_lift_metrics") or {}).get("object_lift_success")),
            "execution_completed": not bool(manifest.get("failure_reason")),
            "depth_m": grasp.get("depth_m"),
            "depth_offset": grasp.get("grasp_depth_offset"),
            "position_base": grasp.get("position_base"),
            "planner_position_base": grasp.get("planner_position_base"),
            "manifest": str(manifest_path.resolve()),
        }
        video = run_dir / "execution.mp4"
        if video.exists():
            copied = seed_out / f"seed{seed}_{label}.mp4"
            shutil.copy2(video, copied)
            videos[label] = copied

    if set(videos) == {"baseline", "depth"}:
        side_by_side = seed_out / f"seed{seed}_side_by_side.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(videos["baseline"]),
                "-i",
                str(videos["depth"]),
                "-filter_complex",
                (
                    "[0:v]scale=640:-2,tpad=stop_mode=clone:stop_duration=120,"
                    "drawtext=text='translation only':x=20:y=20:fontsize=28:"
                    "fontcolor=white:box=1:boxcolor=black@0.6[left];"
                    "[1:v]scale=640:-2,tpad=stop_mode=clone:stop_duration=120,"
                    "drawtext=text='+ ZeroGrasp depth':x=20:y=20:fontsize=28:"
                    "fontcolor=white:box=1:boxcolor=black@0.6[right];"
                    "[left][right]hstack=inputs=2:shortest=1[out]"
                ),
                "-map",
                "[out]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                str(side_by_side),
            ],
            check=True,
        )
        row["side_by_side_video"] = str(side_by_side.resolve())
        shutil.copy2(side_by_side, desktop_root / side_by_side.name)
    rows.append(row)

summary = {
    "description": "Same saved ZeroGrasp candidate, translation-only versus full predicted depth offset.",
    "rows": rows,
    "baseline_successes": sum(row["baseline"]["object_lift_success"] for row in rows),
    "depth_successes": sum(row["depth"]["object_lift_success"] for row in rows),
    "baseline_execution_failures": sum(not row["baseline"]["execution_completed"] for row in rows),
    "depth_execution_failures": sum(not row["depth"]["execution_completed"] for row in rows),
}
(run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(desktop_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
