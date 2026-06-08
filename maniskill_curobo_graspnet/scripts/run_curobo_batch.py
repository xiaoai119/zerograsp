#!/usr/bin/env python3
"""Execute saved GraspNet candidates with the existing ManiSkill/cuRobo stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from maniskill_curobo.scripts.run_zerograsp_depth_ab_batch import (
    DEFAULT_CAMERA_EYE,
    DEFAULT_CAMERA_TARGET,
    PersistentExecutionRunner,
    execute_command,
    run_command,
    summarize_run,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument("--env-id", default="PickSingleYCB-v1")
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument(
        "--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET)
    )
    parser.add_argument("--mask-mode", default="task-target")
    parser.add_argument("--approach-axis", default="positive-x")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--grasp-depth-max-offset", type=float, default=0.04)
    parser.add_argument("--depth-auto-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--workspace-z-min", type=float, default=0.01)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--settle-before-export-steps", type=int, default=20)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-waypoints-per-stage", type=int, default=80)
    parser.add_argument("--robot-config", default="franka.yml")
    parser.add_argument("--scene-source", default="maniskill")
    parser.add_argument("--scene-min-cuboid-dimension", type=float, default=0.005)
    parser.add_argument("--scene-model", default="collision_test.yml")
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--maniskill-python", default="maniskill_curobo/envs/maniskill_curobo/bin/python")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-persistent-worker", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed_start > args.seed_end:
        raise ValueError("--seed-start must not exceed --seed-end")
    if not args.no_persistent_worker and args.env_id != "PickSingleYCB-v1":
        raise ValueError("Persistent execution currently supports PickSingleYCB-v1 only.")

    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root).expanduser().resolve()
    records = load_existing(output_root) if args.reuse_existing else {}
    runner = None
    if not args.no_persistent_worker:
        runner = PersistentExecutionRunner(repo_root=repo_root)
    try:
        for index, seed in enumerate(range(args.seed_start, args.seed_end + 1), 1):
            seed_dir = output_root / f"seed{seed:03d}"
            candidate_dir = seed_dir / "graspnet_output"
            run_dir = seed_dir / "curobo"
            candidate = candidate_dir / "recommended_grasp_top1.json"
            if not candidate.is_file():
                record = {
                    "seed": seed,
                    "status": "missing_graspnet_candidate",
                    "candidate_dir": str(candidate_dir),
                }
            elif (
                args.reuse_existing
                and seed in records
                and (run_dir / "run_manifest.json").is_file()
            ):
                record = {
                    "seed": seed,
                    "status": "complete",
                    "candidate_dir": str(candidate_dir),
                    "execution": summarize_run(run_dir, reused=True),
                }
            else:
                command = execute_command(
                    args,
                    seed=seed,
                    zg_output_dir=candidate_dir,
                    run_dir=run_dir,
                    depth_scale=float(args.depth_scale),
                    depth_auto_fallback=bool(args.depth_auto_fallback),
                )
                if runner is None:
                    command_result = run_command(
                        command,
                        cwd=repo_root,
                        logs_dir=run_dir / "logs",
                        name="execute",
                    )
                else:
                    command_result = runner.run(
                        command,
                        logs_dir=run_dir / "logs",
                        name="execute",
                    )
                record = {
                    "seed": seed,
                    "status": "complete",
                    "candidate_dir": str(candidate_dir),
                    "execution": summarize_run(run_dir, command=command_result),
                }
            records[seed] = record
            write_results(output_root, args, records)
            print(
                f"[{index}/{args.seed_end - args.seed_start + 1}] "
                f"seed={seed} status={record['status']} "
                f"outcome={(record.get('execution') or {}).get('outcome')}",
                flush=True,
            )
    finally:
        if runner is not None:
            runner.close()
    write_results(output_root, args, records)
    return 0


def load_existing(output_root: Path) -> dict[int, dict[str, Any]]:
    path = output_root / "execution_records.jsonl"
    if not path.is_file():
        return {}
    return {
        int(record["seed"]): record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }


def write_results(
    output_root: Path,
    args: argparse.Namespace,
    records: dict[int, dict[str, Any]],
) -> None:
    ordered = [records[seed] for seed in sorted(records)]
    complete = [record for record in ordered if record["status"] == "complete"]
    counts: dict[str, Any] = {
        "processed": len(ordered),
        "complete": len(complete),
        "missing_candidates": sum(
            record["status"] == "missing_graspnet_candidate" for record in ordered
        ),
        "lift_successes": sum(
            bool((record.get("execution") or {}).get("object_lift_success"))
            for record in complete
        ),
        "outcomes": {},
    }
    for record in complete:
        outcome = str(record["execution"]["outcome"])
        counts["outcomes"][outcome] = counts["outcomes"].get(outcome, 0) + 1
    payload = {
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "settings": vars(args),
        "counts": counts,
        "records": ordered,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temp = output_root / "execution_summary.json.tmp"
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(output_root / "execution_summary.json")
    records_temp = output_root / "execution_records.jsonl.tmp"
    with records_temp.open("w", encoding="utf-8") as file:
        for record in ordered:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    records_temp.replace(output_root / "execution_records.jsonl")


if __name__ == "__main__":
    raise SystemExit(main())
