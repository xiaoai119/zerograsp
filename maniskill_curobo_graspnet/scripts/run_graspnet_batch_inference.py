#!/usr/bin/env python3
"""Load GraspNet once and infer candidates for a ManiSkill seed range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterable

from maniskill_curobo_graspnet.inference import GraspNetBaselineRuntime


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument(
        "--repo-root",
        default="maniskill_curobo_graspnet/external/graspnet-baseline",
    )
    parser.add_argument(
        "--checkpoint",
        default="maniskill_curobo_graspnet/checkpoints/checkpoint-rs.tar",
    )
    parser.add_argument("--depth-unit", choices=("auto", "m", "mm"), default="auto")
    parser.add_argument("--target-margin-m", type=float, default=0.02)
    parser.add_argument("--context-margin-m", type=float, default=0.08)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--no-filter-to-target", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed_start > args.seed_end:
        raise ValueError("--seed-start must not exceed --seed-end")
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = GraspNetBaselineRuntime(
        repo_root=args.repo_root,
        checkpoint_path=args.checkpoint,
        num_point=args.num_point,
        num_view=args.num_view,
        collision_thresh=args.collision_thresh,
        voxel_size=args.voxel_size,
    )
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(range(args.seed_start, args.seed_end + 1), 1):
        input_dir = source_root / f"seed{seed:03d}" / "setup" / "zg_input"
        output_dir = output_root / f"seed{seed:03d}" / "graspnet_output"
        candidate_path = output_dir / "recommended_grasp_top1.json"
        started = time.time()
        try:
            if args.reuse_existing and candidate_path.is_file():
                report = json.loads(
                    (output_dir / "run_report.json").read_text(encoding="utf-8")
                )
                status = "reused"
            else:
                report = runtime.predict(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    depth_unit=args.depth_unit,
                    target_margin_m=args.target_margin_m,
                    context_margin_m=args.context_margin_m,
                    filter_to_target=not args.no_filter_to_target,
                    random_seed=seed,
                )
                status = (
                    "success"
                    if report["recommended_grasp"] is not None
                    else "no_grasp"
                )
            record = {
                "seed": seed,
                "status": status,
                "runtime_sec": float(time.time() - started),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "report": report,
            }
        except Exception as exc:
            record = {
                "seed": seed,
                "status": "failed",
                "runtime_sec": float(time.time() - started),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        write_progress(output_root, args, records)
        print(
            f"[{index}/{args.seed_end - args.seed_start + 1}] "
            f"seed={seed} status={record['status']}",
            flush=True,
        )
    write_progress(output_root, args, records)
    return 0 if all(record["status"] != "failed" for record in records) else 1


def write_progress(
    output_root: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> None:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    payload = {
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "processed": len(records),
        "counts": counts,
        "records": records,
    }
    temp = output_root / "inference_summary.json.tmp"
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(output_root / "inference_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
