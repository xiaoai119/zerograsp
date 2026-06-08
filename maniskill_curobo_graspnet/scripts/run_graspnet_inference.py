#!/usr/bin/env python3
"""Run GraspNet baseline on one exported ManiSkill RGB-D bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from maniskill_curobo_graspnet.inference import GraspNetBaselineRuntime


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = GraspNetBaselineRuntime(
        repo_root=args.repo_root,
        checkpoint_path=args.checkpoint,
        num_point=args.num_point,
        num_view=args.num_view,
        collision_thresh=args.collision_thresh,
        voxel_size=args.voxel_size,
    )
    report = runtime.predict(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        depth_unit=args.depth_unit,
        target_margin_m=args.target_margin_m,
        context_margin_m=args.context_margin_m,
        filter_to_target=not args.no_filter_to_target,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["recommended_grasp"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
