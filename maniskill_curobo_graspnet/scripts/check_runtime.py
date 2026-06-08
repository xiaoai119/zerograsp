#!/usr/bin/env python3
"""Check whether GraspNet baseline runtime is ready."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from maniskill_curobo_graspnet.inference import validate_runtime_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default="maniskill_curobo_graspnet/external/graspnet-baseline",
    )
    parser.add_argument(
        "--checkpoint",
        default="maniskill_curobo_graspnet/checkpoints/checkpoint-rs.tar",
    )
    args = parser.parse_args()
    report = {
        "python": sys.version,
        "repo_root": str(Path(args.repo_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "modules": {
            name: bool(importlib.util.find_spec(name))
            for name in (
                "torch",
                "open3d",
                "graspnetAPI",
                "pointnet2",
                "knn_pytorch",
            )
        },
        "runtime_files_ready": False,
    }
    try:
        validate_runtime_files(
            Path(args.repo_root).resolve(),
            Path(args.checkpoint).resolve(),
        )
        report["runtime_files_ready"] = True
    except Exception as exc:
        report["runtime_file_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["runtime_files_ready"] and all(report["modules"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
