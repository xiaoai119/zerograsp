#!/usr/bin/env python3
"""Validate GraspNet point-cloud inputs before model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from maniskill_curobo_graspnet.graspnet_adapter import load_point_cloud_input


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    records = []
    for seed in range(args.seed_start, args.seed_end + 1):
        input_dir = source_root / f"seed{seed:03d}" / "setup" / "zg_input"
        try:
            prepared = load_point_cloud_input(input_dir)
            record = {
                "seed": seed,
                "status": "ready",
                "target_point_count": prepared.target_point_count,
                "context_point_count": prepared.context_point_count,
                "target_label": prepared.target_label,
                "target_limits": prepared.target_limits,
                "depth_scale_to_m": prepared.depth_scale_to_m,
            }
        except Exception as exc:
            record = {
                "seed": seed,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        print(f"[{len(records)}/{args.seed_end - args.seed_start + 1}] seed={seed} {record['status']}")

    ready = [record for record in records if record["status"] == "ready"]
    target_counts = np.asarray(
        [record["target_point_count"] for record in ready], dtype=np.int64
    )
    context_counts = np.asarray(
        [record["context_point_count"] for record in ready], dtype=np.int64
    )
    report = {
        "source_root": str(source_root),
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "ready": len(ready),
        "failed": len(records) - len(ready),
        "target_point_count": summarize_counts(target_counts),
        "context_point_count": summarize_counts(context_counts),
        "records": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if len(ready) == len(records) else 1


def summarize_counts(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {}
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "max": int(values.max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
