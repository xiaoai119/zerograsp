#!/usr/bin/env python3
"""Compare GraspNet and ZeroGrasp lift outcomes seed by seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graspnet-root", required=True)
    parser.add_argument("--zerograsp-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zero-variant", default="depth")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    graspnet_root = Path(args.graspnet_root).expanduser().resolve()
    graspnet_summary = json.loads(
        (graspnet_root / "execution_summary.json").read_text(encoding="utf-8")
    )
    zero_summary = json.loads(
        Path(args.zerograsp_summary).expanduser().resolve().read_text(encoding="utf-8")
    )
    graspnet_records = {
        int(row["seed"]): row for row in graspnet_summary["records"]
    }
    zero_records = {int(row["seed"]): row for row in zero_summary["records"]}
    rows = []
    for seed in range(args.seed_start, args.seed_end + 1):
        graspnet_record = graspnet_records.get(seed, {})
        zero_record = zero_records.get(seed, {})
        graspnet_execution = graspnet_record.get("execution") or {}
        zero_execution = (zero_record.get("variants") or {}).get(args.zero_variant) or {}
        graspnet_success = bool(graspnet_execution.get("object_lift_success"))
        zero_success = bool(zero_execution.get("object_lift_success"))
        if graspnet_success and not zero_success:
            change = "graspnet_only"
        elif zero_success and not graspnet_success:
            change = "zerograsp_only"
        elif graspnet_success:
            change = "both_success"
        else:
            change = "both_failed"
        rows.append(
            {
                "seed": seed,
                "zerograsp_success": zero_success,
                "zerograsp_outcome": zero_execution.get("outcome", "missing"),
                "graspnet_success": graspnet_success,
                "graspnet_outcome": graspnet_execution.get(
                    "outcome", graspnet_record.get("status", "missing")
                ),
                "change": change,
                "zerograsp_video": zero_execution.get("video_path")
                or zero_execution.get("partial_video_saved"),
                "graspnet_video": graspnet_execution.get("video_path")
                or graspnet_execution.get("partial_video_saved"),
            }
        )
    counts = {
        "seeds": len(rows),
        "zerograsp_successes": sum(row["zerograsp_success"] for row in rows),
        "graspnet_successes": sum(row["graspnet_success"] for row in rows),
        "graspnet_only": sum(row["change"] == "graspnet_only" for row in rows),
        "zerograsp_only": sum(row["change"] == "zerograsp_only" for row in rows),
        "both_success": sum(row["change"] == "both_success" for row in rows),
        "both_failed": sum(row["change"] == "both_failed" for row in rows),
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_summary.json").write_text(
        json.dumps({"counts": counts, "records": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with (output / "comparison_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(output / "comparison_report.md", counts, rows)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


def write_markdown(
    path: Path,
    counts: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# GraspNet vs ZeroGrasp, PickSingleYCB seed1-200",
        "",
        "Both methods use the same settled RGB-D observations and the same "
        "ManiSkill/cuRobo execution settings.",
        "",
        f"- ZeroGrasp lift success: {counts['zerograsp_successes']}/{counts['seeds']}",
        f"- GraspNet lift success: {counts['graspnet_successes']}/{counts['seeds']}",
        f"- GraspNet only: {counts['graspnet_only']}",
        f"- ZeroGrasp only: {counts['zerograsp_only']}",
        f"- Both success: {counts['both_success']}",
        f"- Both failed: {counts['both_failed']}",
        "",
        "| Seed | ZeroGrasp | GraspNet | Pairwise result |",
        "|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['zerograsp_outcome']} | "
            f"{row['graspnet_outcome']} | {row['change']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
