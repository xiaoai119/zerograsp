#!/usr/bin/env python3
"""Compare ManiSkill-depth and ZeroGrasp-reconstruction collision worlds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


STAGES = ("m3", "m4a", "m4b", "m4c")
SOURCES = ("maniskill_depth", "zerograsp_reconstruction")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maniskill-m3-records", required=True)
    parser.add_argument("--maniskill-m4-records", required=True)
    parser.add_argument("--zerograsp-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--requested-seeds", type=int, default=200)
    return parser.parse_args(argv)


def load_records(paths: list[str]) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "complete":
                continue
            result.setdefault(str(record["stage"]), {})[int(record["seed"])] = record
    return result


def runtime(record: dict[str, Any]) -> float | None:
    value = record.get("total_runtime_sec")
    if value is not None:
        return float(value)
    command = record.get("command") or {}
    value = command.get("runtime_sec")
    return float(value) if value is not None else None


def outcome_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        outcome = str(record.get("outcome") or "unknown")
        counts[outcome] = counts.get(outcome, 0) + 1
    return dict(sorted(counts.items()))


def source_stats(
    records: dict[int, dict[str, Any]],
    requested_seeds: int,
) -> dict[str, Any]:
    successes = sum(bool(record.get("object_lift_success")) for record in records.values())
    runtimes = [
        value for value in (runtime(record) for record in records.values()) if value is not None
    ]
    return {
        "complete": len(records),
        "successes": successes,
        "success_rate_complete": successes / len(records) if records else None,
        "success_rate_requested": successes / requested_seeds,
        "outcomes": outcome_counts(records.values()),
        "mean_end_to_end_runtime_sec": statistics.fmean(runtimes) if runtimes else None,
        "median_end_to_end_runtime_sec": statistics.median(runtimes) if runtimes else None,
    }


def compare_stage(
    maniskill: dict[int, dict[str, Any]],
    zerograsp: dict[int, dict[str, Any]],
    requested_seeds: int,
) -> dict[str, Any]:
    matched_seeds = sorted(set(maniskill) & set(zerograsp))
    maniskill_successes = sum(
        bool(maniskill[seed].get("object_lift_success")) for seed in matched_seeds
    )
    zerograsp_successes = sum(
        bool(zerograsp[seed].get("object_lift_success")) for seed in matched_seeds
    )
    improved = [
        seed
        for seed in matched_seeds
        if not maniskill[seed].get("object_lift_success")
        and zerograsp[seed].get("object_lift_success")
    ]
    degraded = [
        seed
        for seed in matched_seeds
        if maniskill[seed].get("object_lift_success")
        and not zerograsp[seed].get("object_lift_success")
    ]
    unchanged_success = [
        seed
        for seed in matched_seeds
        if maniskill[seed].get("object_lift_success")
        and zerograsp[seed].get("object_lift_success")
    ]
    unchanged_failure = [
        seed
        for seed in matched_seeds
        if not maniskill[seed].get("object_lift_success")
        and not zerograsp[seed].get("object_lift_success")
    ]
    matched_count = len(matched_seeds)
    return {
        "maniskill_depth": source_stats(maniskill, requested_seeds),
        "zerograsp_reconstruction": source_stats(zerograsp, requested_seeds),
        "matched_seeds": matched_count,
        "maniskill_successes_on_matched": maniskill_successes,
        "zerograsp_successes_on_matched": zerograsp_successes,
        "zerograsp_delta_percentage_points": (
            100.0 * (zerograsp_successes - maniskill_successes) / matched_count
            if matched_count
            else None
        ),
        "net_success_change": len(improved) - len(degraded),
        "improved_seeds": improved,
        "degraded_seeds": degraded,
        "unchanged_success_seeds": unchanged_success,
        "unchanged_failure_seeds": unchanged_failure,
    }


def write_outputs(
    output_dir: Path,
    records: dict[str, dict[str, dict[int, dict[str, Any]]]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "maniskill_vs_zerograsp_pointcloud.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = ["seed"]
    for stage in STAGES:
        for source in SOURCES:
            fields.extend(
                [
                    f"{stage}_{source}_outcome",
                    f"{stage}_{source}_success",
                    f"{stage}_{source}_runtime_sec",
                ]
            )
    with (output_dir / "maniskill_vs_zerograsp_pointcloud.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        all_seeds = sorted(
            set().union(
                *(
                    set(records[source].get(stage, {}))
                    for source in SOURCES
                    for stage in STAGES
                )
            )
        )
        for seed in all_seeds:
            row: dict[str, Any] = {"seed": seed}
            for stage in STAGES:
                for source in SOURCES:
                    record = records[source].get(stage, {}).get(seed)
                    prefix = f"{stage}_{source}"
                    row[f"{prefix}_outcome"] = record.get("outcome", "") if record else ""
                    row[f"{prefix}_success"] = (
                        bool(record.get("object_lift_success")) if record else ""
                    )
                    row[f"{prefix}_runtime_sec"] = runtime(record) if record else ""
            writer.writerow(row)

    lines = [
        "# ManiSkill Depth vs ZeroGrasp Reconstruction",
        "",
        "The grasp candidates and execution protocol are shared. Only the point-cloud",
        "source used to build the cuRobo collision world changes.",
        "",
        "| Stage | ManiSkill depth | ZeroGrasp reconstruction | Delta | Improved | Degraded |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        data = summary["stages"][stage]
        maniskill = data["maniskill_depth"]
        zerograsp = data["zerograsp_reconstruction"]
        delta = data["zerograsp_delta_percentage_points"]
        lines.append(
            f"| `{stage}` | {maniskill['successes']}/{maniskill['complete']} "
            f"({100.0 * maniskill['success_rate_complete']:.2f}%) | "
            f"{zerograsp['successes']}/{zerograsp['complete']} "
            f"({100.0 * zerograsp['success_rate_complete']:.2f}%) | "
            f"{delta:+.2f} pp | {len(data['improved_seeds'])} | "
            f"{len(data['degraded_seeds'])} |"
        )
    lines.extend(["", "## Changed Seeds", ""])
    for stage in STAGES:
        data = summary["stages"][stage]
        lines.extend(
            [
                f"### {stage.upper()}",
                "",
                f"- ManiSkill fail -> ZeroGrasp success: {data['improved_seeds']}",
                f"- ManiSkill success -> ZeroGrasp fail: {data['degraded_seeds']}",
                "",
            ]
        )
    (output_dir / "maniskill_vs_zerograsp_pointcloud.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    maniskill = load_records(
        [args.maniskill_m3_records, args.maniskill_m4_records]
    )
    zerograsp = load_records([args.zerograsp_records])
    records = {
        "maniskill_depth": maniskill,
        "zerograsp_reconstruction": zerograsp,
    }
    summary = {
        "requested_seeds": int(args.requested_seeds),
        "stages": {
            stage: compare_stage(
                maniskill.get(stage, {}),
                zerograsp.get(stage, {}),
                int(args.requested_seeds),
            )
            for stage in STAGES
        },
    }
    write_outputs(
        Path(args.output_dir).expanduser().resolve(),
        records,
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
