#!/usr/bin/env python3
"""Compare M4 collision-world stages against the completed M3 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


STAGES = ("m4a", "m4b", "m4c")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m3-records",
        nargs="+",
        required=True,
        help="One or more M3 records.jsonl files.",
    )
    parser.add_argument("--m4-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--requested-seeds", type=int, default=200)
    return parser.parse_args(argv)


def load_records(paths: list[str]) -> dict[str, dict[int, dict[str, Any]]]:
    records: dict[str, dict[int, dict[str, Any]]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records.setdefault(str(record["stage"]), {})[int(record["seed"])] = record
    return records


def runtime(record: dict[str, Any]) -> float | None:
    command = record.get("command") or {}
    value = command.get("runtime_sec")
    return float(value) if value is not None else None


def scene_build_runtime(record: dict[str, Any]) -> float | None:
    scene_build = record.get("scene_build") or {}
    value = scene_build.get("runtime_sec")
    return float(value) if value is not None else None


def end_to_end_runtime(record: dict[str, Any]) -> float | None:
    value = record.get("total_runtime_sec")
    if value is not None:
        return float(value)
    command_value = runtime(record)
    scene_value = scene_build_runtime(record)
    if command_value is None:
        return scene_value
    return command_value + (scene_value or 0.0)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def runtime_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean_sec": statistics.fmean(values) if values else None,
        "median_sec": statistics.median(values) if values else None,
        "p95_sec": percentile(values, 0.95),
    }


def outcome_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        outcome = str(record.get("outcome") or "unknown")
        counts[outcome] = counts.get(outcome, 0) + 1
    return dict(sorted(counts.items()))


def stage_summary(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    requested_seeds: int,
) -> dict[str, Any]:
    complete = {
        seed: record
        for seed, record in candidate.items()
        if record.get("status") == "complete"
    }
    matched_seeds = sorted(set(baseline) & set(complete))
    baseline_matched = [baseline[seed] for seed in matched_seeds]
    candidate_matched = [complete[seed] for seed in matched_seeds]
    baseline_success = sum(bool(record.get("object_lift_success")) for record in baseline_matched)
    candidate_success = sum(bool(record.get("object_lift_success")) for record in candidate_matched)
    improved = [
        seed
        for seed in matched_seeds
        if not baseline[seed].get("object_lift_success")
        and complete[seed].get("object_lift_success")
    ]
    degraded = [
        seed
        for seed in matched_seeds
        if baseline[seed].get("object_lift_success")
        and not complete[seed].get("object_lift_success")
    ]
    baseline_runtimes = [
        value for value in (runtime(baseline[seed]) for seed in matched_seeds) if value is not None
    ]
    candidate_runtimes = [
        value for value in (runtime(complete[seed]) for seed in matched_seeds) if value is not None
    ]
    baseline_total_runtimes = [
        value
        for value in (end_to_end_runtime(baseline[seed]) for seed in matched_seeds)
        if value is not None
    ]
    candidate_total_runtimes = [
        value
        for value in (end_to_end_runtime(complete[seed]) for seed in matched_seeds)
        if value is not None
    ]
    scene_build_runtimes = [
        value for value in (scene_build_runtime(complete[seed]) for seed in matched_seeds)
        if value is not None
    ]
    baseline_mean = statistics.fmean(baseline_runtimes) if baseline_runtimes else None
    candidate_mean = statistics.fmean(candidate_runtimes) if candidate_runtimes else None
    baseline_total_mean = (
        statistics.fmean(baseline_total_runtimes) if baseline_total_runtimes else None
    )
    candidate_total_mean = (
        statistics.fmean(candidate_total_runtimes) if candidate_total_runtimes else None
    )
    matched_count = len(matched_seeds)
    delta_pp = (
        100.0 * (candidate_success - baseline_success) / matched_count
        if matched_count
        else None
    )
    return {
        "processed": len(candidate),
        "complete": len(complete),
        "successes": sum(bool(record.get("object_lift_success")) for record in complete.values()),
        "success_rate_complete": (
            sum(bool(record.get("object_lift_success")) for record in complete.values())
            / len(complete)
            if complete
            else None
        ),
        "success_rate_requested": (
            sum(bool(record.get("object_lift_success")) for record in complete.values())
            / requested_seeds
        ),
        "outcomes": outcome_counts(list(complete.values())),
        "matched_seeds": matched_count,
        "m3_successes_on_matched": baseline_success,
        "stage_successes_on_matched": candidate_success,
        "success_delta_percentage_points": delta_pp,
        "improved_seeds": improved,
        "degraded_seeds": degraded,
        "net_success_change": len(improved) - len(degraded),
        "runtime": runtime_stats(candidate_runtimes),
        "scene_build_runtime": runtime_stats(scene_build_runtimes),
        "end_to_end_runtime": runtime_stats(candidate_total_runtimes),
        "m3_runtime_on_matched": runtime_stats(baseline_runtimes),
        "m3_end_to_end_runtime_on_matched": runtime_stats(baseline_total_runtimes),
        "mean_runtime_multiplier_vs_m3": (
            candidate_mean / baseline_mean
            if candidate_mean is not None and baseline_mean not in {None, 0.0}
            else None
        ),
        "mean_end_to_end_runtime_multiplier_vs_m3": (
            candidate_total_mean / baseline_total_mean
            if candidate_total_mean is not None
            and baseline_total_mean not in {None, 0.0}
            else None
        ),
    }


def write_outputs(
    *,
    output_dir: Path,
    baseline: dict[int, dict[str, Any]],
    stages: dict[str, dict[int, dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m3_vs_m4_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = ["seed", "m3_outcome", "m3_success"]
    for stage in STAGES:
        fields.extend(
            [
                f"{stage}_outcome",
                f"{stage}_success",
                f"{stage}_runtime_sec",
                f"{stage}_end_to_end_runtime_sec",
            ]
        )
    with (output_dir / "m3_vs_m4_per_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        all_seeds = sorted(set(baseline) | set().union(*(set(value) for value in stages.values())))
        for seed in all_seeds:
            row: dict[str, Any] = {"seed": seed}
            base = baseline.get(seed, {})
            row["m3_outcome"] = base.get("outcome", "")
            row["m3_success"] = bool(base.get("object_lift_success")) if base else ""
            for stage in STAGES:
                record = stages.get(stage, {}).get(seed, {})
                row[f"{stage}_outcome"] = record.get("outcome", "")
                row[f"{stage}_success"] = (
                    bool(record.get("object_lift_success")) if record else ""
                )
                row[f"{stage}_runtime_sec"] = runtime(record) if record else ""
                row[f"{stage}_end_to_end_runtime_sec"] = (
                    end_to_end_runtime(record) if record else ""
                )
            writer.writerow(row)

    lines = [
        "# M3 vs M4 Collision World Benchmark",
        "",
        "| Stage | Complete | Lift success | Matched M3 | Delta | Improved | Degraded | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        data = summary["stages"].get(stage, {})
        delta = data.get("success_delta_percentage_points")
        multiplier = data.get("mean_end_to_end_runtime_multiplier_vs_m3")
        lines.append(
            f"| `{stage}` | {data.get('complete', 0)} | {data.get('successes', 0)} | "
            f"{data.get('matched_seeds', 0)} | "
            f"{delta:+.2f} pp | {len(data.get('improved_seeds', []))} | "
            f"{len(data.get('degraded_seeds', []))} | {multiplier:.2f}x |"
            if delta is not None and multiplier is not None
            else f"| `{stage}` | {data.get('complete', 0)} | {data.get('successes', 0)} | "
            f"{data.get('matched_seeds', 0)} | incomplete | - | - | - |"
        )
    lines.append("")
    (output_dir / "m3_vs_m4_comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_records = load_records(args.m3_records)
    m4_records = load_records([args.m4_records])
    baseline = {
        seed: record
        for seed, record in baseline_records.get("m3", {}).items()
        if record.get("status") == "complete"
    }
    stages = {stage: m4_records.get(stage, {}) for stage in STAGES}
    summary = {
        "requested_seeds": int(args.requested_seeds),
        "m3_complete": len(baseline),
        "m3_successes": sum(
            bool(record.get("object_lift_success")) for record in baseline.values()
        ),
        "stages": {
            stage: stage_summary(
                baseline,
                stages[stage],
                int(args.requested_seeds),
            )
            for stage in STAGES
        },
    }
    write_outputs(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        baseline=baseline,
        stages=stages,
        summary=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
