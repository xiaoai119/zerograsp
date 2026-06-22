#!/usr/bin/env python3
"""Summarize M0/M4C/M5 instance reconstruction benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable

from maniskill_curobo_real.run_world_collision_stages import summarize_run


DEFAULT_M0_ROOT = Path("maniskill_curobo_real/runs/pickclutter_full_depth_m0_seed1_200")
DEFAULT_M4C_ROOT = Path("maniskill_curobo_real/runs/pickclutter_full_depth_m4abc_seed1_200")
DEFAULT_M5_ROOT = Path("maniskill_curobo_real/runs/pickclutter_m5_instance_esdf_no_table_seed1_200")

STAGES = {
    "m0": {
        "label": "M0 ManiSkill truth",
        "root": DEFAULT_M0_ROOT,
        "stage_dir": "m0_maniskill_truth",
        "avg_runtime_sec": 6.40,
    },
    "m4c": {
        "label": "M4C oracle single-view voxel ESDF",
        "root": DEFAULT_M4C_ROOT,
        "stage_dir": "m4c_oracle_instance_voxel_esdf",
        "avg_runtime_sec": 7.75,
    },
    "m5": {
        "label": "M5 multiview instance voxel ESDF no-table",
        "root": DEFAULT_M5_ROOT,
        "stage_dir": "m5_multiview_instance_voxel_esdf_no_table",
        "avg_runtime_sec": None,
    },
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0_ROOT)
    parser.add_argument("--m4c-root", type=Path, default=DEFAULT_M4C_ROOT)
    parser.add_argument("--m5-root", type=Path, default=DEFAULT_M5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    roots = {
        "m0": args.m0_root.expanduser().resolve(),
        "m4c": args.m4c_root.expanduser().resolve(),
        "m5": args.m5_root.expanduser().resolve(),
    }
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else roots["m5"] / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    records = {key: load_stage_records(key, roots[key], seeds) for key in STAGES}
    rows = build_rows(seeds, records)
    metrics = build_metrics(records, seeds)

    csv_path = output_dir / "m0_m4c_m5_per_seed.csv"
    write_csv(csv_path, rows)
    json_path = output_dir / "m0_m4c_m5_summary.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "m0_m4c_m5_analysis.md"
    report_path.write_text(render_report(metrics, csv_path), encoding="utf-8")

    print(json.dumps({"report": str(report_path), "csv": str(csv_path), "summary": str(json_path)}, ensure_ascii=False, indent=2))
    return 0


def load_stage_records(stage_key: str, root: Path, seeds: list[int]) -> dict[int, dict]:
    stage_dir = root / STAGES[stage_key]["stage_dir"]
    summary_records = load_m5_summary_records(stage_key, root)
    out: dict[int, dict] = {}
    for seed in seeds:
        if seed in summary_records:
            out[seed] = normalize_record(summary_records[seed], stage_key, root)
            continue
        run_dir = stage_dir / f"seed{seed:03d}"
        if not (run_dir / "run_manifest.json").is_file():
            run_dir_alt = stage_dir / f"seed{seed}"
            run_dir = run_dir_alt if (run_dir_alt / "run_manifest.json").is_file() else run_dir
        if (run_dir / "run_manifest.json").is_file():
            out[seed] = normalize_record(
                summarize_run(stage=stage_key, seed=seed, run_dir=run_dir),
                stage_key,
                root,
            )
        else:
            out[seed] = {
                "stage": stage_key,
                "seed": int(seed),
                "status": "missing_zerograsp_candidate",
                "outcome": "missing_zerograsp_candidate",
                "object_lift_success": False,
                "failure_reason": "missing_zerograsp_candidate",
                "run_dir": str(run_dir),
            }
    return out


def load_m5_summary_records(stage_key: str, root: Path) -> dict[int, dict]:
    if stage_key != "m5":
        return {}
    path = root / "m5_multiview_instance_voxel_esdf_no_table_summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["seed"]): record for record in payload.get("records", [])}


def normalize_record(record: dict, stage_key: str, root: Path) -> dict:
    normalized = dict(record)
    normalized["stage"] = stage_key
    normalized["success"] = bool(record.get("object_lift_success"))
    normalized["outcome"] = str(record.get("outcome") or record.get("status") or "unknown")
    normalized["failure_reason"] = str(record.get("failure_reason") or normalized["outcome"])
    candidate_selection = record.get("candidate_selection") or {}
    normalized["selected_rank"] = candidate_selection.get("selected_rank")
    normalized["selected_depth_scale"] = candidate_selection.get("selected_depth_scale")
    if normalized["selected_depth_scale"] is None:
        normalized["selected_depth_scale"] = candidate_selection.get("requested_depth_scale")
    normalized["runtime_sec"] = record.get("total_runtime_sec")
    if normalized["runtime_sec"] is None:
        normalized["runtime_sec"] = record.get("end_to_end_runtime_sec")
    normalized["root"] = str(root)
    return normalized


def build_rows(seeds: list[int], records: dict[str, dict[int, dict]]) -> list[dict]:
    rows = []
    for seed in seeds:
        row = {"seed": seed}
        for key in ("m0", "m4c", "m5"):
            record = records[key][seed]
            row[f"{key}_success"] = bool(record.get("success"))
            row[f"{key}_outcome"] = record.get("outcome")
            row[f"{key}_failure_reason"] = record.get("failure_reason")
            row[f"{key}_selected_rank"] = record.get("selected_rank")
            row[f"{key}_selected_depth_scale"] = record.get("selected_depth_scale")
            row[f"{key}_runtime_sec"] = record.get("runtime_sec")
        row["m5_vs_m0"] = compare_success(row["m5_success"], row["m0_success"])
        row["m5_vs_m4c"] = compare_success(row["m5_success"], row["m4c_success"])
        rows.append(row)
    return rows


def compare_success(a: bool, b: bool) -> str:
    if a and not b:
        return "improved"
    if b and not a:
        return "regressed"
    if a and b:
        return "both_success"
    return "both_failed"


def build_metrics(records: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    metrics = {
        "seed_start": min(seeds),
        "seed_end": max(seeds),
        "n_requested": len(seeds),
        "stages": {},
        "comparisons": {},
    }
    for key, stage_records in records.items():
        stage = STAGES[key]
        values = [stage_records[seed] for seed in seeds]
        complete = [r for r in values if r.get("status") == "complete"]
        success_count = sum(1 for r in complete if r.get("success"))
        runtimes = [float(r["runtime_sec"]) for r in values if r.get("runtime_sec") is not None]
        avg_runtime = mean(runtimes) if runtimes else stage["avg_runtime_sec"]
        metrics["stages"][key] = {
            "label": stage["label"],
            "processed": len(values),
            "complete": len(complete),
            "successes": success_count,
            "success_rate_complete": success_count / len(complete) if complete else 0.0,
            "outcomes": dict(Counter(str(r.get("outcome")) for r in values)),
            "avg_runtime_sec": avg_runtime,
            "runtime_source": "records" if runtimes else "README prior benchmark",
        }
    metrics["comparisons"]["m5_vs_m0"] = compare_stage(records["m5"], records["m0"], seeds)
    metrics["comparisons"]["m5_vs_m4c"] = compare_stage(records["m5"], records["m4c"], seeds)
    m0 = metrics["stages"]["m0"]
    m5 = metrics["stages"]["m5"]
    metrics["m5_relative_to_m0"] = {
        "success_rate_delta_pp": 100.0 * (m5["success_rate_complete"] - m0["success_rate_complete"]),
        "success_count_delta": int(m5["successes"] - m0["successes"]),
        "avg_runtime_delta_sec": (
            float(m5["avg_runtime_sec"]) - float(m0["avg_runtime_sec"])
            if m5["avg_runtime_sec"] is not None and m0["avg_runtime_sec"] is not None
            else None
        ),
        "avg_runtime_ratio": (
            float(m5["avg_runtime_sec"]) / float(m0["avg_runtime_sec"])
            if m5["avg_runtime_sec"] is not None and m0["avg_runtime_sec"] not in (None, 0)
            else None
        ),
    }
    return metrics


def compare_stage(a: dict[int, dict], b: dict[int, dict], seeds: list[int]) -> dict:
    improved = [seed for seed in seeds if a[seed].get("success") and not b[seed].get("success")]
    regressed = [seed for seed in seeds if b[seed].get("success") and not a[seed].get("success")]
    both_success = [seed for seed in seeds if a[seed].get("success") and b[seed].get("success")]
    both_failed = [seed for seed in seeds if not a[seed].get("success") and not b[seed].get("success")]
    return {
        "improved_seeds": improved,
        "regressed_seeds": regressed,
        "both_success_seeds": both_success,
        "both_failed_seeds": both_failed,
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "n_both_success": len(both_success),
        "n_both_failed": len(both_failed),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(metrics: dict, csv_path: Path) -> str:
    lines = [
        "# M0 / M4C / M5 Collision World Benchmark",
        "",
        f"Seeds: {metrics['seed_start']}-{metrics['seed_end']}",
        "",
        "## Summary",
        "",
        "| Method | Complete | Lift success | Success rate | Outcomes | Avg runtime | Runtime source |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for key in ("m0", "m4c", "m5"):
        stage = metrics["stages"][key]
        outcomes = ", ".join(f"{k}: {v}" for k, v in sorted(stage["outcomes"].items()))
        avg_runtime = stage["avg_runtime_sec"]
        lines.append(
            f"| {stage['label']} | {stage['complete']}/{stage['processed']} | "
            f"{stage['successes']} | {100*stage['success_rate_complete']:.2f}% | "
            f"{outcomes} | {avg_runtime:.2f}s | {stage['runtime_source']} |"
        )
    rel = metrics["m5_relative_to_m0"]
    lines.extend(
        [
            "",
            "## M5 Relative To M0",
            "",
            f"- Lift success delta: {rel['success_count_delta']} seeds ({rel['success_rate_delta_pp']:.2f} pp).",
            f"- Avg runtime delta: {rel['avg_runtime_delta_sec']:.2f}s, ratio {rel['avg_runtime_ratio']:.2f}x.",
            "",
            "## Seed-Level Changes",
            "",
        ]
    )
    for name in ("m5_vs_m0", "m5_vs_m4c"):
        comp = metrics["comparisons"][name]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Improved: {comp['n_improved']} seeds: {format_seed_list(comp['improved_seeds'])}",
                f"- Regressed: {comp['n_regressed']} seeds: {format_seed_list(comp['regressed_seeds'])}",
                f"- Both success: {comp['n_both_success']}",
                f"- Both failed: {comp['n_both_failed']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-Seed Table",
            "",
            f"CSV: `{csv_path}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def format_seed_list(seeds: list[int], limit: int = 80) -> str:
    if not seeds:
        return "-"
    shown = ", ".join(str(seed) for seed in seeds[:limit])
    if len(seeds) > limit:
        shown += f", ... (+{len(seeds) - limit})"
    return shown


if __name__ == "__main__":
    raise SystemExit(main())
