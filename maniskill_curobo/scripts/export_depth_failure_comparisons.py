"""Export categorized zero-depth versus corrected-depth failure videos."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


CATEGORIES = {
    "planning_failed_pre": "pre_planning_failed",
    "object_not_lifted": "object_not_lifted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def make_comparison(zero: Path, corrected: Path, output: Path) -> None:
    duration = max(video_duration(zero), video_duration(corrected))
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(zero),
            "-i",
            str(corrected),
            "-filter_complex",
            (
                "[0:v]scale=640:-2,tpad=stop_mode=clone:stop_duration=120,"
                "drawtext=text='Zero depth':x=20:y=20:fontsize=26:"
                "fontcolor=white:box=1:boxcolor=black@0.65[left];"
                "[1:v]scale=640:-2,tpad=stop_mode=clone:stop_duration=120,"
                "drawtext=text='Corrected depth':x=20:y=20:fontsize=26:"
                "fontcolor=white:box=1:boxcolor=black@0.65[right];"
                "[left][right]hstack=inputs=2[out]"
            ),
            "-map",
            "[out]",
            "-an",
            "-t",
            f"{duration:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    for record in sorted(summary["records"], key=lambda item: int(item["seed"])):
        depth = record["variants"]["depth"]
        category = CATEGORIES.get(str(depth["outcome"]))
        if category is None:
            continue
        seed = int(record["seed"])
        source_zero = run_root / f"seed{seed:03d}" / "baseline" / "execution.mp4"
        source_depth = run_root / f"seed{seed:03d}" / "depth" / "execution.mp4"
        destination = output_root / category / f"seed{seed:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        zero = destination / f"seed{seed:03d}_zero_depth.mp4"
        corrected = destination / f"seed{seed:03d}_corrected_depth.mp4"
        comparison = destination / f"seed{seed:03d}_comparison.mp4"
        shutil.copy2(source_zero, zero)
        shutil.copy2(source_depth, corrected)
        make_comparison(zero, corrected, comparison)

        manifest = json.loads(
            (run_root / f"seed{seed:03d}" / "depth" / "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        fallback = (manifest.get("grasp") or {}).get(
            "grasp_depth_auto_fallback"
        ) or {}
        rows.append(
            {
                "seed": seed,
                "category": category,
                "zero_outcome": record["variants"]["baseline"]["outcome"],
                "corrected_depth_outcome": depth["outcome"],
                "selected_depth_scale": fallback.get("selected_scale"),
                "failure_reason": manifest.get("failure_reason")
                or (manifest.get("object_lift_metrics") or {}).get("failure_reason"),
                "zero_video": str(zero),
                "corrected_depth_video": str(corrected),
                "comparison_video": str(comparison),
            }
        )

    fieldnames = [
        "seed",
        "category",
        "zero_outcome",
        "corrected_depth_outcome",
        "selected_depth_scale",
        "failure_reason",
        "zero_video",
        "corrected_depth_video",
        "comparison_video",
    ]
    with (output_root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "source_run": str(run_root),
        "output_root": str(output_root),
        "counts": {
            category: sum(row["category"] == category for row in rows)
            for category in CATEGORIES.values()
        },
        "records": rows,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
