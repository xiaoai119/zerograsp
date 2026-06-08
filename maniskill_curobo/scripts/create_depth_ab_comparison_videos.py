"""Create side-by-side videos for improved and regressed depth A/B seeds."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


CHANGES = ("improved", "regressed")


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


def make_side_by_side(baseline: Path, depth: Path, output: Path, change: str) -> None:
    duration = max(video_duration(baseline), video_duration(depth))
    color = "0x16803c" if change == "improved" else "0xb42318"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(baseline),
            "-i",
            str(depth),
            "-filter_complex",
            (
                "[0:v]scale=640:-2,tpad=stop_mode=clone:stop_duration=120,"
                "drawtext=text='ZeroGrasp without depth':x=20:y=20:fontsize=26:"
                "fontcolor=white:box=1:boxcolor=black@0.65[left];"
                "[1:v]scale=640:-2,tpad=stop_mode=clone:stop_duration=120,"
                "drawtext=text='ZeroGrasp with depth':x=20:y=20:fontsize=26:"
                "fontcolor=white:box=1:boxcolor=black@0.65[right];"
                f"[left][right]hstack=inputs=2,"
                f"drawbox=x=0:y=0:w=iw:h=8:color={color}:t=fill[out]"
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
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = summary.get("counts", {})
    if int(counts.get("processed_seeds", 0)) != int(counts.get("requested_seeds", 0)):
        raise RuntimeError(f"A/B run is incomplete: {counts}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    exported: list[dict[str, object]] = []
    for record in sorted(summary.get("records", []), key=lambda item: int(item["seed"])):
        change = record.get("comparison", {}).get("change")
        if change not in CHANGES:
            continue
        seed = int(record["seed"])
        seed_dir = output_root / change / f"seed{seed:03d}"
        baseline = run_root / f"seed{seed:03d}" / "baseline" / "execution.mp4"
        depth = run_root / f"seed{seed:03d}" / "depth" / "execution.mp4"
        if not baseline.is_file() or not depth.is_file():
            exported.append(
                {
                    "seed": seed,
                    "change": change,
                    "status": "missing_video",
                    "baseline_video": str(baseline),
                    "depth_video": str(depth),
                }
            )
            continue

        seed_dir.mkdir(parents=True, exist_ok=True)
        baseline_copy = seed_dir / f"seed{seed:03d}_baseline.mp4"
        depth_copy = seed_dir / f"seed{seed:03d}_depth.mp4"
        comparison = seed_dir / f"seed{seed:03d}_{change}_comparison.mp4"
        shutil.copy2(baseline, baseline_copy)
        shutil.copy2(depth, depth_copy)
        make_side_by_side(baseline_copy, depth_copy, comparison, change)
        exported.append(
            {
                "seed": seed,
                "change": change,
                "status": "exported",
                "baseline_outcome": record["comparison"].get("baseline_outcome"),
                "depth_outcome": record["comparison"].get("depth_outcome"),
                "baseline_video": str(baseline_copy),
                "depth_video": str(depth_copy),
                "comparison_video": str(comparison),
            }
        )

    report = {
        "source_run": str(run_root),
        "output_root": str(output_root),
        "counts": {
            change: sum(
                row["change"] == change and row["status"] == "exported" for row in exported
            )
            for change in CHANGES
        },
        "records": exported,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "change",
                "status",
                "baseline_outcome",
                "depth_outcome",
                "baseline_video",
                "depth_video",
                "comparison_video",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
