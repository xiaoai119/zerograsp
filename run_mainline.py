#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the minimal ZeroGrasp mainline on one RGB-D + mask + camera input set, "
            "then export raw reconstruction/grasp outputs plus a human-readable top-1 grasp summary."
        )
    )
    parser.add_argument("--img_path", required=True, help="RGB image path.")
    parser.add_argument("--depth_path", required=True, help="Depth image path.")
    parser.add_argument("--mask_path", required=True, help="Mask image path.")
    parser.add_argument("--camera_info_path", required=True, help="Camera intrinsics/meta path.")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
        help="Checkpoint path relative to the package root by default.",
    )
    parser.add_argument(
        "--config",
        default="configs/demo.yaml",
        help="Config path relative to the package root by default.",
    )
    return parser.parse_args()


def parse_counts(log_text: str) -> tuple[list[int], list[int]]:
    before = [int(m.group(1)) for m in re.finditer(r"Number of grasps before collision detection (\d+)", log_text)]
    after = [int(m.group(1)) for m in re.finditer(r"Number of grasps after collision detection (\d+)", log_text)]
    return before, after


def parse_object_index(path: Path) -> int | None:
    stem = path.stem
    if stem.endswith(".grasp"):
        stem = stem[: -len(".grasp")]
    maybe_idx = stem.split("_")[-1]
    return int(maybe_idx) if maybe_idx.isdigit() else None


def summarize_top1(grasp_files: list[Path]) -> dict[str, object] | None:
    best = None
    for grasp_path in grasp_files:
        grasps = np.asarray(np.load(grasp_path), dtype=np.float64)
        if grasps.size == 0:
            continue
        row = grasps[0]
        entry = {
            "source_file": grasp_path.name,
            "object_index": parse_object_index(grasp_path),
            "score": float(row[0]),
            "width_m": float(row[1]),
            "height_m": float(row[2]),
            "depth_m": float(row[3]),
            "rotation_matrix_camera": row[4:13].reshape(3, 3).tolist(),
            "translation_m_camera": row[13:16].tolist(),
            "object_id": int(row[16]) if row.shape[0] > 16 else None,
        }
        if best is None or entry["score"] > best["score"]:
            best = entry
    return best


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_outputs_dir = output_dir / "raw_outputs"
    logs_dir = output_dir / "logs"
    for path in [output_dir, raw_outputs_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (package_root / checkpoint_path).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (package_root / config_path).resolve()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    command = [
        sys.executable,
        str((package_root / "demo.py").resolve()),
        "--img_path",
        str(Path(args.img_path).expanduser().resolve()),
        "--depth_path",
        str(Path(args.depth_path).expanduser().resolve()),
        "--mask_path",
        str(Path(args.mask_path).expanduser().resolve()),
        "--camera_info_path",
        str(Path(args.camera_info_path).expanduser().resolve()),
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(config_path),
        "--output_dir",
        str(raw_outputs_dir),
    ]

    started = time.time()
    proc = subprocess.run(
        command,
        cwd=str(package_root),
        env=env,
        capture_output=True,
        text=True,
    )
    runtime_sec = time.time() - started

    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    grasp_files = sorted(raw_outputs_dir.glob("*.grasp.npy"))
    ply_files = sorted(raw_outputs_dir.glob("*.ply"))
    raw_counts, filtered_counts = parse_counts(proc.stdout + "\n" + proc.stderr)
    top1 = summarize_top1(grasp_files)

    report = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "exit_code": int(proc.returncode),
        "runtime_sec": float(runtime_sec),
        "command": command,
        "raw_outputs_dir": str(raw_outputs_dir),
        "grasp_files": [path.name for path in grasp_files],
        "ply_files": [path.name for path in ply_files],
        "n_grasp_files": len(grasp_files),
        "n_ply_files": len(ply_files),
        "raw_counts_by_object": raw_counts,
        "post_collision_counts_by_object": filtered_counts,
        "recommended_grasp_top1": top1,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }

    (output_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if top1 is not None:
        (output_dir / "recommended_grasp_top1.json").write_text(
            json.dumps(top1, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "reconstruction_summary.json").write_text(
        json.dumps(
            {
                "ply_files": [path.name for path in ply_files],
                "note": "Each PLY is one reconstructed object-level point cloud exported by the ZeroGrasp mainline.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
