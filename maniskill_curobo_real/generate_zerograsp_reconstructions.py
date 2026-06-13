#!/usr/bin/env python3
"""Generate all-object ZeroGrasp surface reconstructions for M3/M4 worlds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from maniskill_curobo.scripts.run_zerograsp_depth_ab_batch import (
    PersistentZeroGraspRunner,
)
from maniskill_curobo_real.run_world_collision_stages import (
    DEFAULT_CAMERA_EYE,
    DEFAULT_CAMERA_TARGET,
    DEFAULT_MANISKILL_PYTHON,
    DEFAULT_ZEROGRASP_PYTHON,
    build_real_benchmark_env,
    write_json_atomic,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="PickClutterYCB-v1")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument(
        "--output-root",
        default=(
            "maniskill_curobo_real/runs/"
            "pickclutter_zerograsp_reconstructions_seed1_200"
        ),
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=list(DEFAULT_CAMERA_TARGET),
    )
    parser.add_argument("--settle-before-export-steps", type=int, default=20)
    parser.add_argument("--zerograsp-python", default=str(DEFAULT_ZEROGRASP_PYTHON))
    parser.add_argument("--maniskill-python", default=str(DEFAULT_MANISKILL_PYTHON))
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
    )
    parser.add_argument("--config", default="configs/maniskill.yaml")
    parser.add_argument(
        "--collision-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--worker-timeout-sec", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed_start > args.seed_end:
        raise ValueError("--seed-start must not exceed --seed-end.")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    records = load_records(output_root) if args.reuse_existing else {}
    env = build_real_benchmark_env(args)
    worker = PersistentZeroGraspRunner(
        repo_root=repo_root,
        python=Path(args.zerograsp_python).expanduser(),
        checkpoint=args.checkpoint,
        config=args.config,
        collision_detection=bool(args.collision_detection),
        timeout_sec=float(args.worker_timeout_sec),
        logs_dir=output_root / "worker_logs",
    )
    try:
        for index, seed in enumerate(seeds, start=1):
            print(f"[reconstruction] seed {seed} ({index}/{len(seeds)})", flush=True)
            record = generate_seed(
                args=args,
                env=env,
                worker=worker,
                output_root=output_root,
                seed=seed,
            )
            records[seed] = record
            write_results(output_root, args, seeds, records, worker.ready_report)
            print(
                f"  status={record['status']} objects={record.get('n_objects')} "
                f"points={record.get('n_reconstruction_points')}",
                flush=True,
            )
    finally:
        worker.close()
        env.close()
    summary = write_results(output_root, args, seeds, records, worker.ready_report)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2), flush=True)
    return 0


def generate_seed(
    *,
    args: argparse.Namespace,
    env: Any,
    worker: PersistentZeroGraspRunner,
    output_root: Path,
    seed: int,
) -> dict[str, Any]:
    from maniskill_codex.zerograsp_inputs import (
        extract_zerograsp_input,
        save_zerograsp_input_bundle,
    )
    from maniskill_curobo.scripts import execute_curobo_pick as execute

    started = time.time()
    setup_dir = output_root / f"seed{seed:03d}" / "setup"
    input_dir = setup_dir / "zg_input"
    output_dir = setup_dir / "zg_output"
    reconstruction_files = list(
        (output_dir / "raw_outputs").glob("*.reconstruction.npz")
    )
    if args.reuse_existing and reconstruction_files:
        return summarize_seed(seed, input_dir, output_dir, reused=True)

    obs, _ = env.reset(seed=seed)
    if int(args.settle_before_export_steps) > 0:
        obs, _ = execute.settle_environment(
            env,
            steps=int(args.settle_before_export_steps),
            gripper=1.0,
        )
    bundle = extract_zerograsp_input(
        obs,
        env,
        args.camera,
        mask_mode="all-objects",
    )
    save_zerograsp_input_bundle(bundle, input_dir)
    camera_model = execute.camera_model_matrix(env, args.camera)
    world_from_base = execute.robot_base_matrix(env)
    np.savez_compressed(
        input_dir / "capture_calibration.npz",
        camera_model_matrix=np.asarray(camera_model, dtype=np.float64),
        world_from_base_matrix=np.asarray(world_from_base, dtype=np.float64),
    )
    command = worker.run(
        input_dir=input_dir,
        output_dir=output_dir,
        random_seed=seed,
        logs_dir=setup_dir / "logs",
        name="zerograsp_reconstruction",
    )
    record = summarize_seed(seed, input_dir, output_dir, command=command)
    record["total_runtime_sec"] = float(time.time() - started)
    return record


def summarize_seed(
    seed: int,
    input_dir: Path,
    output_dir: Path,
    *,
    command: dict[str, Any] | None = None,
    reused: bool = False,
) -> dict[str, Any]:
    report_path = output_dir / "run_report.json"
    if not report_path.is_file():
        return {
            "seed": int(seed),
            "status": "missing_run_report",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "command": command,
            "reused": reused,
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    objects = [
        record
        for record in report.get("objects", [])
        if record.get("reconstruction_file")
        and (output_dir / record["reconstruction_file"]).is_file()
    ]
    camera = json.loads((input_dir / "camera.json").read_text(encoding="utf-8"))
    return {
        "seed": int(seed),
        "status": "complete" if objects else "no_reconstruction",
        "n_objects": len(objects),
        "n_input_objects": len(camera.get("objects", [])),
        "n_target_objects": sum(
            bool(record.get("is_task_target"))
            for record in camera.get("objects", [])
        ),
        "n_reconstruction_points": sum(
            int(record.get("n_reconstruction_points", 0)) for record in objects
        ),
        "model_runtime_sec": report.get("runtime_sec"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "command": command,
        "reused": reused,
    }


def load_records(output_root: Path) -> dict[int, dict[str, Any]]:
    path = output_root / "records.jsonl"
    if not path.is_file():
        return {}
    return {
        int(record["seed"]): record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }


def write_results(
    output_root: Path,
    args: argparse.Namespace,
    seeds: list[int],
    records: dict[int, dict[str, Any]],
    ready_report: dict[str, Any],
) -> dict[str, Any]:
    ordered = [records[seed] for seed in seeds if seed in records]
    complete = [record for record in ordered if record.get("status") == "complete"]
    summary = {
        "env_id": args.env_id,
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_end),
        "mask_mode": "all-objects",
        "settle_before_export_steps": int(args.settle_before_export_steps),
        "collision_detection": bool(args.collision_detection),
        "worker": ready_report,
        "counts": {
            "processed": len(ordered),
            "complete": len(complete),
            "objects": sum(int(record.get("n_objects", 0)) for record in complete),
            "reconstruction_points": sum(
                int(record.get("n_reconstruction_points", 0))
                for record in complete
            ),
        },
    }
    write_json_atomic(output_root / "summary.json", summary)
    write_json_atomic(
        output_root / "progress.json",
        {
            "processed": len(ordered),
            "requested": len(seeds),
            "counts": summary["counts"],
        },
    )
    temporary = output_root / "records.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as file:
        for record in ordered:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_root / "records.jsonl")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
