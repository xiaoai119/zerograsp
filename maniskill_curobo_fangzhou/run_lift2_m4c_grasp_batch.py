#!/usr/bin/env python3
"""Run ZeroGrasp inference and Lift2 M4C grasp execution over a seed range."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_ROOT = (
    PROJECT_ROOT
    / "maniskill_curobo_fangzhou"
    / "runs"
    / "m4c_lift2_collision_spheres_seed1_200"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "maniskill_curobo_fangzhou"
    / "runs"
    / "lift2_m4c_grasp_seed001_050_new_tcp"
)
DEFAULT_ZEROGRASP_PYTHON = Path(
    "/home/openclaw-server/miniconda3/envs/graduate/bin/python"
)
DEFAULT_MANISKILL_PYTHON = (
    PROJECT_ROOT / "maniskill_curobo" / "envs" / "maniskill_curobo" / "bin" / "python"
)
WORKER_READY_PREFIX = "LIFT2_M4C_WORKER_READY "
WORKER_RESPONSE_PREFIX = "LIFT2_M4C_WORKER_RESPONSE "


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--zerograsp-python", type=Path, default=DEFAULT_ZEROGRASP_PYTHON)
    parser.add_argument("--maniskill-python", type=Path, default=DEFAULT_MANISKILL_PYTHON)
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument("--pregrasp-offset", type=float, default=0.16)
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--grasp-depth-scale", type=float, default=1.0)
    parser.add_argument("--grasp-depth-max-offset", type=float, default=0.04)
    parser.add_argument(
        "--grasp-depth-fallback-fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
    )
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--persistent-execution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed_start > args.seed_end:
        raise ValueError("seed-start must be <= seed-end")

    scene_root = args.scene_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "records.jsonl"
    records = load_records(records_path) if args.resume else {}
    write_json(
        output_root / "command.json",
        {
            **sanitize(vars(args)),
            "project_root": str(PROJECT_ROOT),
            "pid": os.getpid(),
        },
    )

    print("Loading ZeroGrasp model once for the full batch...", flush=True)
    from maniskill_codex.run_zerograsp_inference import (
        save_zerograsp_result,
        seed_inference,
    )
    from zerograsp.pipeline import ZeroGraspPipeline

    seed_inference(0)
    pipeline = ZeroGraspPipeline(
        checkpoint_path=str(
            (PROJECT_ROOT / "checkpoints" / "zerograsp_cvpr2025" / "zerograsp_demo.ckpt")
            .resolve()
        ),
        config_path=str((PROJECT_ROOT / "configs" / "maniskill.yaml").resolve()),
        device=None,
    )
    pipeline._config.use_collision_detection = True
    print("ZeroGrasp model ready.", flush=True)

    execution_worker = (
        PersistentLift2ExecutionWorker(
            python=args.maniskill_python,
            logs_dir=output_root / "worker_logs",
        )
        if args.persistent_execution
        else None
    )
    try:
        for index, seed in enumerate(range(args.seed_start, args.seed_end + 1), start=1):
            if args.resume and is_complete(records.get(seed)):
                print(f"[{index}/{seed_count(args)}] seed {seed}: reused", flush=True)
                continue
            print(f"[{index}/{seed_count(args)}] seed {seed}: starting", flush=True)
            record = run_seed(
                args=args,
                scene_root=scene_root,
                output_root=output_root,
                seed=seed,
                pipeline=pipeline,
                save_zerograsp_result=save_zerograsp_result,
                seed_inference=seed_inference,
                execution_worker=execution_worker,
            )
            records[seed] = record
            write_records(records_path, records)
            write_summary(output_root / "summary.json", records, args)
            print(
                f"  status={record['status']} "
                f"inference={record.get('inference_status')} "
                f"execution={record.get('execution_status')} "
                f"lift={record.get('object_lift_success')} "
                f"runtime={record['runtime_sec']:.1f}s",
                flush=True,
            )
    finally:
        if execution_worker is not None:
            execution_worker.close()
    return 0


def run_seed(
    *,
    args: argparse.Namespace,
    scene_root: Path,
    output_root: Path,
    seed: int,
    pipeline: Any,
    save_zerograsp_result: Any,
    seed_inference: Any,
    execution_worker: "PersistentLift2ExecutionWorker | None",
) -> dict[str, Any]:
    started = time.time()
    seed_dir = output_root / f"seed{seed:03d}"
    zero_dir = seed_dir / "zerograsp_output"
    execution_dir = seed_dir / "execution"
    input_dir = scene_root / f"seed{seed:03d}" / "zerograsp_input"
    scene_model = scene_root / f"seed{seed:03d}" / "real_scene" / "curobo_scene_voxel.npz"
    seed_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "seed": seed,
        "status": "failed",
        "input_dir": str(input_dir),
        "scene_model": str(scene_model),
        "zerograsp_output": str(zero_dir),
        "execution_dir": str(execution_dir),
    }
    try:
        require_file(input_dir / "rgb.png")
        require_file(input_dir / "depth.png")
        require_file(input_dir / "mask.png")
        require_file(input_dir / "camera.json")
        require_file(scene_model)

        inference_log = seed_dir / "zerograsp.log"
        inference_returncode = 0
        try:
            zero_dir.mkdir(parents=True, exist_ok=True)
            seed_inference(0)
            with inference_log.open("w", encoding="utf-8") as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = pipeline.predict_from_files(
                        rgb_path=str(input_dir / "rgb.png"),
                        depth_path=str(input_dir / "depth.png"),
                        mask_path=str(input_dir / "mask.png"),
                        camera_path=str(input_dir / "camera.json"),
                    )
                    report = save_zerograsp_result(result, zero_dir)
                    report["use_collision_detection"] = True
                    report["random_seed"] = 0
                    write_json(zero_dir / "run_report.json", report)
            if report.get("recommended_grasp") is None:
                inference_returncode = 2
        except Exception:
            inference_returncode = 1
            inference_log.write_text(traceback.format_exc(), encoding="utf-8")
        record["inference_returncode"] = inference_returncode
        record["inference_log"] = str(inference_log)
        record["inference_status"] = "ok" if inference_returncode == 0 else "failed"
        if inference_returncode != 0:
            record["status"] = "inference_failed"
            return finish_record(record, started)
        from maniskill_codex.grasp_projection import draw_grasp_projection

        top1_projection = seed_dir / "zerograsp_top1_projection.png"
        draw_grasp_projection(
            rgb_path=input_dir / "rgb.png",
            camera_path=input_dir / "camera.json",
            grasp_path=zero_dir / "recommended_grasp_top1.json",
            output_path=top1_projection,
            approach_axis="positive-x",
        )
        record["zerograsp_top1_projection"] = str(top1_projection)

        execution_log = seed_dir / "execution.log"
        execution_command = [
            str(args.maniskill_python.expanduser().resolve()),
            "-m",
            "maniskill_curobo_fangzhou.execute_lift2_m4c_grasp",
            "--seed",
            str(seed),
            "--robot-uid",
            "lift2_full_collision",
            "--config",
            str(
                PROJECT_ROOT
                / "maniskill_curobo_fangzhou"
                / "config"
                / "lift2_right_arm_curobo.yml"
            ),
            "--scene-model",
            str(scene_model),
            "--zerograsp-output",
            str(zero_dir),
            "--candidate-top-k",
            str(args.candidate_top_k),
            "--approach-axis",
            "positive-x",
            "--orientation-mode",
            "lift2-gripper",
            "--pregrasp-offset",
            str(args.pregrasp_offset),
            "--lift-offset",
            str(args.lift_offset),
            "--grasp-depth-scale",
            str(args.grasp_depth_scale),
            "--grasp-depth-max-offset",
            str(args.grasp_depth_max_offset),
            "--grasp-depth-fallback-fractions",
            *[str(value) for value in args.grasp_depth_fallback_fractions],
            "--settle-steps",
            str(args.settle_steps),
            "--close-steps",
            str(args.close_steps),
            "--action-repeat",
            str(args.action_repeat),
            "--render-width",
            "1280",
            "--render-height",
            "720",
            "--render-camera-eye",
            "-0.05",
            "0.72",
            "0.40",
            "--render-camera-target",
            "0.30",
            "0.35",
            "0.12",
            "--output-dir",
            str(execution_dir),
        ]
        if not args.video:
            execution_command.append("--no-video")
        if execution_worker is None:
            execution_returncode = run_logged(execution_command, execution_log)
            execution_result: dict[str, Any] = {
                "persistent_execution": False,
            }
        else:
            execution_result = execution_worker.run(
                argv=execution_command[3:],
                stdout_log=execution_log,
                stderr_log=seed_dir / "execution.stderr.log",
            )
            execution_returncode = int(execution_result["exit_code"])
        record["execution_returncode"] = execution_returncode
        record["execution_log"] = str(execution_log)
        record["execution_worker"] = execution_result
        record["execution_status"] = "ok" if execution_returncode == 0 else "failed"
        manifest = load_json(execution_dir / "run_manifest.json")
        metrics = manifest.get("object_lift_metrics", {}) if manifest else {}
        record["object_lift_success"] = metrics.get("object_lift_success")
        record["height_delta_m"] = metrics.get("height_delta_m")
        record["failure_reason"] = metrics.get("failure_reason")
        record["selected_candidate"] = manifest.get("selected_candidate") if manifest else None
        record["video"] = manifest.get("video_saved") if manifest else None
        selected_candidate = manifest.get("selected_candidate") if manifest else None
        if selected_candidate:
            selected_grasp_path = execution_dir / "selected_grasp_camera.json"
            write_json(selected_grasp_path, selected_candidate)
            selected_projection = execution_dir / "selected_grasp_projection.png"
            draw_grasp_projection(
                rgb_path=input_dir / "rgb.png",
                camera_path=input_dir / "camera.json",
                grasp_path=selected_grasp_path,
                output_path=selected_projection,
                approach_axis="positive-x",
            )
            record["selected_grasp_projection"] = str(selected_projection)
        record["status"] = "completed" if execution_returncode == 0 else "execution_failed"
        return finish_record(record, started)
    except Exception as exc:
        error_path = seed_dir / "batch_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_trace"] = str(error_path)
        return finish_record(record, started)


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def finish_record(record: dict[str, Any], started: float) -> dict[str, Any]:
    record["runtime_sec"] = float(time.time() - started)
    return record


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["seed"])] = record
    return records


def is_complete(record: dict[str, Any] | None) -> bool:
    return bool(record and record.get("status") in {"completed", "execution_failed", "inference_failed"})


def write_records(path: Path, records: dict[int, dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(records[seed], ensure_ascii=False, sort_keys=True) + "\n"
            for seed in sorted(records)
        ),
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    records: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    values = [records[seed] for seed in sorted(records)]
    completed = [item for item in values if item.get("status") == "completed"]
    lift_successes = [item for item in completed if item.get("object_lift_success") is True]
    summary = {
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "expected": seed_count(args),
        "processed": len(values),
        "completed": len(completed),
        "lift_success": len(lift_successes),
        "lift_success_rate_over_completed": (
            len(lift_successes) / len(completed) if completed else 0.0
        ),
        "inference_failed": sum(item.get("status") == "inference_failed" for item in values),
        "execution_failed": sum(item.get("status") == "execution_failed" for item in values),
        "other_failed": sum(item.get("status") == "failed" for item in values),
        "updated_at_unix": time.time(),
    }
    write_json(path, summary)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def seed_count(args: argparse.Namespace) -> int:
    return int(args.seed_end - args.seed_start + 1)


def sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


class PersistentLift2ExecutionWorker:
    def __init__(self, *, python: Path, logs_dir: Path):
        self.logs_dir = logs_dir.expanduser().resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.logs_dir / "worker.stdout.log"
        self.stderr_path = self.logs_dir / "worker.stderr.log"
        self._stdout_log = self.stdout_path.open("a", encoding="utf-8")
        self._stderr_log = self.stderr_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(python.expanduser().resolve()),
                "-u",
                "-m",
                "maniskill_curobo_fangzhou.lift2_m4c_execution_worker",
            ],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_log,
            text=True,
            bufsize=1,
        )
        self._request_count = 0
        self.ready = self._read_message(WORKER_READY_PREFIX, request_id=None)

    def run(
        self,
        *,
        argv: list[str],
        stdout_log: Path,
        stderr_log: Path,
    ) -> dict[str, Any]:
        self._request_count += 1
        request_id = self._request_count
        request = {
            "request_id": request_id,
            "argv": argv,
            "stdout_log": str(stdout_log.expanduser().resolve()),
            "stderr_log": str(stderr_log.expanduser().resolve()),
        }
        if self.process.stdin is None:
            raise RuntimeError("Lift2 execution worker stdin is unavailable")
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        return self._read_message(WORKER_RESPONSE_PREFIX, request_id=request_id)

    def _read_message(
        self,
        prefix: str,
        *,
        request_id: int | None,
    ) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("Lift2 execution worker stdout is unavailable")
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"Lift2 execution worker exited with code {self.process.poll()}"
                )
            self._stdout_log.write(line)
            self._stdout_log.flush()
            text = line.strip()
            if not text.startswith(prefix):
                continue
            payload = json.loads(text[len(prefix) :])
            if request_id is None or payload.get("request_id") == request_id:
                return payload

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self._stdout_log.close()
        self._stderr_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
