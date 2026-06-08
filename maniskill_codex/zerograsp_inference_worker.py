"""Persistent JSON-lines worker for repeated ZeroGrasp inference."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable


READY_PREFIX = "ZEROGRASP_WORKER_READY "
RESPONSE_PREFIX = "ZEROGRASP_WORKER_RESPONSE "


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
    )
    parser.add_argument("--config", default="configs/maniskill.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--enable-collision-detection", action="store_true")
    return parser.parse_args(argv)


def emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=False), flush=True)


def synchronize_cuda(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    # Keep stdout reserved for the worker protocol. Third-party import and
    # inference diagnostics remain visible on the worker stderr log.
    with contextlib.redirect_stdout(sys.stderr):
        import torch

        from maniskill_codex.run_zerograsp_inference import (
            save_zerograsp_result,
            seed_inference,
        )
        from zerograsp.pipeline import ZeroGraspPipeline

        seed_inference(0)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        load_started = time.perf_counter()
        pipeline = ZeroGraspPipeline(
            checkpoint_path=str(Path(args.checkpoint).expanduser().resolve()),
            config_path=str(Path(args.config).expanduser().resolve()),
            device=device,
        )
        pipeline._config.use_collision_detection = bool(
            args.enable_collision_detection
        )
        synchronize_cuda(torch, device)
        model_load_sec = time.perf_counter() - load_started

    emit(
        READY_PREFIX,
        {
            "pid": int(__import__("os").getpid()),
            "device": device,
            "collision_detection": bool(args.enable_collision_detection),
            "model_load_sec": model_load_sec,
            "startup_sec": time.perf_counter() - started,
        },
    )

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(text)
            request_id = request.get("request_id")
            if request.get("command") == "shutdown":
                emit(
                    RESPONSE_PREFIX,
                    {"request_id": request_id, "ok": True, "shutdown": True},
                )
                return 0

            input_dir = Path(request["input_dir"]).expanduser().resolve()
            output_dir = Path(request["output_dir"]).expanduser().resolve()
            random_seed = int(request.get("random_seed", 0))
            seed_inference(random_seed)

            synchronize_cuda(torch, device)
            inference_started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                result = pipeline.predict_from_files(
                    rgb_path=str(input_dir / "rgb.png"),
                    depth_path=str(input_dir / "depth.png"),
                    mask_path=str(input_dir / "mask.png"),
                    camera_path=str(input_dir / "camera.json"),
                )
                report = save_zerograsp_result(result, output_dir)
            synchronize_cuda(torch, device)
            elapsed = time.perf_counter() - inference_started

            report["use_collision_detection"] = bool(
                args.enable_collision_detection
            )
            report["random_seed"] = random_seed
            report["worker_inference_and_save_sec"] = elapsed
            report["persistent_worker"] = True
            (output_dir / "run_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            emit(
                RESPONSE_PREFIX,
                {
                    "request_id": request_id,
                    "ok": report["recommended_grasp"] is not None,
                    "runtime_sec": elapsed,
                    "model_runtime_sec": float(report["runtime_sec"]),
                    "recommended_grasp_available": (
                        report["recommended_grasp"] is not None
                    ),
                    "output_dir": str(output_dir),
                },
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit(
                RESPONSE_PREFIX,
                {
                    "request_id": request.get("request_id"),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
