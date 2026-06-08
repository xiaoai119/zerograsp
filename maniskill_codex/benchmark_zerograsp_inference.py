"""Benchmark ZeroGrasp cold loading, warmup, and steady-state inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable


SCRIPT_STARTED = time.perf_counter()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing rgb/depth/mask/camera.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
    )
    parser.add_argument("--config", default="configs/maniskill.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--enable-collision-detection", action="store_true")
    parser.add_argument("--output", default=None, help="Optional benchmark JSON path.")
    return parser.parse_args(argv)


def synchronize_cuda(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def timed_call(function, *, torch_module: Any, device: str):
    synchronize_cuda(torch_module, device)
    started = time.perf_counter()
    result = function()
    synchronize_cuda(torch_module, device)
    return result, time.perf_counter() - started


def timing_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "count": len(ordered),
        "values_sec": ordered,
        "mean_sec": statistics.mean(ordered),
        "median_sec": statistics.median(ordered),
        "min_sec": ordered[0],
        "max_sec": ordered[-1],
        "p90_sec": percentile(0.90),
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be positive.")

    input_dir = Path(args.input_dir).expanduser().resolve()
    paths = {
        "rgb": input_dir / "rgb.png",
        "depth": input_dir / "depth.png",
        "mask": input_dir / "mask.png",
        "camera": input_dir / "camera.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ZeroGrasp input files: {missing}")

    import_started = time.perf_counter()
    import torch

    from maniskill_codex.run_zerograsp_inference import seed_inference
    from zerograsp.pipeline import ZeroGraspPipeline
    from zerograsp.utils.dataset import fetch_data

    import_sec = time.perf_counter() - import_started
    seed_inference(args.random_seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    config = str(Path(args.config).expanduser().resolve())
    pipeline, model_load_sec = timed_call(
        lambda: ZeroGraspPipeline(
            checkpoint_path=checkpoint,
            config_path=config,
            device=device,
        ),
        torch_module=torch,
        device=device,
    )
    pipeline._config.use_collision_detection = bool(args.enable_collision_detection)

    batch, input_prepare_sec = timed_call(
        lambda: fetch_data(
            str(paths["rgb"]),
            str(paths["depth"]),
            str(paths["mask"]),
            str(paths["camera"]),
            pipeline._config,
            1.0,
            device=device,
        ),
        torch_module=torch,
        device=device,
    )

    warmup_times = []
    for _ in range(args.warmup_runs):
        _, elapsed = timed_call(
            lambda: pipeline._infer(batch),
            torch_module=torch,
            device=device,
        )
        warmup_times.append(elapsed)

    full_inference_times = []
    last_result = None
    for _ in range(args.benchmark_runs):
        last_result, elapsed = timed_call(
            lambda: pipeline._infer(batch),
            torch_module=torch,
            device=device,
        )
        full_inference_times.append(elapsed)

    model_forward_times = []
    for _ in range(args.benchmark_runs):
        _, elapsed = timed_call(
            lambda: _model_forward(pipeline, batch, torch),
            torch_module=torch,
            device=device,
        )
        model_forward_times.append(elapsed)

    file_end_to_end_times = []
    for _ in range(args.benchmark_runs):
        _, elapsed = timed_call(
            lambda: pipeline.predict_from_files(
                rgb_path=str(paths["rgb"]),
                depth_path=str(paths["depth"]),
                mask_path=str(paths["mask"]),
                camera_path=str(paths["camera"]),
            ),
            torch_module=torch,
            device=device,
        )
        file_end_to_end_times.append(elapsed)

    full_mean = statistics.mean(full_inference_times)
    forward_mean = statistics.mean(model_forward_times)
    report = {
        "input_dir": str(input_dir),
        "checkpoint": checkpoint,
        "checkpoint_size_mb": Path(checkpoint).stat().st_size / (1024**2),
        "config": config,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "collision_detection": bool(args.enable_collision_detection),
        "random_seed": int(args.random_seed),
        "process_to_benchmark_import_sec": import_started - SCRIPT_STARTED,
        "heavy_import_sec": import_sec,
        "model_load_and_device_transfer_sec": model_load_sec,
        "input_prepare_once_sec": input_prepare_sec,
        "warmup": timing_summary(warmup_times),
        "steady_state_full_inference": timing_summary(full_inference_times),
        "steady_state_model_forward": timing_summary(model_forward_times),
        "steady_state_file_end_to_end": timing_summary(file_end_to_end_times),
        "estimated_postprocess_and_collision_mean_sec": max(0.0, full_mean - forward_mean),
        "recommended_grasp_available": bool(
            last_result is not None and last_result.recommended_grasp() is not None
        ),
        "cuda_memory": _cuda_memory_report(torch, device),
        "total_internal_runtime_sec": time.perf_counter() - SCRIPT_STARTED,
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_dir / "zerograsp_benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"benchmark_json={output_path}")
    return 0


def _model_forward(pipeline: Any, batch: tuple, torch_module: Any) -> Any:
    pipeline._model.eval()
    with torch_module.no_grad():
        return pipeline._model.model(batch)


def _cuda_memory_report(torch_module: Any, device: str) -> dict[str, Any] | None:
    if not str(device).startswith("cuda") or not torch_module.cuda.is_available():
        return None
    return {
        "allocated_mb": torch_module.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch_module.cuda.memory_reserved() / (1024**2),
        "peak_allocated_mb": torch_module.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch_module.cuda.max_memory_reserved() / (1024**2),
    }


if __name__ == "__main__":
    raise SystemExit(main())
