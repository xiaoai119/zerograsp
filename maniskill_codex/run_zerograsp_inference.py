"""Run ZeroGrasp on one saved RGB-D/mask input bundle and write outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ZeroGrasp inference on rgb/depth/mask/camera files.")
    parser.add_argument("--img-path", required=True, help="RGB image path.")
    parser.add_argument("--depth-path", required=True, help="Depth image path.")
    parser.add_argument("--mask-path", required=True, help="Instance mask image path.")
    parser.add_argument("--camera-info-path", required=True, help="Camera JSON path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt",
        help="ZeroGrasp checkpoint path.",
    )
    parser.add_argument("--config", default="configs/maniskill.yaml", help="ZeroGrasp config path.")
    parser.add_argument("--device", default=None, help="Optional torch device.")
    parser.add_argument(
        "--enable-collision-detection",
        action="store_true",
        help="Enable ZeroGrasp collision filtering.",
    )
    return parser.parse_args(argv)


def grasp_prediction_to_json(grasp: Any, source_file: str | None = None) -> dict[str, Any]:
    data = {
        "score": float(grasp.score),
        "width_m": float(grasp.width),
        "height_m": float(grasp.height),
        "depth_m": float(grasp.depth),
        "rotation_matrix_camera": np.asarray(grasp.rotation_matrix, dtype=np.float64).reshape(3, 3).tolist(),
        "translation_m_camera": np.asarray(grasp.translation, dtype=np.float64).reshape(3).tolist(),
        "object_id": int(grasp.object_id),
    }
    if source_file is not None:
        data["source_file"] = source_file
    return data


def save_zerograsp_result(result: Any, output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    raw_dir = out / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    object_reports = []
    source_by_object_id = {}
    for obj in result.objects:
        raw_name = f"object_{int(obj.object_index):03d}_label_{int(obj.object_id)}.grasp.npy"
        raw_path = raw_dir / raw_name
        np.save(raw_path, np.asarray(obj.grasp_group_array, dtype=np.float64))
        rel_raw = str(raw_path.relative_to(out))
        source_by_object_id[int(obj.object_id)] = rel_raw
        object_reports.append(
            {
                "object_id": int(obj.object_id),
                "object_index": int(obj.object_index),
                "raw_grasp_file": rel_raw,
                "n_grasps_before_collision": int(obj.n_grasps_before_collision),
                "n_grasps_after_collision": int(obj.n_grasps_after_collision),
                "n_grasps_final": int(obj.n_grasps_final),
            }
        )

    recommended = result.recommended_grasp()
    recommended_json = None
    if recommended is not None:
        recommended_json = grasp_prediction_to_json(
            recommended,
            source_file=source_by_object_id.get(int(recommended.object_id)),
        )
        (out / "recommended_grasp_top1.json").write_text(
            json.dumps(recommended_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    report = {
        "runtime_sec": float(getattr(result, "runtime_sec", 0.0)),
        "n_objects": len(result.objects),
        "objects": object_reports,
        "recommended_grasp": recommended_json,
    }
    (out / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    from zerograsp.pipeline import ZeroGraspPipeline

    pipeline = ZeroGraspPipeline(
        checkpoint_path=str(Path(args.checkpoint).expanduser().resolve()),
        config_path=str(Path(args.config).expanduser().resolve()),
        device=args.device,
    )
    pipeline._config.use_collision_detection = bool(args.enable_collision_detection)
    result = pipeline.predict_from_files(
        rgb_path=str(Path(args.img_path).expanduser().resolve()),
        depth_path=str(Path(args.depth_path).expanduser().resolve()),
        mask_path=str(Path(args.mask_path).expanduser().resolve()),
        camera_path=str(Path(args.camera_info_path).expanduser().resolve()),
    )
    report = save_zerograsp_result(result, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["recommended_grasp"] is None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
