"""Export one ManiSkill scene observation as a ZeroGrasp input bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from maniskill_codex.camera_views import add_camera_view_args
from maniskill_codex.execute_zerograsp_pick import build_env
from maniskill_codex.zerograsp_inputs import (
    MASK_MODES,
    extract_zerograsp_input,
    save_zerograsp_input_bundle,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save ManiSkill RGB-D/mask/camera files for ZeroGrasp.")
    parser.add_argument("--output-dir", required=True, help="Directory to write rgb.png/depth.png/mask.png/camera.json.")
    parser.add_argument("--env-id", default="PickClutterYCB-v1", help="ManiSkill environment id.")
    parser.add_argument("--seed", type=int, default=42, help="ManiSkill reset seed.")
    parser.add_argument("--camera", default="base_camera", help="ManiSkill sensor name.")
    parser.add_argument("--width", type=int, default=1280, help="Sensor image width.")
    parser.add_argument("--height", type=int, default=1024, help="Sensor image height.")
    add_camera_view_args(parser)
    parser.add_argument(
        "--mask-mode",
        choices=MASK_MODES,
        default="task-target",
        help=(
            "Which ManiSkill segmentation ids to pass to ZeroGrasp: task-target, "
            "all-objects, or legacy visible-area filtering."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    env = build_env(
        width=args.width,
        height=args.height,
        render_width=args.width,
        render_height=args.height,
        env_id=args.env_id,
        camera_name=args.camera,
        camera_eye=args.camera_eye,
        camera_target=args.camera_target,
    )
    try:
        obs, _ = env.reset(seed=args.seed)
        bundle = extract_zerograsp_input(obs, env, args.camera, mask_mode=args.mask_mode)
        out = save_zerograsp_input_bundle(bundle, args.output_dir)
        scene = {
            "env_id": args.env_id,
            "seed": args.seed,
            "camera": args.camera,
            "width": args.width,
            "height": args.height,
            "camera_eye": args.camera_eye,
            "camera_target": args.camera_target,
            "mask_mode": args.mask_mode,
            "n_objects": len(bundle.object_records),
            "objects": bundle.object_records,
        }
        (Path(out) / "scene.json").write_text(
            json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output_dir": str(out), **scene}, ensure_ascii=False, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
