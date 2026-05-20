"""Project a ZeroGrasp camera-frame grasp back onto an RGB image."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from maniskill_codex.grasp_axes import (
    APPROACH_AXIS_CHOICES,
    zerograsp_approach_vector,
    zerograsp_width_vector,
)

GREEN = (0, 255, 0)
YELLOW = (255, 230, 0)
BLACK = (0, 0, 0)


def project_3d_to_2d(point_3d: np.ndarray, camera_matrix: np.ndarray) -> tuple[int, int] | None:
    """Project one OpenCV camera-frame 3D point into image pixel coordinates."""

    point = np.asarray(point_3d, dtype=np.float64).reshape(3)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    x, y, z = point
    if z <= 0:
        return None

    u = int(round(K[0, 0] * x / z + K[0, 2]))
    v = int(round(K[1, 1] * y / z + K[1, 2]))
    return (u, v)


def draw_grasp_projection(
    rgb_path: str | Path,
    camera_path: str | Path,
    grasp_path: str | Path,
    output_path: str | Path,
    approach_axis: str = "negative-x",
) -> Path:
    """Draw a ZeroGrasp grasp projection on top of an RGB input image."""

    rgb_path = Path(rgb_path)
    output_path = Path(output_path)
    K = _load_camera_matrix(camera_path)
    grasp = _load_grasp(grasp_path)

    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    _draw_grasp(draw, grasp, K, approach_axis=approach_axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a ZeroGrasp grasp projection on an RGB image.")
    parser.add_argument("--rgb", required=True, help="RGB input PNG path.")
    parser.add_argument("--camera", required=True, help="camera.json containing cam_K.")
    parser.add_argument("--grasp", required=True, help="recommended_grasp_top1.json path.")
    parser.add_argument("--output", required=True, help="Output overlay PNG path.")
    parser.add_argument(
        "--approach-axis",
        choices=APPROACH_AXIS_CHOICES,
        default="negative-x",
        help="Which ZeroGrasp local X direction to draw as the approach direction.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = draw_grasp_projection(
        args.rgb,
        args.camera,
        args.grasp,
        args.output,
        approach_axis=args.approach_axis,
    )
    print(f"Grasp projection saved: {output}")
    return 0


def _draw_grasp(
    draw: ImageDraw.ImageDraw,
    grasp: dict,
    K: np.ndarray,
    approach_axis: str = "negative-x",
) -> None:
    position = np.asarray(grasp["translation_m_camera"], dtype=np.float64).reshape(3)
    rotation = np.asarray(grasp["rotation_matrix_camera"], dtype=np.float64).reshape(3, 3)
    width = float(grasp.get("width_m", 0.04))
    depth = float(grasp.get("depth_m", 0.02))
    score = float(grasp.get("score", 0.0))
    object_id = grasp.get("object_id", "?")

    center = project_3d_to_2d(position, K)
    if center is None:
        raise ValueError(f"Cannot project grasp center behind camera: {position.tolist()}")

    approach = _unit(
        zerograsp_approach_vector(rotation, approach_axis),
        fallback=np.array([0.0, 0.0, -1.0]),
    )
    binormal = _unit(zerograsp_width_vector(rotation), fallback=np.array([0.0, -1.0, 0.0]))

    approach_tip = project_3d_to_2d(position + approach * 0.08, K)
    finger_a_3d = position + binormal * (width / 2.0)
    finger_b_3d = position - binormal * (width / 2.0)
    finger_a = project_3d_to_2d(finger_a_3d, K)
    finger_b = project_3d_to_2d(finger_b_3d, K)
    depth_a = project_3d_to_2d(finger_a_3d - approach * depth, K)
    depth_b = project_3d_to_2d(finger_b_3d - approach * depth, K)

    color = GREEN
    if approach_tip is not None:
        _arrow(draw, center, approach_tip, color, width=3)
    if finger_a is not None and finger_b is not None:
        draw.line([finger_a, finger_b], fill=color, width=3)
        _square(draw, finger_a, 8, color)
        _square(draw, finger_b, 8, color)
    if finger_a is not None and depth_a is not None:
        draw.line([finger_a, depth_a], fill=YELLOW, width=2)
    if finger_b is not None and depth_b is not None:
        draw.line([finger_b, depth_b], fill=YELLOW, width=2)

    _circle(draw, center, 5, color)
    label = f"obj={object_id} score={score:.3f} w={width * 100:.1f}cm"
    text_xy = (center[0] + 12, center[1] - 16)
    _text_with_shadow(draw, text_xy, label, fill=color)


def _load_camera_matrix(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "cam_K" not in data:
        raise ValueError(f"{path} missing cam_K.")
    return np.asarray(data["cam_K"], dtype=np.float64).reshape(3, 3)


def _load_grasp(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ["translation_m_camera", "rotation_matrix_camera"]:
        if key not in data:
            raise ValueError(f"{path} missing {key}.")
    return data


def _unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return fallback.astype(np.float64)
    return arr / norm


def _circle(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, fill: tuple[int, int, int]) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=BLACK)


def _square(draw: ImageDraw.ImageDraw, center: tuple[int, int], size: int, fill: tuple[int, int, int]) -> None:
    x, y = center
    half = size // 2
    draw.rectangle((x - half, y - half, x + half, y + half), fill=fill, outline=BLACK)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: tuple[int, int, int],
    width: int,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    left = (-uy, ux)
    head = 12
    wing = 6
    p1 = (end[0] - ux * head + left[0] * wing, end[1] - uy * head + left[1] * wing)
    p2 = (end[0] - ux * head - left[0] * wing, end[1] - uy * head - left[1] * wing)
    draw.polygon([end, p1, p2], fill=fill)


def _text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
) -> None:
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=BLACK)
    draw.text((x, y), text, fill=fill)


if __name__ == "__main__":
    raise SystemExit(main())
