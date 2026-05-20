"""Shared camera view defaults and CLI helpers."""

from __future__ import annotations

import argparse
from typing import Iterable


DEFAULT_CAMERA_EYE = (-0.30, 0.0, 0.55)
DEFAULT_CAMERA_TARGET = (0.05, 0.0, 0.08)


def add_camera_view_args(parser: argparse.ArgumentParser) -> None:
    """Add shared camera extrinsic arguments to a command-line parser."""

    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=list(DEFAULT_CAMERA_EYE),
        metavar=("X", "Y", "Z"),
        help="World-frame camera position for the ZeroGrasp sensor.",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=list(DEFAULT_CAMERA_TARGET),
        metavar=("X", "Y", "Z"),
        help="World-frame point the ZeroGrasp sensor looks at.",
    )


def camera_view_cli_args(camera_eye: Iterable[float], camera_target: Iterable[float]) -> list[str]:
    """Return CLI args for forwarding a camera eye/target pair."""

    return [
        "--camera-eye",
        *[str(float(v)) for v in camera_eye],
        "--camera-target",
        *[str(float(v)) for v in camera_target],
    ]
