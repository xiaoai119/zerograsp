#!/usr/bin/env python3
"""Fit cuRobo-style collision spheres to the Lift2 arm meshes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
MESH_ROOT = PACKAGE_ROOT / "urdf" / "lift2" / "meshes"
DEFAULT_OUTPUT = PACKAGE_ROOT / "generated" / "lift2_collision_spheres.yml"
DEFAULT_METRICS = PACKAGE_ROOT / "generated" / "lift2_collision_sphere_metrics.json"

# Long links need a chain of spheres; compact joints and fingers need fewer.
LEFT_LINK_SPHERE_COUNTS = {
    "left_link11": 3,
    "left_link12": 6,
    "left_link13": 6,
    "left_link14": 3,
    "left_link15": 3,
    "left_link16": 4,
    "left_link17": 3,
    "left_link18": 3,
}
RIGHT_LINK_BY_LEFT = {
    f"left_link{left_id}": f"right_link{right_id}"
    for left_id, right_id in zip(range(11, 19), range(21, 29))
}
MANUAL_SPHERE_OVERRIDES = {
    # This compact shoulder shell is too thin for volumetric MorphIt fitting.
    # One conservative sphere covers its complete local-space AABB.
    "left_link11": [
        {
            "center": [0.0078, 0.0, 0.0288],
            "radius": 0.058,
        }
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--radius-padding",
        type=float,
        default=0.003,
        help="Conservative radius padding in metres.",
    )
    return parser.parse_args()


def rounded_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 7) for value in values]


def main() -> int:
    args = parse_args()

    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    collision_spheres: dict[str, list[dict[str, object]]] = {}
    metrics: dict[str, object] = {
        "fit_type": SphereFitType.MORPHIT.value,
        "iterations": args.iterations,
        "radius_padding": args.radius_padding,
        "links": {},
    }

    for link_name, sphere_count in LEFT_LINK_SPHERE_COUNTS.items():
        mesh_path = MESH_ROOT / f"{link_name}.STL"
        mesh = trimesh.load_mesh(mesh_path, process=True)
        if link_name in MANUAL_SPHERE_OVERRIDES:
            result = None
            specs = MANUAL_SPHERE_OVERRIDES[link_name]
        else:
            result = fit_spheres_to_mesh(
                mesh,
                num_spheres=sphere_count,
                fit_type=SphereFitType.MORPHIT,
                iterations=args.iterations,
                compute_metrics=True,
            )
            centers = result.centers.detach().cpu().numpy()
            radii = result.radii.detach().cpu().numpy() + args.radius_padding
            specs = [
                {
                    "center": rounded_list(center),
                    "radius": round(float(radius), 7),
                }
                for center, radius in zip(centers, radii)
            ]
        collision_spheres[link_name] = specs

        right_link_name = RIGHT_LINK_BY_LEFT[link_name]
        collision_spheres[right_link_name] = [
            {"center": list(spec["center"]), "radius": spec["radius"]}
            for spec in specs
        ]

        metrics["links"][link_name] = {
            "mesh": str(mesh_path.resolve()),
            "requested_spheres": sphere_count,
            "generated_spheres": len(specs),
            "fit_time_s": result.fit_time_s if result is not None else 0.0,
            "manual_override": result is None,
            "fit_metrics_before_padding": (
                asdict(result.metrics)
                if result is not None and result.metrics is not None
                else None
            ),
        }
        fit_time = result.fit_time_s if result is not None else 0.0
        print(
            f"{link_name}: requested={sphere_count} generated={len(specs)} "
            f"time={fit_time:.2f}s"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(
            {
                "collision_spheres": collision_spheres,
                "metadata": {
                    "source": "cuRobo MorphIt fit on Lift2 visual meshes",
                    "units": "metres",
                    "radius_padding": args.radius_padding,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    args.metrics.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"collision_spheres={args.output}")
    print(f"metrics={args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
