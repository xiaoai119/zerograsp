"""Structural checks for the generated Lift2 collision-sphere model."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .urdf_adapter import ensure_collision_sphere_urdf, load_collision_spheres


def test_collision_spheres_match_generated_urdf() -> None:
    specs = load_collision_spheres()
    expected_count = sum(len(items) for items in specs.values())

    root = ET.parse(ensure_collision_sphere_urdf()).getroot()
    collisions = root.findall(".//collision")
    spheres = root.findall(".//collision/geometry/sphere")

    assert len(specs) == 16
    assert expected_count > 0
    assert len(collisions) == expected_count
    assert len(spheres) == expected_count
    assert all(float(sphere.get("radius", "0")) > 0 for sphere in spheres)


def test_debug_urdf_has_one_visual_per_collision_sphere() -> None:
    expected_count = sum(len(items) for items in load_collision_spheres().values())
    root = ET.parse(
        ensure_collision_sphere_urdf(show_spheres=True)
    ).getroot()
    debug_visuals = [
        visual
        for visual in root.findall(".//visual")
        if visual.get("name", "").startswith("collision_sphere_visual_")
    ]
    assert len(debug_visuals) == expected_count
