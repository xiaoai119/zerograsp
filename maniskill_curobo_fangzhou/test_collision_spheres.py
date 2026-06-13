"""Structural checks for the generated Lift2 collision-sphere model."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .urdf_adapter import (
    ensure_collision_sphere_urdf,
    expand_collision_sphere_entry,
    is_collision_sphere_visible,
    load_collision_sphere_config,
    load_collision_spheres,
)


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
    config = load_collision_sphere_config()
    styles = config.get("collision_sphere_styles", {})
    expected_count = sum(
        len(items)
        for link_name, items in load_collision_spheres().items()
        if is_collision_sphere_visible(link_name, styles=styles)
    )
    root = ET.parse(
        ensure_collision_sphere_urdf(show_spheres=True)
    ).getroot()
    debug_visuals = [
        visual
        for visual in root.findall(".//visual")
        if visual.get("name", "").startswith("collision_sphere_visual_")
    ]
    assert len(debug_visuals) == expected_count


def test_line_collision_sphere_spec_expands_to_straight_chain() -> None:
    spheres = expand_collision_sphere_entry(
        "test_link",
        {
            "line": {
                "start": [0.0, 0.0, 0.0],
                "end": [0.3, 0.0, 0.0],
                "count": 4,
                "radius": [0.01, 0.04],
            }
        },
    )

    assert spheres == [
        {"center": [0.0, 0.0, 0.0], "radius": 0.01},
        {"center": [0.1, 0.0, 0.0], "radius": 0.02},
        {"center": [0.2, 0.0, 0.0], "radius": 0.03},
        {"center": [0.3, 0.0, 0.0], "radius": 0.04},
    ]
