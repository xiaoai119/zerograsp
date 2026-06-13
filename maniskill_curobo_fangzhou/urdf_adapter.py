"""Prepare the exported Lift2 URDF for a first ManiSkill visual smoke test."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_URDF = PACKAGE_ROOT / "urdf" / "lift2" / "urdf" / "lift2.urdf"
GENERATED_DIR = PACKAGE_ROOT / "generated"
VISUAL_URDF = GENERATED_DIR / "lift2_maniskill_visual.urdf"
COLLISION_SPHERES = GENERATED_DIR / "lift2_collision_spheres.yml"
COLLISION_URDF = GENERATED_DIR / "lift2_maniskill_collision_spheres.urdf"
COLLISION_DEBUG_URDF = (
    GENERATED_DIR / "lift2_maniskill_collision_spheres_debug.urdf"
)


def _load_adapted_tree() -> ET.ElementTree:
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    mesh_root = PACKAGE_ROOT / "urdf" / "lift2" / "meshes"

    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("package://lift2/meshes/"):
            mesh_name = filename.removeprefix("package://lift2/meshes/")
            mesh.set("filename", str((mesh_root / mesh_name).resolve()))

    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
    return tree


def _write_tree(tree: ET.ElementTree, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def ensure_visual_urdf() -> Path:
    """Generate a collision-free URDF with filesystem-relative mesh paths."""

    if (
        VISUAL_URDF.exists()
        and VISUAL_URDF.stat().st_mtime_ns >= SOURCE_URDF.stat().st_mtime_ns
    ):
        return VISUAL_URDF
    return _write_tree(_load_adapted_tree(), VISUAL_URDF)


def load_collision_spheres() -> dict[str, list[dict[str, object]]]:
    data = load_collision_sphere_config()
    return normalize_collision_sphere_specs(data["collision_spheres"])


def load_collision_sphere_config() -> dict[str, object]:
    if not COLLISION_SPHERES.exists():
        raise FileNotFoundError(
            f"{COLLISION_SPHERES} is missing; run "
            "`python -m maniskill_curobo_fangzhou.fit_lift2_collision_spheres`."
        )
    return yaml.safe_load(COLLISION_SPHERES.read_text(encoding="utf-8"))


def ensure_collision_sphere_urdf(*, show_spheres: bool = False) -> Path:
    """Generate a URDF whose arm collisions are sphere approximations."""

    output_path = COLLISION_DEBUG_URDF if show_spheres else COLLISION_URDF
    newest_input = max(
        SOURCE_URDF.stat().st_mtime_ns,
        COLLISION_SPHERES.stat().st_mtime_ns,
        Path(__file__).stat().st_mtime_ns,
    )
    if output_path.exists() and output_path.stat().st_mtime_ns >= newest_input:
        return output_path

    tree = _load_adapted_tree()
    links = {link.get("name"): link for link in tree.getroot().findall("link")}
    sphere_config = load_collision_sphere_config()
    sphere_specs = normalize_collision_sphere_specs(sphere_config["collision_spheres"])
    palette = sphere_config.get("color_palette", {})
    styles = sphere_config.get("collision_sphere_styles", {})

    for link_name, specs in sphere_specs.items():
        link = links[link_name]
        sphere_color = resolve_collision_sphere_color(
            link_name,
            palette=palette,
            styles=styles,
        )
        for index, spec in enumerate(specs):
            center = " ".join(str(value) for value in spec["center"])
            radius = str(spec["radius"])

            collision = ET.SubElement(
                link,
                "collision",
                {"name": f"collision_sphere_{index:02d}"},
            )
            ET.SubElement(collision, "origin", {"xyz": center, "rpy": "0 0 0"})
            collision_geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(collision_geometry, "sphere", {"radius": radius})

            if show_spheres and is_collision_sphere_visible(
                link_name,
                styles=styles,
            ):
                visual = ET.SubElement(
                    link,
                    "visual",
                    {"name": f"collision_sphere_visual_{index:02d}"},
                )
                ET.SubElement(visual, "origin", {"xyz": center, "rpy": "0 0 0"})
                visual_geometry = ET.SubElement(visual, "geometry")
                ET.SubElement(visual_geometry, "sphere", {"radius": radius})
                material = ET.SubElement(
                    visual,
                    "material",
                    {"name": f"{link_name}_collision_sphere"},
                )
                ET.SubElement(material, "color", {"rgba": sphere_color})

    return _write_tree(tree, output_path)


def normalize_collision_sphere_specs(
    raw_specs: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    return {
        link_name: expand_collision_sphere_entry(link_name, entry)
        for link_name, entry in raw_specs.items()
    }


def expand_collision_sphere_entry(
    link_name: str,
    entry: object,
) -> list[dict[str, object]]:
    if isinstance(entry, list):
        return [
            normalize_explicit_sphere(link_name, sphere, index=index)
            for index, sphere in enumerate(entry)
        ]
    if not isinstance(entry, dict):
        raise TypeError(
            f"Invalid collision sphere entry for {link_name}: expected list or dict, "
            f"got {type(entry).__name__}."
        )
    if "line" in entry:
        return expand_line_spheres(link_name, entry["line"])
    if {"start", "end", "count", "radius"}.issubset(entry.keys()):
        return expand_line_spheres(link_name, entry)
    if "segments" in entry:
        segments = entry["segments"]
        if not isinstance(segments, list):
            raise TypeError(f"`segments` for {link_name} must be a list.")
        spheres: list[dict[str, object]] = []
        for segment in segments:
            spheres.extend(expand_line_spheres(link_name, segment))
        return spheres
    if "spheres" in entry:
        spheres = entry["spheres"]
        if not isinstance(spheres, list):
            raise TypeError(f"`spheres` for {link_name} must be a list.")
        return [
            normalize_explicit_sphere(link_name, sphere, index=index)
            for index, sphere in enumerate(spheres)
        ]
    raise ValueError(
        f"Invalid collision sphere entry for {link_name}. Use either an explicit "
        "sphere list, `line: {start, end, count, radius}`, or `segments`."
    )


def normalize_explicit_sphere(
    link_name: str,
    sphere: object,
    *,
    index: int,
) -> dict[str, object]:
    if not isinstance(sphere, dict):
        raise TypeError(f"Sphere {index} for {link_name} must be a mapping.")
    center = parse_vec3(sphere.get("center"), f"{link_name}[{index}].center")
    radius = parse_positive_float(sphere.get("radius"), f"{link_name}[{index}].radius")
    return {"center": center, "radius": radius}


def expand_line_spheres(
    link_name: str,
    line: object,
) -> list[dict[str, object]]:
    if not isinstance(line, dict):
        raise TypeError(f"Line collision sphere spec for {link_name} must be a mapping.")
    start = parse_vec3(line.get("start"), f"{link_name}.line.start")
    end = parse_vec3(line.get("end"), f"{link_name}.line.end")
    count = int(line.get("count", 0))
    if count < 1:
        raise ValueError(f"{link_name}.line.count must be >= 1.")
    radii = expand_radii(line.get("radius"), count, f"{link_name}.line.radius")

    spheres: list[dict[str, object]] = []
    for index in range(count):
        alpha = 0.0 if count == 1 else index / float(count - 1)
        center = [
            round(start[axis] + (end[axis] - start[axis]) * alpha, 7)
            for axis in range(3)
        ]
        spheres.append({"center": center, "radius": radii[index]})
    return spheres


def parse_vec3(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3D list like [x, y, z].")
    return [round(float(component), 7) for component in value]


def parse_positive_float(value: Any, name: str) -> float:
    number = round(float(value), 7)
    if number <= 0:
        raise ValueError(f"{name} must be > 0.")
    return number


def expand_radii(value: Any, count: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        radius = parse_positive_float(value, name)
        return [radius] * count
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a number, [start, end], or a list of {count} values.")
    if len(value) == count:
        return [
            parse_positive_float(radius, f"{name}[{index}]")
            for index, radius in enumerate(value)
        ]
    if len(value) == 2:
        start = parse_positive_float(value[0], f"{name}[0]")
        end = parse_positive_float(value[1], f"{name}[1]")
        return [
            round(start + (end - start) * (0.0 if count == 1 else index / float(count - 1)), 7)
            for index in range(count)
        ]
    raise ValueError(f"{name} must be a number, [start, end], or a list of {count} values.")


def is_collision_sphere_visible(
    link_name: str,
    *,
    styles: dict[str, object],
) -> bool:
    style = styles.get(link_name, {}) if isinstance(styles, dict) else {}
    if not isinstance(style, dict):
        return False
    return bool(style.get("visible", False))


def resolve_collision_sphere_color(
    link_name: str,
    *,
    palette: dict[str, object],
    styles: dict[str, object],
) -> str:
    default_color = "0.1 0.45 1.0 0.38" if link_name.startswith("left_") else "1.0 0.25 0.1 0.38"
    style = styles.get(link_name, {}) if isinstance(styles, dict) else {}
    color = style.get("color") if isinstance(style, dict) else None
    if color is None:
        return default_color
    if isinstance(color, str):
        color = palette.get(color, color)
    if isinstance(color, str):
        return color
    if isinstance(color, (list, tuple)) and len(color) == 4:
        return " ".join(str(float(value)) for value in color)
    raise ValueError(
        f"Invalid collision sphere color for {link_name}: {color!r}. "
        "Use a palette name or an RGBA list of four numbers."
    )
