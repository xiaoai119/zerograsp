"""Prepare the exported Lift2 URDF for a first ManiSkill visual smoke test."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

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
    if not COLLISION_SPHERES.exists():
        raise FileNotFoundError(
            f"{COLLISION_SPHERES} is missing; run "
            "`python -m maniskill_curobo_fangzhou.fit_lift2_collision_spheres`."
        )
    data = yaml.safe_load(COLLISION_SPHERES.read_text(encoding="utf-8"))
    return data["collision_spheres"]


def ensure_collision_sphere_urdf(*, show_spheres: bool = False) -> Path:
    """Generate a URDF whose arm collisions are sphere approximations."""

    output_path = COLLISION_DEBUG_URDF if show_spheres else COLLISION_URDF
    newest_input = max(
        SOURCE_URDF.stat().st_mtime_ns,
        COLLISION_SPHERES.stat().st_mtime_ns,
    )
    if output_path.exists() and output_path.stat().st_mtime_ns >= newest_input:
        return output_path

    tree = _load_adapted_tree()
    links = {link.get("name"): link for link in tree.getroot().findall("link")}
    sphere_specs = load_collision_spheres()

    for link_name, specs in sphere_specs.items():
        link = links[link_name]
        side_color = "0.1 0.45 1.0 0.38" if link_name.startswith("left_") else "1.0 0.25 0.1 0.38"
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

            if show_spheres:
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
                ET.SubElement(material, "color", {"rgba": side_color})

    return _write_tree(tree, output_path)
