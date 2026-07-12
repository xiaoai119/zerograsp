"""Generate ManiSkill-friendly Unitree H2 URDF variants."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .h2_constants import H2_STL_URDF, PACKAGE_ROOT


GENERATED_ROOT = PACKAGE_ROOT / "maniskill_unitree_h2" / "generated"
H2_UPPER_BODY_GRIPPER_URDF = GENERATED_ROOT / "h2_upper_body_gripper.urdf"


def ensure_upper_body_gripper_urdf() -> Path:
    """Build a compact H2 upper-body URDF with simple two-finger grippers."""

    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(H2_STL_URDF)
    root = tree.getroot()

    remove_subtrees(root, ("left_hip_pitch_joint", "right_hip_pitch_joint"))
    remove_subtrees(root, ("left_hand_joint", "right_hand_joint"))

    original_links = [child for child in root if child.tag == "link"]
    original_joints = [child for child in root if child.tag == "joint"]
    for child in list(root):
        if child.tag in {"link", "joint"}:
            root.remove(child)

    insert_index = find_insert_index(root)
    root.insert(insert_index, mobile_base_link())
    root.insert(insert_index + 1, fixed_joint("mobile_base_to_pelvis", "mobile_base_link", "pelvis", xyz="0 0 0.56"))
    for element in original_links + original_joints:
        root.append(element)

    for side in ("left", "right"):
        add_simple_gripper(root, side)

    indent(root)
    tree.write(H2_UPPER_BODY_GRIPPER_URDF, encoding="utf-8", xml_declaration=True)
    return H2_UPPER_BODY_GRIPPER_URDF


def remove_subtrees(root: ET.Element, joint_names: tuple[str, ...]) -> None:
    child_by_joint = {}
    joints_by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        child_link = child.attrib["link"]
        child_by_joint[joint.attrib["name"]] = child_link
        joints_by_parent.setdefault(parent.attrib["link"], []).append(joint)

    links_to_remove: set[str] = set()
    joints_to_remove: set[str] = set()
    stack = [child_by_joint[name] for name in joint_names if name in child_by_joint]
    joints_to_remove.update(name for name in joint_names if name in child_by_joint)

    while stack:
        link = stack.pop()
        if link in links_to_remove:
            continue
        links_to_remove.add(link)
        for joint in joints_by_parent.get(link, []):
            joints_to_remove.add(joint.attrib["name"])
            child = joint.find("child")
            if child is not None:
                stack.append(child.attrib["link"])

    for element in list(root):
        if element.tag == "link" and element.attrib.get("name") in links_to_remove:
            root.remove(element)
        elif element.tag == "joint" and element.attrib.get("name") in joints_to_remove:
            root.remove(element)


def find_insert_index(root: ET.Element) -> int:
    for index, child in enumerate(root):
        if child.tag == "link":
            return index
    return len(root)


def mobile_base_link() -> ET.Element:
    link = ET.Element("link", {"name": "mobile_base_link"})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0.12", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "18.0"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.45",
            "ixy": "0",
            "ixz": "0",
            "iyy": "0.45",
            "iyz": "0",
            "izz": "0.70",
        },
    )
    add_box_visual(link, "0 0 0.22", "0.58 0.46 0.22", "0.18 0.18 0.20 1")
    add_box_collision(link, "0 0 0.22", "0.58 0.46 0.22")
    for y in ("0.29", "-0.29"):
        for x in ("0.20", "-0.20"):
            add_cylinder_visual(link, f"{x} {y} 0.09", "1.5708 0 0", "0.08", "0.06", "0.05 0.05 0.05 1")
            add_cylinder_collision(link, f"{x} {y} 0.09", "1.5708 0 0", "0.08", "0.06")
    return link


def add_simple_gripper(root: ET.Element, side: str) -> None:
    sign = 1.0 if side == "left" else -1.0
    wrist = f"{side}_wrist_yaw_link"
    palm = f"{side}_gripper_palm_link"
    left_finger = f"{side}_gripper_inner_finger_link"
    right_finger = f"{side}_gripper_outer_finger_link"

    palm_link = ET.Element("link", {"name": palm})
    add_inertial(palm_link, mass="0.18")
    add_box_visual(palm_link, "0.045 0 0", "0.09 0.07 0.055", "0.08 0.08 0.09 1")
    add_box_collision(palm_link, "0.045 0 0", "0.09 0.07 0.055")
    root.append(palm_link)
    root.append(fixed_joint(f"{side}_gripper_palm_joint", wrist, palm, xyz="0.075 0 0"))

    for name, y, axis in (
        (left_finger, 0.036 * sign, f"0 {sign:.1f} 0"),
        (right_finger, -0.036 * sign, f"0 {-sign:.1f} 0"),
    ):
        finger_link = ET.Element("link", {"name": name})
        add_inertial(finger_link, mass="0.05")
        add_box_visual(finger_link, "0.055 0 0", "0.11 0.018 0.035", "0.12 0.12 0.13 1")
        add_box_collision(finger_link, "0.055 0 0", "0.11 0.018 0.035")
        root.append(finger_link)

        joint = ET.Element("joint", {"name": name.replace("_link", "_joint"), "type": "prismatic"})
        ET.SubElement(joint, "origin", {"xyz": f"0.075 {y:.4f} 0", "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": palm})
        ET.SubElement(joint, "child", {"link": name})
        ET.SubElement(joint, "axis", {"xyz": axis})
        ET.SubElement(joint, "limit", {"lower": "0.0", "upper": "0.035", "effort": "40", "velocity": "0.4"})
        root.append(joint)


def fixed_joint(name: str, parent: str, child: str, xyz: str = "0 0 0", rpy: str = "0 0 0") -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    return joint


def add_inertial(link: ET.Element, mass: str) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": mass})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "0.0001", "ixy": "0", "ixz": "0", "iyy": "0.0001", "iyz": "0", "izz": "0.0001"},
    )


def add_box_visual(link: ET.Element, xyz: str, size: str, color: str) -> None:
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": xyz, "rpy": "0 0 0"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": size})
    material = ET.SubElement(visual, "material", {"name": f"mat_{link.attrib['name']}_{len(link)}"})
    ET.SubElement(material, "color", {"rgba": color})


def add_box_collision(link: ET.Element, xyz: str, size: str) -> None:
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", {"xyz": xyz, "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": size})


def add_cylinder_visual(link: ET.Element, xyz: str, rpy: str, radius: str, length: str, color: str) -> None:
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": xyz, "rpy": rpy})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "cylinder", {"radius": radius, "length": length})
    material = ET.SubElement(visual, "material", {"name": f"mat_{link.attrib['name']}_{len(link)}"})
    ET.SubElement(material, "color", {"rgba": color})


def add_cylinder_collision(link: ET.Element, xyz: str, rpy: str, radius: str, length: str) -> None:
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", {"xyz": xyz, "rpy": rpy})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "cylinder", {"radius": radius, "length": length})


def indent(element: ET.Element, level: int = 0) -> None:
    space = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = space + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = space
    if level and (not element.tail or not element.tail.strip()):
        element.tail = space
