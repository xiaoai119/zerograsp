"""Export ManiSkill scene collision geometry as a cuRobo scene model."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MIN_CUBOID_DIMENSION = 0.005


@dataclass(frozen=True)
class SceneExport:
    scene_path: Path
    metadata_path: Path
    scene: dict[str, Any]
    records: list[dict[str, Any]]


def maniskill_scene_to_curobo_dict(
    env: Any,
    world_from_base_matrix: np.ndarray,
    *,
    exclude_segmentation_ids: Iterable[int] = (),
    exclude_actor_names: Iterable[str] = (),
    min_cuboid_dimension: float = DEFAULT_MIN_CUBOID_DIMENSION,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert visible ManiSkill collision shapes to cuRobo cuboid obstacles.

    The returned poses are in the robot base frame because cuRobo plans in the
    same frame as the robot model. Convex mesh collision shapes are represented
    by conservative local bounding cuboids.
    """

    root = getattr(env, "unwrapped", env)
    scene = getattr(root, "scene", None)
    get_all_actors = getattr(scene, "get_all_actors", None)
    if not callable(get_all_actors):
        raise ValueError("ManiSkill scene does not expose get_all_actors().")

    base_from_world = np.linalg.inv(_matrix4(world_from_base_matrix, "world_from_base_matrix"))
    excluded_ids = {int(v) for v in exclude_segmentation_ids}
    excluded_names = {str(v) for v in exclude_actor_names}

    cuboids: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for actor in get_all_actors():
        actor_name = _actor_name(actor)
        segmentation_id = _actor_segmentation_id(actor)
        if _skip_actor(actor_name, segmentation_id, excluded_ids, excluded_names):
            continue
        for component in _iter_collision_components(actor):
            component_matrix = base_from_world @ _pose_to_matrix(_component_pose(component))
            component_name = _component_name(component)
            for shape_index, shape in enumerate(_collision_shapes(component)):
                obstacle = _shape_to_cuboid(
                    shape,
                    component_matrix,
                    min_cuboid_dimension=min_cuboid_dimension,
                )
                if obstacle is None:
                    continue
                shape_type = obstacle.pop("_shape_type")
                name = _unique_name(
                    _sanitize_name(f"{actor_name}_{shape_type}_{shape_index:02d}"),
                    used_names,
                )
                cuboids[name] = obstacle
                records.append(
                    {
                        "obstacle_name": name,
                        "actor_name": actor_name,
                        "segmentation_id": segmentation_id,
                        "component_name": component_name,
                        "shape_index": shape_index,
                        "shape_type": shape_type,
                        "dims": obstacle["dims"],
                        "pose": obstacle["pose"],
                    }
                )

    return {"cuboid": cuboids}, records


def export_maniskill_scene_to_curobo(
    env: Any,
    scene_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    world_from_base_matrix: np.ndarray,
    exclude_segmentation_ids: Iterable[int] = (),
    exclude_actor_names: Iterable[str] = (),
    min_cuboid_dimension: float = DEFAULT_MIN_CUBOID_DIMENSION,
) -> SceneExport:
    """Write a per-run cuRobo scene model plus a human-readable metadata file."""

    scene, records = maniskill_scene_to_curobo_dict(
        env,
        world_from_base_matrix,
        exclude_segmentation_ids=exclude_segmentation_ids,
        exclude_actor_names=exclude_actor_names,
        min_cuboid_dimension=min_cuboid_dimension,
    )
    out = Path(scene_path).expanduser().resolve()
    meta = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else out.with_name(f"{out.stem}_metadata.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    meta.write_text(
        json.dumps(
            {
                "scene_path": str(out),
                "n_obstacles": len(records),
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SceneExport(scene_path=out, metadata_path=meta, scene=scene, records=records)


def target_segmentation_ids_from_zerograsp_scene(
    scene_json_path: str | Path,
    object_id: int | None,
) -> set[int]:
    """Map a ZeroGrasp compact object label back to ManiSkill segmentation ids."""

    path = Path(scene_json_path).expanduser().resolve()
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        return set()

    if object_id is not None:
        matches = {
            int(record["segmentation_id"])
            for record in objects
            if isinstance(record, dict)
            and _optional_int(record.get("label")) == int(object_id)
            and _optional_int(record.get("segmentation_id")) is not None
        }
        if matches:
            return matches

    return {
        int(record["segmentation_id"])
        for record in objects
        if isinstance(record, dict)
        and bool(record.get("is_task_target"))
        and _optional_int(record.get("segmentation_id")) is not None
    }


def _shape_to_cuboid(
    shape: Any,
    base_from_component_matrix: np.ndarray,
    *,
    min_cuboid_dimension: float,
) -> dict[str, Any] | None:
    half_size = _shape_half_size(shape)
    if half_size is not None:
        local_matrix = _pose_to_matrix(_shape_local_pose(shape))
        dims = _clamp_dims(2.0 * half_size, min_cuboid_dimension)
        matrix = base_from_component_matrix @ local_matrix
        return {
            "_shape_type": "box",
            "dims": _round_list(dims),
            "pose": _round_list(_pose_from_matrix(matrix)),
        }

    vertices = _shape_vertices(shape)
    if vertices is None or vertices.size == 0:
        return None

    scale = _shape_scale(shape)
    vertices = vertices * scale.reshape(1, 3)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    dims = _clamp_dims(maxs - mins, min_cuboid_dimension)
    center = 0.5 * (mins + maxs)
    local_matrix = _pose_to_matrix(_shape_local_pose(shape)) @ _translation_matrix(center)
    matrix = base_from_component_matrix @ local_matrix
    return {
        "_shape_type": "convex_mesh",
        "dims": _round_list(dims),
        "pose": _round_list(_pose_from_matrix(matrix)),
    }


def _iter_collision_components(actor: Any):
    seen: set[int] = set()
    for source in (getattr(actor, "components", None), _call_noarg(actor, "get_components")):
        if source is None:
            continue
        values = source.values() if isinstance(source, dict) else source
        for component in values:
            if id(component) in seen:
                continue
            seen.add(id(component))
            if callable(getattr(component, "get_collision_shapes", None)):
                yield component
    for component in getattr(actor, "_bodies", []) or []:
        if id(component) in seen:
            continue
        seen.add(id(component))
        if callable(getattr(component, "get_collision_shapes", None)):
            yield component


def _collision_shapes(component: Any) -> list[Any]:
    try:
        return list(component.get_collision_shapes())
    except Exception:
        return []


def _skip_actor(
    actor_name: str,
    segmentation_id: int | None,
    excluded_ids: set[int],
    excluded_names: set[str],
) -> bool:
    if actor_name in excluded_names:
        return True
    if segmentation_id is not None and segmentation_id in excluded_ids:
        return True
    lowered = actor_name.lower()
    return (
        "goal_site" in lowered
        or "grasp_marker" in lowered
        or lowered.startswith("zg_")
        or lowered.startswith("marker_")
    )


def _component_pose(component: Any) -> Any:
    for attr in ("entity_pose", "pose"):
        pose = getattr(component, attr, None)
        if pose is not None:
            return pose
    for method in ("get_entity_pose", "get_pose"):
        pose = _call_noarg(component, method)
        if pose is not None:
            return pose
    return _identity_pose()


def _shape_local_pose(shape: Any) -> Any:
    pose = getattr(shape, "local_pose", None)
    if pose is not None:
        return pose
    pose = _call_noarg(shape, "get_local_pose")
    if pose is not None:
        return pose
    return _identity_pose()


def _shape_half_size(shape: Any) -> np.ndarray | None:
    value = getattr(shape, "half_size", None)
    if value is None:
        value = _call_noarg(shape, "get_half_size")
    if value is None:
        return None
    return _vector(value, 3, "half_size")


def _shape_vertices(shape: Any) -> np.ndarray | None:
    value = getattr(shape, "vertices", None)
    if value is None:
        value = _call_noarg(shape, "get_vertices")
    if value is None:
        return None
    vertices = _numpy(value, "vertices").reshape(-1, 3)
    return vertices.astype(np.float64, copy=False)


def _shape_scale(shape: Any) -> np.ndarray:
    value = getattr(shape, "scale", None)
    if value is None:
        value = _call_noarg(shape, "get_scale")
    if value is None:
        return np.ones(3, dtype=np.float64)
    return _vector(value, 3, "scale")


def _pose_to_matrix(pose: Any) -> np.ndarray:
    raw = getattr(pose, "raw_pose", None)
    if raw is not None:
        values = _vector(raw, 7, "raw_pose")
        p = values[:3]
        q = values[3:]
    else:
        p = _vector(getattr(pose, "p", [0.0, 0.0, 0.0]), 3, "pose.p")
        q = _vector(getattr(pose, "q", [1.0, 0.0, 0.0, 0.0]), 4, "pose.q")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_wxyz_to_matrix(q)
    matrix[:3, 3] = p
    return matrix


def _pose_from_matrix(matrix: np.ndarray) -> np.ndarray:
    mat = _matrix4(matrix, "matrix")
    quat = _matrix_to_quat_wxyz(mat[:3, :3])
    return np.concatenate([mat[:3, 3], quat])


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _unit(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        return _unit(np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]))
    index = int(np.argmax(np.diag(r)))
    if index == 0:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    elif index == 1:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    return _unit(q)


def _translation_matrix(offset: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = _vector(offset, 3, "offset")
    return matrix


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}.")
    return matrix


def _clamp_dims(dims: np.ndarray, minimum: float) -> np.ndarray:
    return np.maximum(np.asarray(dims, dtype=np.float64).reshape(3), float(minimum))


def _unit(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return arr / norm


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    arr = _numpy(value, name).reshape(-1)
    if arr.size < length:
        raise ValueError(f"{name} must contain at least {length} values, got {arr.size}.")
    return arr[:length].astype(np.float64, copy=False)


def _numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.dtype == object:
        raise ValueError(f"{name} cannot be converted to a numeric array.")
    return arr


def _round_list(values: np.ndarray, digits: int = 8) -> list[float]:
    return [float(round(v, digits)) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def _actor_name(actor: Any) -> str:
    return str(getattr(actor, "name", "") or actor.__class__.__name__)


def _component_name(component: Any) -> str:
    return str(getattr(component, "name", "") or component.__class__.__name__)


def _actor_segmentation_id(actor: Any) -> int | None:
    raw = getattr(actor, "per_scene_id", None)
    if raw is None:
        raw = getattr(actor, "id", None)
    if raw is None:
        return None
    try:
        value = int(_numpy(raw, "segmentation_id").reshape(-1)[0])
    except Exception:
        return None
    return value if value > 0 else None


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return cleaned or "obstacle"


def _unique_name(base: str, used_names: set[str]) -> str:
    if base not in used_names:
        used_names.add(base)
        return base
    index = 1
    while f"{base}_{index}" in used_names:
        index += 1
    name = f"{base}_{index}"
    used_names.add(name)
    return name


def _call_noarg(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _identity_pose() -> Any:
    class IdentityPose:
        p = np.zeros(3, dtype=np.float64)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return IdentityPose()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
