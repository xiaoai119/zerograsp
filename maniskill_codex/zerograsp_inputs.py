"""Save ManiSkill sensor observations in ZeroGrasp input format."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


MASK_MODES = ("task-target", "all-objects", "visible-area")
YCB_MODEL_ID_RE = re.compile(r"(?P<model_id>\d{3}(?:-[A-Za-z])?(?:_[A-Za-z0-9]+)+)(?:-\d+)?$")


@dataclass(frozen=True)
class ZeroGraspInputBundle:
    """RGB-D, instance mask, and camera metadata for one ZeroGrasp call."""

    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    camera_matrix: np.ndarray
    depth_scale: float
    camera_name: str
    object_records: list[dict[str, Any]]
    mask_mode: str = "unknown"


def extract_zerograsp_input(
    obs: dict[str, Any],
    env: Any,
    camera_name: str,
    min_pixels: int = 1000,
    max_pixels: int = 50000,
    mask_mode: str = "task-target",
) -> ZeroGraspInputBundle:
    """Extract the RGB, depth, mask, and intrinsics that ZeroGrasp consumes."""

    if mask_mode not in MASK_MODES:
        raise ValueError(f"Unsupported mask_mode={mask_mode!r}. Expected one of {MASK_MODES}.")
    sensor = obs["sensor_data"][camera_name]
    rgb = _to_numpy(sensor["Color"])[0, :, :, :3].astype(np.uint8, copy=True)
    position_segmentation = _to_numpy(sensor["PositionSegmentation"])[0].astype(np.float32)
    depth = np.abs(position_segmentation[:, :, 2]).astype(np.float32)
    segmentation = position_segmentation[:, :, 3].astype(np.int32)
    actor_records = None
    if mask_mode != "visible-area":
        actor_records = collect_mask_actor_records(env, mask_mode)
        if not actor_records:
            actor_records = None
    else:
        max_pixels = max(max_pixels, int(segmentation.size) + 1)
    mask, object_records = build_instance_mask(
        segmentation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        actor_records=actor_records,
    )
    if mask_mode == "visible-area":
        mask, object_records = _ensure_background_label(mask, object_records)
    camera_matrix = _to_numpy(
        env.unwrapped.scene.sensors[camera_name].camera.get_intrinsic_matrix()
    )[0].astype(np.float32)
    return ZeroGraspInputBundle(
        rgb=rgb,
        depth=depth,
        mask=mask,
        camera_matrix=camera_matrix,
        depth_scale=1.0,
        camera_name=camera_name,
        object_records=object_records,
        mask_mode=mask_mode,
    )


def build_instance_mask(
    segmentation: np.ndarray,
    min_pixels: int = 1000,
    max_pixels: int = 50000,
    actor_records: Iterable[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Convert ManiSkill segmentation ids to compact ZeroGrasp instance labels."""

    seg = np.asarray(segmentation, dtype=np.int32)
    mask = np.zeros(seg.shape, dtype=np.uint8)
    records: list[dict[str, Any]] = []
    next_label = 1
    candidates = (
        _dedupe_actor_records(actor_records)
        if actor_records is not None
        else [{"segmentation_id": int(v)} for v in np.unique(seg) if int(v) > 0]
    )
    for candidate in candidates:
        uid = int(candidate["segmentation_id"])
        pixels = int(np.count_nonzero(seg == uid))
        if actor_records is None and not (min_pixels < pixels < max_pixels):
            continue
        if actor_records is not None and pixels <= 0:
            continue
        if next_label > np.iinfo(np.uint8).max:
            raise ValueError("Too many object instances for uint8 mask labels.")
        mask[seg == uid] = next_label
        record = {
            "label": next_label,
            "segmentation_id": uid,
            "pixel_count": pixels,
        }
        if actor_records is not None:
            record.update(
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"label", "segmentation_id", "pixel_count"}
                }
            )
        records.append(record)
        next_label += 1
    return mask, records


def collect_mask_actor_records(env: Any, mask_mode: str = "task-target") -> list[dict[str, Any]]:
    """Collect ManiSkill actor ids that should become ZeroGrasp mask instances."""

    if mask_mode not in MASK_MODES:
        raise ValueError(f"Unsupported mask_mode={mask_mode!r}. Expected one of {MASK_MODES}.")
    if mask_mode == "visible-area":
        return []

    root = getattr(env, "unwrapped", env)
    target_records = _collect_task_target_records(root)
    object_records = _collect_scene_object_records(root, target_records)
    target_records = _merge_scene_metadata(target_records, object_records)
    if mask_mode == "task-target" and target_records:
        return target_records

    return object_records or target_records


def save_zerograsp_input_bundle(bundle: ZeroGraspInputBundle, output_dir: str | Path) -> Path:
    """Write one ZeroGrasp input bundle to disk."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    Image.fromarray(_rgb_uint8(bundle.rgb)).save(out / "rgb.png")
    Image.fromarray(_depth_uint16(bundle.depth)).save(out / "depth.png")
    Image.fromarray(_mask_uint8(bundle.mask)).save(out / "mask.png")
    np.savez_compressed(
        out / "rgbd.npz",
        rgb=_rgb_uint8(bundle.rgb),
        depth=bundle.depth.astype(np.float32),
        mask=_mask_uint8(bundle.mask),
        cam_K=np.asarray(bundle.camera_matrix, dtype=np.float32).reshape(3, 3),
    )

    camera = {
        "cam_K": np.asarray(bundle.camera_matrix, dtype=np.float32).reshape(3, 3).ravel().tolist(),
        "depth_scale": float(bundle.depth_scale),
        "camera_name": bundle.camera_name,
        "mask_mode": bundle.mask_mode,
        "objects": bundle.object_records,
    }
    (out / "camera.json").write_text(
        json.dumps(camera, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.shape[-1] != 3:
        raise ValueError(f"RGB image must have 3 channels, got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _depth_uint16(depth: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rint(np.clip(depth, 0, 65535)).astype(np.uint16))


def _mask_uint8(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip(mask, 0, 255).astype(np.uint8))


def _ensure_background_label(
    mask: np.ndarray,
    object_records: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = np.asarray(mask, dtype=np.uint8).copy()
    if out.size == 0 or np.any(out == 0):
        return out, object_records
    removed_label = int(out.flat[0])
    out.flat[0] = 0
    records = [dict(record) for record in object_records]
    for record in records:
        if int(record.get("label", -1)) == removed_label:
            record["pixel_count"] = max(int(record.get("pixel_count", 0)) - 1, 0)
            break
    return out, records


def _collect_task_target_records(root: Any) -> list[dict[str, Any]]:
    if hasattr(root, "target_object"):
        return _dedupe_actor_records(
            _actor_records_from_value(root.target_object, "target_object", is_task_target=True)
        )

    records: list[dict[str, Any]] = []
    if hasattr(root, "_objs"):
        records.extend(_actor_records_from_value(root._objs, "_objs", is_task_target=True))
    if hasattr(root, "obj"):
        records.extend(_actor_records_from_value(root.obj, "obj", is_task_target=True))
    return _dedupe_actor_records(records)


def _collect_scene_object_records(
    root: Any,
    target_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_ids = {int(record["segmentation_id"]) for record in target_records}
    hidden_ids = _hidden_actor_ids(root)
    scene = getattr(root, "scene", None)
    get_all_actors = getattr(scene, "get_all_actors", None)
    if not callable(get_all_actors):
        return []

    records: list[dict[str, Any]] = []
    for actor in get_all_actors():
        segmentation_id = _actor_segmentation_id(actor)
        if segmentation_id is None or segmentation_id in hidden_ids:
            continue
        name = _actor_name(actor)
        if _is_excluded_scene_actor(name):
            continue
        records.append(
            _actor_record(
                segmentation_id=segmentation_id,
                actor_name=name,
                source="scene.get_all_actors",
                is_task_target=segmentation_id in target_ids,
            )
        )
    return _dedupe_actor_records(records)


def _actor_records_from_value(
    value: Any,
    source: str,
    is_task_target: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for actor in _iter_actor_like(value):
        segmentation_id = _actor_segmentation_id(actor)
        if segmentation_id is None:
            continue
        records.append(
            _actor_record(
                segmentation_id=segmentation_id,
                actor_name=_actor_name(actor),
                source=source,
                is_task_target=is_task_target,
            )
        )
    return records


def _iter_actor_like(value: Any):
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_actor_like(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_actor_like(item)
        return
    yield value


def _actor_segmentation_id(actor: Any) -> int | None:
    raw = getattr(actor, "per_scene_id", None)
    if raw is None:
        raw = getattr(actor, "id", None)
    if raw is None:
        return None
    arr = np.asarray(_to_numpy(raw)).reshape(-1)
    if arr.size == 0:
        return None
    segmentation_id = int(arr[0])
    return segmentation_id if segmentation_id > 0 else None


def _actor_name(actor: Any) -> str:
    name = getattr(actor, "name", None)
    return str(name) if name is not None else actor.__class__.__name__


def _hidden_actor_ids(root: Any) -> set[int]:
    ids: set[int] = set()
    for actor in _iter_actor_like(getattr(root, "_hidden_objects", [])):
        segmentation_id = _actor_segmentation_id(actor)
        if segmentation_id is not None:
            ids.add(segmentation_id)
    return ids


def _is_excluded_scene_actor(name: str) -> bool:
    lowered = name.lower()
    excluded_tokens = (
        "table",
        "ground",
        "goal",
        "floor",
        "wall",
        "camera",
        "robot",
        "panda",
        "link",
    )
    return any(token in lowered for token in excluded_tokens)


def _actor_record(
    segmentation_id: int,
    actor_name: str,
    source: str,
    is_task_target: bool,
) -> dict[str, Any]:
    record = {
        "segmentation_id": int(segmentation_id),
        "actor_name": actor_name,
        "source": source,
        "is_task_target": bool(is_task_target),
    }
    record.update(_semantic_fields_from_actor_name(actor_name))
    return record


def _merge_scene_metadata(
    target_records: Iterable[dict[str, Any]],
    object_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    objects_by_id = {int(record["segmentation_id"]): record for record in object_records}
    merged_records: list[dict[str, Any]] = []
    for target_record in target_records:
        segmentation_id = int(target_record["segmentation_id"])
        scene_record = objects_by_id.get(segmentation_id)
        if scene_record is None:
            merged = dict(target_record)
        else:
            merged = dict(target_record)
            for key, value in scene_record.items():
                if key not in {"label", "pixel_count", "segmentation_id", "is_task_target"}:
                    merged[key] = value
            if target_record.get("source") and scene_record.get("source"):
                merged["source"] = f"{target_record['source']}+{scene_record['source']}"
        merged["segmentation_id"] = segmentation_id
        merged["is_task_target"] = True
        merged_records.append(merged)
    return _dedupe_actor_records(merged_records)


def _semantic_fields_from_actor_name(actor_name: str) -> dict[str, str]:
    match = YCB_MODEL_ID_RE.search(actor_name)
    if match is None:
        return {}
    model_id = match.group("model_id")
    category = re.sub(r"^\d{3}(?:-[A-Za-z])?_", "", model_id)
    fields = {"model_id": model_id}
    if category:
        fields["category"] = category
        fields["display_name"] = category.replace("_", " ")
    return fields


def _dedupe_actor_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        segmentation_id = int(record["segmentation_id"])
        if segmentation_id not in by_id:
            by_id[segmentation_id] = dict(record, segmentation_id=segmentation_id)
            continue
        existing = by_id[segmentation_id]
        existing["is_task_target"] = bool(existing.get("is_task_target")) or bool(
            record.get("is_task_target")
        )
        if not existing.get("actor_name") and record.get("actor_name"):
            existing["actor_name"] = record["actor_name"]
    return [by_id[key] for key in sorted(by_id)]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
