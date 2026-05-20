"""Load offline ZeroGrasp grasp outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraspRecord:
    """One ZeroGrasp grasp candidate in OpenCV camera coordinates."""

    score: float
    width_m: float
    height_m: float
    depth_m: float
    rotation_matrix_camera: np.ndarray
    translation_m_camera: np.ndarray
    source: str
    object_index: int | None = None
    object_id: int | None = None


def load_best_grasp(output_dir: str | Path) -> GraspRecord:
    """Load the best grasp from a ZeroGrasp output directory.

    The preferred input is ``recommended_grasp_top1.json``. If it is absent,
    the loader scans ``raw_outputs/*.grasp.npy`` and returns the highest-score
    row across all files.
    """

    root = Path(output_dir).expanduser().resolve()
    json_path = root / "recommended_grasp_top1.json"
    if json_path.is_file():
        return _load_recommended_json(json_path)

    grasp_files = sorted((root / "raw_outputs").glob("*.grasp.npy"))
    if not grasp_files:
        grasp_files = sorted(root.glob("*.grasp.npy"))
    if not grasp_files:
        raise FileNotFoundError(
            f"No ZeroGrasp grasp output found under {root}. Expected "
            "recommended_grasp_top1.json or raw_outputs/*.grasp.npy."
        )

    best: GraspRecord | None = None
    for grasp_path in grasp_files:
        arr = np.asarray(np.load(grasp_path), dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 16:
            raise ValueError(
                f"{grasp_path} has shape {arr.shape}; expected rows with at least 16 values."
            )

        row = arr[int(np.argmax(arr[:, 0]))]
        source = _relative_source(grasp_path, root)
        record = _record_from_row(row, source=source, object_index=_parse_object_index(grasp_path))
        if best is None or record.score > best.score:
            best = record

    if best is None:
        raise ValueError(f"All grasp arrays under {root} are empty.")
    return best


def _load_recommended_json(path: Path) -> GraspRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GraspRecord(
        score=float(_required(data, "score", path)),
        width_m=float(_required(data, "width_m", path)),
        height_m=float(_required(data, "height_m", path)),
        depth_m=float(_required(data, "depth_m", path)),
        rotation_matrix_camera=_matrix3(_required(data, "rotation_matrix_camera", path), path),
        translation_m_camera=_vector3(_required(data, "translation_m_camera", path), path),
        source=path.name,
        object_index=_optional_int(data.get("object_index")),
        object_id=_optional_int(data.get("object_id")),
    )


def _record_from_row(row: np.ndarray, source: str, object_index: int | None) -> GraspRecord:
    return GraspRecord(
        score=float(row[0]),
        width_m=float(row[1]),
        height_m=float(row[2]),
        depth_m=float(row[3]),
        rotation_matrix_camera=np.asarray(row[4:13], dtype=np.float64).reshape(3, 3),
        translation_m_camera=np.asarray(row[13:16], dtype=np.float64).reshape(3),
        source=source,
        object_index=object_index,
        object_id=_optional_int(row[16]) if row.shape[0] > 16 else None,
    )


def _required(data: dict[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise ValueError(f"{path} is missing required field {key!r}.")
    return data[key]


def _matrix3(value: Any, path: Path) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{path} rotation_matrix_camera must have shape (3, 3), got {arr.shape}.")
    return arr


def _vector3(value: Any, path: Path) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{path} translation_m_camera must have shape (3,), got {arr.shape}.")
    return arr


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    value_int = int(value)
    return value_int if value_int >= 0 else None


def _parse_object_index(path: Path) -> int | None:
    stem = path.stem
    if stem.endswith(".grasp"):
        stem = stem[: -len(".grasp")]
    maybe_idx = stem.split("_")[-1]
    return int(maybe_idx) if maybe_idx.isdigit() else None


def _relative_source(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
