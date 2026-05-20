"""ZeroGrasp HTTP service."""

from __future__ import annotations

import io
import json
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from zerograsp.pipeline import ZeroGraspPipeline
from zerograsp.utils.dataset import extract_camera_matrix

_pipeline: Optional[ZeroGraspPipeline] = None

CONFIG_PATHS = {
    "demo": "configs/demo.yaml",
    "maniskill": "configs/maniskill.yaml",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    package_root = Path(__file__).resolve().parent
    _pipeline = ZeroGraspPipeline(
        checkpoint_path=str(package_root / "checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt"),
        config_path=str(package_root / CONFIG_PATHS["demo"]),
    )
    yield


app = FastAPI(lifespan=lifespan, title="ZeroGrasp Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    mask: UploadFile = File(...),
    camera: UploadFile = File(...),
    depth_scale: float = Form(1.0),
    return_points: bool = Form(False),
    collision_detection: bool = Form(True),
):
    try:
        rgb_bytes = await rgb.read()
        depth_bytes = await depth.read()
        mask_bytes = await mask.read()
        camera_bytes = await camera.read()

        rgb_img = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
        depth_img = np.asarray(iio.imread(depth_bytes), dtype=np.float32)
        mask_img = np.asarray(Image.open(io.BytesIO(mask_bytes)))
        camera_text = camera_bytes.decode("utf-8")

        # Parse camera: try JSON first, then YAML
        try:
            K = np.array(json.loads(camera_text)["cam_K"], dtype=np.float32).reshape(3, 3)
        except (json.JSONDecodeError, KeyError):
            K = extract_camera_matrix(yaml.safe_load(camera_text)).astype(np.float32).reshape(3, 3)

        # Temporarily override collision detection setting
        saved = _pipeline._config.use_collision_detection
        _pipeline._config.use_collision_detection = collision_detection
        try:
            result = _pipeline.predict(rgb_img, depth_img, mask_img, K, depth_scale=depth_scale)
        finally:
            _pipeline._config.use_collision_detection = saved

        response = {
            "runtime_sec": round(result.runtime_sec, 4),
            "n_objects": len(result.objects),
            "objects": [],
        }

        recommended = result.recommended_grasp()
        if recommended is not None:
            response["recommended_grasp"] = _grasp_to_json(recommended)

        for obj in result.objects:
            obj_entry = {
                "object_id": obj.object_id,
                "object_index": obj.object_index,
                "n_points": int(obj.point_cloud.shape[0]),
                "n_grasps_before_collision": obj.n_grasps_before_collision,
                "n_grasps_after_collision": obj.n_grasps_after_collision,
                "n_grasps_final": obj.n_grasps_final,
                "grasps": [_grasp_to_json(g) for g in obj.grasps],
            }
            if return_points:
                obj_entry["point_cloud_mm"] = obj.point_cloud.tolist()
                obj_entry["normals"] = obj.normals.tolist()
            response["objects"].append(obj_entry)

        return JSONResponse(content=response)

    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


def _grasp_to_json(g) -> dict:
    return {
        "score": round(g.score, 6),
        "width_m": round(g.width, 6),
        "height_m": round(g.height, 6),
        "depth_m": round(g.depth, 6),
        "rotation_matrix": g.rotation_matrix.tolist(),
        "translation_m": g.translation.tolist(),
        "object_id": g.object_id,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
