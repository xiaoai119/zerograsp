#!/usr/bin/env python3
"""One-off ManiSkill video showing cuRobo Franka collision spheres."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_curobo.scripts.execute_curobo_pick import (
    DEFAULT_CAMERA_EYE,
    DEFAULT_CAMERA_TARGET,
    VideoRecorder,
    build_env,
    first_vector,
    pose_to_matrix,
    write_json,
)


DEFAULT_ROBOT_CONFIG = Path(
    "maniskill_curobo/external/curobo/curobo/content/configs/robot/franka.yml"
)

LINK_COLORS = {
    "panda_link0": [0.55, 0.55, 0.55, 0.55],
    "panda_link1": [1.00, 0.15, 0.10, 0.55],
    "panda_link2": [1.00, 0.55, 0.05, 0.55],
    "panda_link3": [1.00, 0.90, 0.05, 0.55],
    "panda_link4": [0.20, 0.85, 0.20, 0.55],
    "panda_link5": [0.05, 0.60, 1.00, 0.55],
    "panda_link6": [0.35, 0.25, 1.00, 0.55],
    "panda_link7": [0.80, 0.20, 1.00, 0.55],
    "panda_hand": [0.10, 1.00, 0.90, 0.65],
    "panda_leftfinger": [1.00, 0.10, 0.85, 0.75],
    "panda_rightfinger": [1.00, 0.10, 0.85, 0.75],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render cuRobo Franka collision spheres as visual-only ManiSkill actors."
    )
    parser.add_argument("--env-id", default="PickClutterYCB-v1")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--robot-config-path", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maniskill_curobo_real/runs/collision_spheres_once/seed009"),
    )
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=1024)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=list(DEFAULT_CAMERA_EYE))
    parser.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--video-frames", type=int, default=160)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Scale visual sphere radii. Keep 1.0 to match cuRobo config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    try:
        obs, info = env.reset(seed=args.seed)
        action = current_pd_action(env, gripper=1.0)
        for _ in range(max(0, int(args.settle_steps))):
            obs, _, _, _, info = env.step(action[None, :])

        sphere_specs = load_collision_spheres(args.robot_config_path)
        link_map = robot_link_map(env)
        sphere_actors, manifest = add_collision_sphere_visuals(
            env.unwrapped.scene,
            sphere_specs,
            link_map,
            radius_scale=float(args.radius_scale),
        )

        recorder = VideoRecorder(args.output_dir / "collision_spheres.mp4", fps=args.video_fps)
        q0 = current_qpos(env)
        for frame_idx in range(int(args.video_frames)):
            phase = frame_idx / max(1, int(args.video_frames) - 1)
            target = q0.copy()
            # Gentle arm motion makes it clear the spheres are attached to links.
            target[0] += 0.25 * np.sin(2.0 * np.pi * phase)
            target[1] += 0.18 * np.sin(2.0 * np.pi * phase + 0.7)
            target[3] += 0.12 * np.sin(2.0 * np.pi * phase + 1.4)
            action = pd_action_from_qpos(env, target, gripper=1.0)
            obs, _, _, _, info = env.step(action[None, :])
            update_collision_sphere_visuals(sphere_actors, link_map)
            recorder.capture(env)
        video_path = recorder.save()

        manifest.update(
            {
                "env_id": args.env_id,
                "seed": int(args.seed),
                "video_path": str(video_path),
                "robot_config_path": str(args.robot_config_path),
                "radius_scale": float(args.radius_scale),
                "settle_steps": int(args.settle_steps),
                "video_frames": int(args.video_frames),
                "video_fps": int(args.video_fps),
                "note": "Visual-only actors; cuRobo still uses the spheres from franka.yml for planning.",
            }
        )
        write_json(args.output_dir / "collision_spheres_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    finally:
        env.close()


def load_collision_spheres(path: Path) -> dict[str, list[dict[str, Any]]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    spheres = data["robot_cfg"]["kinematics"]["collision_spheres"]
    out: dict[str, list[dict[str, Any]]] = {}
    for link_name, items in spheres.items():
        out[str(link_name)] = [
            {
                "center": np.asarray(item["center"], dtype=np.float64).reshape(3),
                "radius": float(item["radius"]),
            }
            for item in items
        ]
    return out


def robot_link_map(env: Any) -> dict[str, Any]:
    robot = env.unwrapped.agent.robot
    return {str(link.name): link for link in robot.get_links()}


def add_collision_sphere_visuals(
    scene: Any,
    sphere_specs: dict[str, list[dict[str, Any]]],
    link_map: dict[str, Any],
    *,
    radius_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import sapien

    actors: list[dict[str, Any]] = []
    link_counts: dict[str, int] = {}
    skipped_links: list[str] = []
    for link_name, specs in sphere_specs.items():
        link = link_map.get(link_name)
        if link is None:
            skipped_links.append(link_name)
            continue
        color = LINK_COLORS.get(link_name, [1.0, 1.0, 1.0, 0.45])
        material = sapien.render.RenderMaterial(base_color=color)
        link_counts[link_name] = len(specs)
        for index, spec in enumerate(specs):
            builder = scene.create_actor_builder()
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
                radius=float(spec["radius"]) * radius_scale,
                material=material,
            )
            pose = sphere_world_pose(link, spec["center"])
            builder.set_initial_pose(pose)
            actor = builder.build_kinematic(name=f"curobo_collision_sphere_{link_name}_{index:02d}")
            actors.append({"actor": actor, "link": link, "center": spec["center"], "radius": spec["radius"]})
    return actors, {
        "sphere_count": len(actors),
        "link_counts": link_counts,
        "skipped_links": skipped_links,
    }


def update_collision_sphere_visuals(actors: list[dict[str, Any]], link_map: dict[str, Any]) -> None:
    for item in actors:
        item["actor"].set_pose(sphere_world_pose(item["link"], item["center"]))


def sphere_world_pose(link: Any, center_link: np.ndarray) -> Any:
    import sapien

    world_from_link = pose_to_matrix(getattr(link, "pose", None), f"{link.name} pose")
    center_world = (world_from_link @ np.array([*center_link, 1.0], dtype=np.float64))[:3]
    return sapien.Pose(p=center_world.tolist())


def current_qpos(env: Any) -> np.ndarray:
    robot = env.unwrapped.agent.robot
    return first_vector(robot.get_qpos(), "robot qpos").astype(np.float64)


def current_pd_action(env: Any, *, gripper: float) -> np.ndarray:
    return pd_action_from_qpos(env, current_qpos(env), gripper=gripper)


def pd_action_from_qpos(env: Any, qpos: np.ndarray, *, gripper: float) -> np.ndarray:
    dim = int(np.prod(getattr(env.action_space, "shape", (0,))))
    action = np.zeros(dim, dtype=np.float32)
    n_arm = min(7, dim, qpos.shape[0])
    action[:n_arm] = np.asarray(qpos[:n_arm], dtype=np.float32)
    if dim >= 8:
        action[7] = float(gripper)
    elif dim >= 1:
        action[-1] = float(gripper)
    return action


if __name__ == "__main__":
    main()
