#!/usr/bin/env python3
"""Persistent ManiSkill/cuRobo worker for Lift2 M4C grasp episodes."""

from __future__ import annotations

import contextlib
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

from maniskill_curobo_real.run_world_collision_stages import (
    load_planner_scene_model,
    planner_scene_family,
)

from . import execute_lift2_m4c_grasp as execute


READY_PREFIX = "LIFT2_M4C_WORKER_READY "
RESPONSE_PREFIX = "LIFT2_M4C_WORKER_RESPONSE "


class NonClosingEnvProxy:
    def __init__(self, env: Any):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._env, name, value)

    def close(self) -> None:
        return None


class NonDestroyingPlannerProxy:
    def __init__(self, planner: Any):
        object.__setattr__(self, "_planner", planner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._planner, name)

    def destroy(self) -> None:
        return None


class PersistentLift2Runner:
    def __init__(self) -> None:
        self._original_build_env = execute.build_lift2_execution_env
        self._original_build_planner = execute.build_lift2_planner
        self._env: Any | None = None
        self._env_proxy: NonClosingEnvProxy | None = None
        self._env_signature: tuple[Any, ...] | None = None
        self._planner: Any | None = None
        self._planner_proxy: NonDestroyingPlannerProxy | None = None
        self._scene_family: str | None = None
        self._planner_signature: tuple[Any, ...] | None = None
        self._episode_count = 0
        self._planner_rebuild_count = 0
        self._last_planner_reused = False
        execute.build_lift2_execution_env = self._build_env
        execute.build_lift2_planner = self._build_planner

    def _build_env(self, args: Any, output_dir: Path) -> NonClosingEnvProxy:
        signature = (
            str(args.robot_uid),
            str(args.camera),
            int(args.render_width),
            int(args.render_height),
            tuple(float(v) for v in args.camera_eye),
            tuple(float(v) for v in args.camera_target),
        )
        if self._env is None:
            self._env = self._original_build_env(args, output_dir)
            self._env_proxy = NonClosingEnvProxy(self._env)
            self._env_signature = signature
        elif signature != self._env_signature:
            raise RuntimeError(
                "Lift2 persistent environment configuration changed: "
                f"{self._env_signature} != {signature}"
            )
        assert self._env_proxy is not None
        return self._env_proxy

    def _build_planner(self, args: Any, scene_model: Any) -> NonDestroyingPlannerProxy:
        family = planner_scene_family(scene_model)
        planner_signature = (
            str(Path(args.config).expanduser().resolve()),
            int(args.num_ik_seeds),
            int(args.num_trajopt_seeds),
        )
        if (
            self._planner is not None
            and family == self._scene_family
            and planner_signature == self._planner_signature
        ):
            self._planner.clear_scene_cache()
            self._planner.update_world(scene_model)
            reset_seed = getattr(self._planner, "reset_seed", None)
            if callable(reset_seed):
                reset_seed()
            self._last_planner_reused = True
            assert self._planner_proxy is not None
            return self._planner_proxy

        self._destroy_planner()
        self._planner = self._original_build_planner(args, scene_model)
        self._planner_proxy = NonDestroyingPlannerProxy(self._planner)
        self._scene_family = family
        self._planner_signature = planner_signature
        self._planner_rebuild_count += 1
        self._last_planner_reused = False
        return self._planner_proxy

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = int(request["request_id"])
        argv = [str(value) for value in request["argv"]]
        stdout_path = Path(request["stdout_log"]).expanduser().resolve()
        stderr_path = Path(request["stderr_log"]).expanduser().resolve()
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        env_reused = self._env is not None
        rebuild_before = self._planner_rebuild_count
        started = time.time()
        exit_code = 0
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = execute.main(argv)
                exit_code = int(result or 0)
            except Exception:
                exit_code = 1
                traceback.print_exc(file=stderr)
        self._episode_count += 1
        return {
            "request_id": request_id,
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "runtime_sec": float(time.time() - started),
            "episode_index": self._episode_count,
            "environment_reused": env_reused,
            "planner_reused": bool(self._last_planner_reused),
            "planner_rebuilt": self._planner_rebuild_count > rebuild_before,
            "planner_rebuild_count": self._planner_rebuild_count,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    def close(self) -> None:
        execute.build_lift2_execution_env = self._original_build_env
        execute.build_lift2_planner = self._original_build_planner
        self._destroy_planner()
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_proxy = None

    def _destroy_planner(self) -> None:
        if self._planner is None:
            return
        destroy = getattr(self._planner, "destroy", None)
        if callable(destroy):
            destroy()
        self._planner = None
        self._planner_proxy = None
        self._scene_family = None
        self._planner_signature = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> int:
    runner = PersistentLift2Runner()
    print(
        READY_PREFIX
        + json.dumps({"pid": os.getpid(), "protocol_version": 1}, ensure_ascii=False),
        flush=True,
    )
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                response = runner.run(request)
            except Exception as exc:
                response = {
                    "request_id": request.get("request_id") if "request" in locals() else None,
                    "ok": False,
                    "exit_code": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            print(RESPONSE_PREFIX + json.dumps(response, ensure_ascii=False), flush=True)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
