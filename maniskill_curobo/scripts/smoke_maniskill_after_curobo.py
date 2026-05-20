#!/usr/bin/env python3
"""Verify ManiSkill can start after cuRobo is imported in the same process."""

from __future__ import annotations


def main() -> int:
    import curobo
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make(
        "PickSingleYCB-v1",
        render_mode="rgb_array",
        control_mode="pd_joint_pos",
        robot_uids="panda",
        obs_mode="state",
        max_episode_steps=5,
    )
    try:
        obs, _ = env.reset(seed=1)
        print(f"curobo_version={getattr(curobo, '__version__', '<unknown>')}")
        print(f"env_created={env.spec.id}")
        print(f"obs_type={type(obs).__name__}")
        print(f"control_mode={getattr(env.unwrapped, 'control_mode', '<unknown>')}")
    finally:
        env.close()
    print("maniskill_after_curobo_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
