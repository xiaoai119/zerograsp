#!/usr/bin/env python3
"""Smoke-test imports for the isolated ManiSkill + cuRobo environment."""

from __future__ import annotations


def main() -> int:
    import torch

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"capability={torch.cuda.get_device_capability(0)}")
        print(f"cuda_tensor_sum={torch.ones(4, device='cuda').sum().item()}")

    import gymnasium
    import imageio
    import mani_skill
    import sapien

    print(f"gymnasium={gymnasium.__version__}")
    print(f"imageio={imageio.__version__}")
    print(f"mani_skill={getattr(mani_skill, '__version__', '<unknown>')}")
    print(f"sapien={getattr(sapien, '__version__', '<unknown>')}")

    import curobo

    print(f"curobo={getattr(curobo, '__version__', '<unknown>')}")
    print("smoke_imports_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
