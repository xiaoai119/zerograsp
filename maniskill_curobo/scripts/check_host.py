#!/usr/bin/env python3
"""Print host diagnostics for the isolated ManiSkill + cuRobo experiment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.returncode, completed.stdout.strip()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    experiment_root = Path(__file__).resolve().parents[1]
    print(f"repo_root={repo_root}")
    print(f"experiment_root={experiment_root}")
    print(f"python={sys.executable}")
    print(f"python_version={sys.version.split()[0]}")
    print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST', '<unset>')}")

    for binary in ("conda", "nvidia-smi", "nvcc", "git"):
        path = shutil.which(binary)
        print(f"{binary}={path or '<not found>'}")

    code, output = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"])
    print("nvidia_smi_query:")
    print(output if code == 0 else f"failed rc={code}: {output}")

    code, output = run(["bash", "-lc", "free -h | sed -n '1,2p'"])
    print("memory:")
    print(output if code == 0 else f"failed rc={code}: {output}")

    code, output = run(["bash", "-lc", "df -h . | sed -n '1,2p'"])
    print("disk:")
    print(output if code == 0 else f"failed rc={code}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
