import os
import stat
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DockerAssetsTests(unittest.TestCase):
    def test_full_dockerfile_targets_3090_and_full_pipeline(self):
        dockerfile = ROOT / "docker" / "Dockerfile.full"

        text = dockerfile.read_text(encoding="utf-8")

        self.assertIn("FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel", text)
        self.assertIn("TORCH_CUDA_ARCH_LIST=8.6", text)
        self.assertIn("python -m pip install mani_skill", text)
        self.assertIn("third_party/octree_feature_extractor", text)
        self.assertIn("maniskill_codex.run_full_pipeline", text)
        self.assertIn("--no-conda", text)

    def test_full_build_and_run_scripts_are_executable(self):
        for relative in [
            "docker/build_full_image.sh",
            "docker/run_full_pipeline_in_docker.sh",
        ]:
            path = ROOT / relative
            mode = path.stat().st_mode

            self.assertTrue(mode & stat.S_IXUSR, f"{relative} should be executable")

    def test_full_run_script_mounts_output_and_forwards_args(self):
        script = (ROOT / "docker" / "run_full_pipeline_in_docker.sh").read_text(encoding="utf-8")

        self.assertIn("zerograsp-maniskill:3090", script)
        self.assertIn("--gpus all", script)
        self.assertIn("-v \"${OUTPUT_ROOT}:/workspace/output\"", script)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", script)
        self.assertIn("--no-conda", script)
        self.assertIn("\"$@\"", script)

    def test_dockerignore_excludes_generated_runs_and_caches(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

        for pattern in [
            "maniskill_codex/runs",
            "maniskill_codex/videos",
            "maniskill_codex/zg_inputs",
            "maniskill_codex/zg_outputs",
            "__pycache__",
            "*.pyc",
        ]:
            self.assertIn(pattern, dockerignore)


if __name__ == "__main__":
    unittest.main()
