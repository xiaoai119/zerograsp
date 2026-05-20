# ManiSkill Codex ZeroGrasp Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `maniskill_codex` runner that reads offline ZeroGrasp grasp output and executes a Panda pick attempt in ManiSkill.

**Architecture:** Split pure logic from simulation side effects. `zerograsp_outputs.py` parses JSON/NumPy output into a typed grasp record, `transforms.py` handles coordinate conversions, and `execute_zerograsp_pick.py` owns ManiSkill environment creation and robot action sequencing.

**Tech Stack:** Python 3, NumPy, stdlib `argparse`/`dataclasses`/`unittest`, ManiSkill 3 through the `maniskill` conda environment.

---

## File Structure

- Create `maniskill_codex/__init__.py`: package marker and public module docstring.
- Create `maniskill_codex/zerograsp_outputs.py`: load `recommended_grasp_top1.json` or select the highest score from `raw_outputs/*.grasp.npy`.
- Create `maniskill_codex/transforms.py`: convert ZeroGrasp OpenCV camera-frame positions to world and robot base coordinates.
- Create `maniskill_codex/execute_zerograsp_pick.py`: command-line runner for environment setup and grasp execution.
- Create `maniskill_codex/tests/test_zerograsp_outputs.py`: unit tests for JSON and `.npy` parsing.
- Create `maniskill_codex/tests/test_transforms.py`: unit tests for coordinate conversion math.

### Task 1: ZeroGrasp Output Parsing

**Files:**
- Create: `maniskill_codex/tests/test_zerograsp_outputs.py`
- Create: `maniskill_codex/__init__.py`
- Create: `maniskill_codex/zerograsp_outputs.py`

- [ ] **Step 1: Write failing parsing tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from maniskill_codex.zerograsp_outputs import GraspRecord, load_best_grasp


class ZeroGraspOutputTests(unittest.TestCase):
    def test_loads_recommended_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "recommended_grasp_top1.json").write_text(json.dumps({
                "score": 0.9,
                "width_m": 0.04,
                "height_m": 0.02,
                "depth_m": 0.03,
                "rotation_matrix_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translation_m_camera": [0.1, 0.2, 0.3],
                "source_file": "000_0.grasp.npy",
                "object_index": 0
            }), encoding="utf-8")

            record = load_best_grasp(root)

            self.assertIsInstance(record, GraspRecord)
            self.assertEqual(record.source, "recommended_grasp_top1.json")
            self.assertEqual(record.score, 0.9)
            np.testing.assert_allclose(record.translation_m_camera, [0.1, 0.2, 0.3])

    def test_selects_highest_score_from_raw_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw_outputs"
            raw.mkdir()
            np.save(raw / "000_0.grasp.npy", np.array([
                [0.1, 0.04, 0.02, 0.03, *np.eye(3).reshape(-1), 0.1, 0.2, 0.3, -1],
            ], dtype=np.float64))
            np.save(raw / "000_1.grasp.npy", np.array([
                [0.8, 0.05, 0.02, 0.04, *np.eye(3).reshape(-1), 0.4, 0.5, 0.6, -1],
            ], dtype=np.float64))

            record = load_best_grasp(root)

            self.assertEqual(record.score, 0.8)
            self.assertEqual(record.source, "raw_outputs/000_1.grasp.npy")
            np.testing.assert_allclose(record.translation_m_camera, [0.4, 0.5, 0.6])

    def test_raises_when_no_grasp_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "recommended_grasp_top1.json"):
                load_best_grasp(Path(tmp))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest maniskill_codex.tests.test_zerograsp_outputs -v`

Expected: FAIL or ERROR because `maniskill_codex.zerograsp_outputs` is not implemented.

- [ ] **Step 3: Implement minimal parser**

Create a `GraspRecord` dataclass with score, dimensions, rotation, translation, object metadata, and source. Implement JSON parsing with field checks and raw `.npy` fallback that skips empty arrays and selects the highest score row.

- [ ] **Step 4: Run parsing tests to verify they pass**

Run: `python -m unittest maniskill_codex.tests.test_zerograsp_outputs -v`

Expected: all three parsing tests pass.

### Task 2: Coordinate Transforms

**Files:**
- Create: `maniskill_codex/tests/test_transforms.py`
- Create: `maniskill_codex/transforms.py`

- [ ] **Step 1: Write failing transform tests**

```python
import unittest

import numpy as np

from maniskill_codex.transforms import opencv_camera_to_sapien_camera, opencv_camera_to_base


class TransformTests(unittest.TestCase):
    def test_opencv_to_sapien_flips_y_and_z(self):
        np.testing.assert_allclose(
            opencv_camera_to_sapien_camera(np.array([1.0, 2.0, 3.0])),
            np.array([1.0, -2.0, -3.0]),
        )

    def test_opencv_camera_to_base_applies_camera_and_base_matrices(self):
        camera_model = np.eye(4)
        camera_model[:3, 3] = [10.0, 20.0, 30.0]
        world_from_base = np.eye(4)
        world_from_base[:3, 3] = [1.0, 2.0, 3.0]

        result = opencv_camera_to_base(
            np.array([0.5, 0.25, 0.75]),
            camera_model,
            world_from_base,
        )

        np.testing.assert_allclose(result, [9.5, 17.75, 26.25])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest maniskill_codex.tests.test_transforms -v`

Expected: FAIL or ERROR because `maniskill_codex.transforms` is not implemented.

- [ ] **Step 3: Implement transform helpers**

Implement `opencv_camera_to_sapien_camera(position)` and `opencv_camera_to_base(position, camera_model_matrix, world_from_base_matrix)` with shape validation and `float64` math.

- [ ] **Step 4: Run transform tests to verify they pass**

Run: `python -m unittest maniskill_codex.tests.test_transforms -v`

Expected: both transform tests pass.

### Task 3: ManiSkill Execution Runner

**Files:**
- Create: `maniskill_codex/execute_zerograsp_pick.py`

- [ ] **Step 1: Write command-line runner**

Implement a script with:

```bash
PYTHONPATH=. python -m maniskill_codex.execute_zerograsp_pick --zerograsp-output output --episodes 1 --seed 42
```

Required behavior:

- Import ManiSkill inside functions so parser tests do not require ManiSkill.
- Create `PickSingleYCB-v1` with `render_mode="rgb_array"`, `control_mode="pd_ee_pose"`, `robot_uids="panda"`, `obs_mode="sensor_data"`.
- Read the environment camera model matrix from sensor `base_camera` by default.
- Read the robot base pose transformation matrix.
- Convert the selected ZeroGrasp grasp center to base coordinates.
- Clamp the base target to a conservative workspace before executing.
- Execute four stages: `pre`, `descend`, `close`, `lift`.
- Print score, source, target point, per-stage actual TCP position, and final `info`.

- [ ] **Step 2: Run import smoke test**

Run: `python -m maniskill_codex.execute_zerograsp_pick --help`

Expected: exits 0 and prints CLI options without importing ManiSkill.

- [ ] **Step 3: Run unit test suite**

Run: `python -m unittest discover -s maniskill_codex/tests -v`

Expected: all unit tests pass.

- [ ] **Step 4: Run ManiSkill smoke test when environment is available**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maniskill
PYTHONPATH=. python -m maniskill_codex.execute_zerograsp_pick --zerograsp-output output --episodes 1 --seed 42
```

Expected: environment starts, one episode resets, the runner prints all four stage names or stops with a clear ManiSkill runtime error.

### Task 4: Final Verification

**Files:**
- Read: `docs/superpowers/specs/2026-05-11-maniskill-codex-zerograsp-execution-design.md`
- Read: `docs/superpowers/plans/2026-05-11-maniskill-codex-zerograsp-execution.md`

- [ ] **Step 1: Run all unit tests**

Run: `python -m unittest discover -s maniskill_codex/tests -v`

Expected: all unit tests pass.

- [ ] **Step 2: Run CLI help**

Run: `python -m maniskill_codex.execute_zerograsp_pick --help`

Expected: exits 0 and shows `--zerograsp-output`, `--episodes`, and `--seed`.

- [ ] **Step 3: Run one-episode ManiSkill smoke test**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maniskill
PYTHONPATH=. python -m maniskill_codex.execute_zerograsp_pick --zerograsp-output output --episodes 1 --seed 42
```

Expected: if ManiSkill can run in the current machine session, the command exits after one episode. If rendering or GPU setup blocks simulation, report the exact error and keep unit-level verification evidence.

- [ ] **Step 4: Report non-git status**

Run: `git status --short`

Expected: command reports that this is not a git repository. Include that in the final response instead of claiming a commit.

