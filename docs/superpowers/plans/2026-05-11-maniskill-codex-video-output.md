# ManiSkill Codex Video Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional MP4 recording to the existing ZeroGrasp ManiSkill pick runner.

**Architecture:** Keep simulation logic in `execute_zerograsp_pick.py` and add a tiny `VideoRecorder` helper that stores RGB frames from `env.render()`. The recorder is disabled unless `--video-out` is provided, so existing CLI behavior stays unchanged.

**Tech Stack:** Python 3, NumPy, stdlib unittest, imageio v3 for MP4 writing, ManiSkill 3 for smoke testing.

---

### Task 1: CLI and Recorder Unit Tests

**Files:**
- Modify: `maniskill_codex/tests/test_execute_zerograsp_pick.py`
- Modify: `maniskill_codex/execute_zerograsp_pick.py`

- [ ] **Step 1: Write failing tests**

Add tests that verify `parse_args` accepts `--video-out` and `--video-fps`, and that `VideoRecorder.capture()` appends a uint8 RGB frame from a fake env.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m unittest maniskill_codex.tests.test_execute_zerograsp_pick -v`

Expected: failure because `video_out`, `video_fps`, and `VideoRecorder` are not implemented.

- [ ] **Step 3: Implement parser args and VideoRecorder**

Add `--video-out`, `--video-fps`, a no-op disabled recorder, frame normalization for `(H,W,3)` and `(1,H,W,3)`, and `imageio.v3.imwrite(..., fps=...)`.

- [ ] **Step 4: Re-run tests**

Run: `python -m unittest maniskill_codex.tests.test_execute_zerograsp_pick -v`

Expected: tests pass.

### Task 2: Simulation Integration and Verification

**Files:**
- Modify: `maniskill_codex/execute_zerograsp_pick.py`

- [ ] **Step 1: Capture frames during execution**

Capture one frame after reset and one frame after every `env.step()`. Save after all episodes before closing env.

- [ ] **Step 2: Verify pure tests and CLI**

Run:

```bash
python -m unittest discover -s maniskill_codex/tests -v
python -m maniskill_codex.execute_zerograsp_pick --help
```

Expected: tests pass and help shows `--video-out`.

- [ ] **Step 3: Generate MP4 smoke test**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maniskill
PYTHONPATH=. python -m maniskill_codex.execute_zerograsp_pick --zerograsp-output output --episodes 1 --seed 42 --video-out maniskill_codex/videos/zg_pick_seed42.mp4
```

Expected: command exits 0 and writes a non-empty MP4 file.

