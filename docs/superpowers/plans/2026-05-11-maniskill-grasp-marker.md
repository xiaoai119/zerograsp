# ManiSkill Grasp Marker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the current ZeroGrasp output in the ManiSkill scene as a visual-only 3D marker.

**Architecture:** Add `maniskill_codex/grasp_markers.py` for pure marker geometry and SAPIEN visual actor creation. The runner gets `--show-grasp-marker`; after reset it converts the selected ZeroGrasp grasp from camera frame to world frame and adds marker actors before video capture/execution.

**Tech Stack:** Python 3, NumPy, ManiSkill/SAPIEN visual-only kinematic actors, unittest.

---

### Task 1: Marker Geometry

**Files:**
- Create: `maniskill_codex/grasp_markers.py`
- Create: `maniskill_codex/tests/test_grasp_markers.py`

- [ ] Write failing tests for OpenCV camera rotation to world axes and marker primitive geometry.
- [ ] Run `python -m unittest maniskill_codex.tests.test_grasp_markers -v` and confirm import failure.
- [ ] Implement geometry helpers with no SAPIEN dependency for tests.
- [ ] Re-run tests and confirm pass.

### Task 2: Runner Integration

**Files:**
- Modify: `maniskill_codex/execute_zerograsp_pick.py`
- Modify: `maniskill_codex/tests/test_execute_zerograsp_pick.py`

- [ ] Add failing CLI test for `--show-grasp-marker`.
- [ ] Add `--show-grasp-marker` and call marker creation after reset, before capture.
- [ ] Run all unit tests.
- [ ] Run one ManiSkill smoke test with `--show-grasp-marker --video-out ...`.

