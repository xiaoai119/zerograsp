# ManiSkill Grasp Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a PNG that overlays a ZeroGrasp camera-frame grasp pose on the RGB input image.

**Architecture:** Add a standalone `maniskill_codex/grasp_projection.py` module with pure projection math, PIL drawing, and a CLI entrypoint. Keep it independent from the simulator runner so it can visualize any saved ZeroGrasp input/output pair.

**Tech Stack:** Python 3, NumPy, PIL, stdlib unittest/argparse/json.

---

### Task 1: Projection Math and Drawing

**Files:**
- Create: `maniskill_codex/tests/test_grasp_projection.py`
- Create: `maniskill_codex/grasp_projection.py`

- [ ] **Step 1: Write failing tests**

Test `project_3d_to_2d()` with valid and behind-camera points, and test that `draw_grasp_projection()` writes a non-empty RGB image with overlay pixels changed.

- [ ] **Step 2: Run red tests**

Run: `python -m unittest maniskill_codex.tests.test_grasp_projection -v`

Expected: import failure because `grasp_projection.py` does not exist.

- [ ] **Step 3: Implement module**

Implement JSON loading, projection math, grasp geometry construction, PIL overlay drawing, and CLI args:

```bash
python -m maniskill_codex.grasp_projection \
  --rgb maniskill_codex/zg_inputs/seed42/episode_000/rgb.png \
  --camera maniskill_codex/zg_inputs/seed42/episode_000/camera.json \
  --grasp maniskill_codex/zg_outputs/seed42_episode_000_pipeline/recommended_grasp_top1.json \
  --output maniskill_codex/zg_outputs/seed42_episode_000_pipeline/grasp_projection.png
```

- [ ] **Step 4: Verify**

Run all unit tests and then generate the actual PNG for seed 42.

