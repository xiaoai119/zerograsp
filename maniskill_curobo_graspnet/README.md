# GraspNet Baseline + ManiSkill + cuRobo

This directory evaluates the official GraspNet baseline as a drop-in
replacement for ZeroGrasp while keeping the rest of the execution stack fixed.

## Fair comparison

The seed1-200 comparison reuses RGB-D, target masks, camera intrinsics, and
20-step scene settling from:

```text
maniskill_curobo/runs/depth_corrected_settle20_seed1_200_rerun
```

Both models are executed with the same:

- PickSingleYCB seed;
- camera and target mask;
- Panda hand/TCP calibration;
- 10 cm pre-grasp offset;
- full predicted grasp depth with automatic shallow fallback;
- cuRobo table collision world;
- 15 cm lift and object-lift success criterion.

GraspNet receives a local point cloud around the target mask. Since the
baseline does not accept an object prompt, predicted grasp centers are filtered
to the target object's 3D bounding box before cuRobo execution.

## Environment

Create the runtime:

```bash
bash maniskill_curobo_graspnet/scripts/create_env.sh
```

Download the RealSense checkpoint:

```bash
bash maniskill_curobo_graspnet/scripts/download_checkpoint.sh
```

Check readiness:

```bash
PYTHONPATH=. maniskill_curobo_graspnet/envs/graspnet/bin/python \
  -m maniskill_curobo_graspnet.scripts.check_runtime
```

## Run seed1-200

```bash
bash maniskill_curobo_graspnet/scripts/run_seed1_200.sh
```

By default this run skips cuRobo videos to keep the full 200-seed comparison
fast. To record execution videos as well:

```bash
CUROBO_VIDEO=1 bash maniskill_curobo_graspnet/scripts/run_seed1_200.sh
```

Results are written under:

```text
maniskill_curobo_graspnet/runs/seed1_200
```

The final pairwise report is:

```text
maniskill_curobo_graspnet/runs/seed1_200/comparison/comparison_report.md
```

The existing fair ZeroGrasp reference completed 106 successful lifts out of
200 seeds.
