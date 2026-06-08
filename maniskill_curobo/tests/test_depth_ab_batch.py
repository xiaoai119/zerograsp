from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from maniskill_curobo.scripts.run_zerograsp_depth_ab_batch import (
    PersistentExecutionRunner,
    PersistentInputExportRunner,
    compare_variants,
    environment_signature,
    generated_candidates_are_needed,
    input_bundle_is_valid,
    in_process_execute_argv,
    in_process_export_argv,
    manifest_outcome,
    parse_args,
    scene_model_signature,
    validate_persistent_worker_args,
)


class DepthAbBatchTest(unittest.TestCase):
    def test_parse_args_uses_full_depth_treatment(self) -> None:
        args = parse_args(
            [
                "--env-id",
                "PickClutterYCB-v1",
                "--output-root",
                "out",
            ]
        )

        self.assertEqual(args.seed_start, 1)
        self.assertEqual(args.seed_end, 200)
        self.assertEqual(args.baseline_depth_scale, 0.0)
        self.assertEqual(args.depth_scale, 1.0)
        self.assertEqual(args.grasp_depth_max_offset, 0.04)
        self.assertEqual(args.settle_before_export_steps, 20)
        self.assertFalse(args.depth_auto_fallback)
        self.assertTrue(args.persistent_zerograsp_worker)

    def test_parse_args_can_enable_depth_auto_fallback(self) -> None:
        args = parse_args(
            [
                "--env-id",
                "PickSingleYCB-v1",
                "--output-root",
                "out",
                "--depth-auto-fallback",
            ]
        )

        self.assertTrue(args.depth_auto_fallback)

    def test_parse_args_can_enable_persistent_worker(self) -> None:
        args = parse_args(
            [
                "--env-id",
                "PickSingleYCB-v1",
                "--output-root",
                "out",
                "--reuse-candidate-root",
                "candidates",
                "--persistent-worker",
            ]
        )

        self.assertTrue(args.persistent_worker)
        self.assertFalse(args.persistent_child)

    def test_persistent_worker_accepts_generated_picksingle(self) -> None:
        generated = parse_args(
            [
                "--env-id",
                "PickSingleYCB-v1",
                "--output-root",
                "out",
                "--persistent-worker",
            ]
        )
        clutter = parse_args(
            [
                "--env-id",
                "PickClutterYCB-v1",
                "--output-root",
                "out",
                "--reuse-candidate-root",
                "candidates",
                "--persistent-worker",
            ]
        )

        validate_persistent_worker_args(generated)
        with self.assertRaises(ValueError):
            validate_persistent_worker_args(clutter)

    def test_manifest_outcome_prioritizes_planning_failure(self) -> None:
        manifest = {
            "failure_reason": "cuRobo planning failed for stage=grasp: result_none",
            "object_lift_metrics": {
                "object_lift_success": False,
                "failure_reason": "object_not_lifted",
            },
        }

        self.assertEqual(manifest_outcome(manifest), "planning_failed_grasp")

    def test_compare_variants_identifies_improvement_and_regression(self) -> None:
        failed = {"outcome": "object_not_lifted", "object_lift_success": False}
        success = {"outcome": "success", "object_lift_success": True}

        self.assertEqual(compare_variants(failed, success)["change"], "improved")
        self.assertEqual(compare_variants(success, failed)["change"], "regressed")

    def test_input_bundle_rejects_an_empty_target_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("rgb.png", "depth.png"):
                Image.new("L", (4, 4), 1).save(root / name)
            Image.new("L", (4, 4), 0).save(root / "mask.png")
            (root / "camera.json").write_text("{}", encoding="utf-8")
            (root / "scene.json").write_text("{}", encoding="utf-8")

            self.assertFalse(input_bundle_is_valid(root))

            mask = Image.new("L", (4, 4), 0)
            mask.putpixel((1, 1), 1)
            mask.save(root / "mask.png")
            self.assertTrue(input_bundle_is_valid(root))

    def test_generated_candidates_only_start_worker_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = (
                root
                / "seed001"
                / "setup"
                / "zg_output"
                / "recommended_grasp_top1.json"
            )
            candidate.parent.mkdir(parents=True)
            candidate.write_text("{}", encoding="utf-8")

            self.assertFalse(
                generated_candidates_are_needed(
                    output_root=root,
                    seeds=[1],
                    reuse_existing=True,
                )
            )
            self.assertTrue(
                generated_candidates_are_needed(
                    output_root=root,
                    seeds=[1, 2],
                    reuse_existing=True,
                )
            )
            self.assertTrue(
                generated_candidates_are_needed(
                    output_root=root,
                    seeds=[1],
                    reuse_existing=False,
                )
            )

    def test_environment_and_scene_signatures_are_stable(self) -> None:
        args = parse_args(
            [
                "--env-id",
                "PickSingleYCB-v1",
                "--output-root",
                "out",
            ]
        )
        same = parse_args(
            [
                "--env-id",
                "PickSingleYCB-v1",
                "--output-root",
                "different",
            ]
        )

        self.assertEqual(environment_signature(args), environment_signature(same))
        self.assertEqual(
            scene_model_signature({"cuboid": {"table": {"dims": [1, 1, 1]}}}),
            scene_model_signature({"cuboid": {"table": {"dims": [1, 1, 1]}}}),
        )

    def test_in_process_execute_argv_strips_python_module_prefix(self) -> None:
        command = [
            "python",
            "-m",
            "maniskill_curobo.scripts.execute_curobo_pick",
            "--seed",
            "3",
        ]

        self.assertEqual(in_process_execute_argv(command), ["--seed", "3"])

    def test_in_process_export_argv_strips_python_module_prefix(self) -> None:
        command = [
            "python",
            "-m",
            "maniskill_codex.export_zerograsp_input",
            "--seed",
            "3",
        ]

        self.assertEqual(in_process_export_argv(command), ["--seed", "3"])

    def test_persistent_export_runner_reuses_environment(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        fake_env = FakeEnv()
        created = {"env": 0}

        def build_env(**_kwargs):
            created["env"] += 1
            return fake_env

        module = SimpleNamespace(build_env=build_env)

        def fake_main(_argv):
            env = module.build_env(
                width=1280,
                height=1024,
                env_id="PickSingleYCB-v1",
                camera_name="base_camera",
                camera_eye=[-0.3, 0.0, 0.55],
                camera_target=[0.05, 0.0, 0.08],
                control_mode="pd_joint_pos",
            )
            env.close()
            return 0

        module.main = fake_main
        command = [
            "python",
            "-m",
            "maniskill_codex.export_zerograsp_input",
            "--seed",
            "1",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            runner = PersistentInputExportRunner(
                repo_root=Path(tmp),
                export_module=module,
            )
            first = runner.run(command, logs_dir=Path(tmp) / "first", name="export")
            second = runner.run(command, logs_dir=Path(tmp) / "second", name="export")
            runner.close()

        self.assertEqual(first["exit_code"], 0)
        self.assertFalse(first["environment_reused"])
        self.assertTrue(second["environment_reused"])
        self.assertEqual(created["env"], 1)
        self.assertEqual(fake_env.close_calls, 1)

    def test_persistent_runner_reuses_env_and_planner(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class FakePlanner:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.destroy_calls = 0

            def reset_seed(self) -> None:
                self.reset_calls += 1

            def destroy(self) -> None:
                self.destroy_calls += 1

        fake_env = FakeEnv()
        fake_planner = FakePlanner()
        created = {"env": 0, "planner": 0}
        args = SimpleNamespace(
            env_id="PickSingleYCB-v1",
            camera="base_camera",
            width=1280,
            height=1024,
            render_width=1280,
            render_height=1024,
            camera_eye=[-0.3, 0.0, 0.55],
            camera_target=[0.05, 0.0, 0.08],
        )

        def build_env(_args):
            created["env"] += 1
            return fake_env

        def build_planner(_args, scene_model=None):
            created["planner"] += 1
            return fake_planner

        module = SimpleNamespace(build_env=build_env, build_planner=build_planner)

        def fake_main(_argv):
            env = module.build_env(args)
            module.build_planner(args, scene_model={"cuboid": {"table": {}}})
            env.close()
            return 0

        module.main = fake_main
        command = [
            "python",
            "-m",
            "maniskill_curobo.scripts.execute_curobo_pick",
            "--seed",
            "1",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            runner = PersistentExecutionRunner(
                repo_root=Path(tmp),
                execute_module=module,
            )
            first = runner.run(command, logs_dir=Path(tmp) / "first", name="execute")
            second = runner.run(command, logs_dir=Path(tmp) / "second", name="execute")
            runner.close()

        self.assertEqual(first["exit_code"], 0)
        self.assertFalse(first["environment_reused"])
        self.assertFalse(first["planner_reused"])
        self.assertTrue(second["environment_reused"])
        self.assertTrue(second["planner_reused"])
        self.assertEqual(created, {"env": 1, "planner": 1})
        self.assertEqual(fake_planner.reset_calls, 1)
        self.assertEqual(fake_planner.destroy_calls, 1)
        self.assertEqual(fake_env.close_calls, 1)

    def test_persistent_runner_rejects_a_changed_scene(self) -> None:
        planner = SimpleNamespace(reset_seed=lambda: None, destroy=lambda: None)
        module = SimpleNamespace(
            build_env=lambda _args: SimpleNamespace(close=lambda: None),
            build_planner=lambda _args, scene_model=None: planner,
            main=lambda _argv: 0,
        )
        args = SimpleNamespace()

        runner = PersistentExecutionRunner(
            repo_root=Path("."),
            execute_module=module,
        )
        runner._build_planner(args, scene_model={"cuboid": {"table": {"dims": [1, 1, 1]}}})
        with self.assertRaises(RuntimeError):
            runner._build_planner(
                args,
                scene_model={"cuboid": {"table": {"dims": [2, 1, 1]}}},
            )
        runner.close()


if __name__ == "__main__":
    unittest.main()
