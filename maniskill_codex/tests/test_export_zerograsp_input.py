import unittest

from maniskill_codex.export_zerograsp_input import parse_args


class ExportZeroGraspInputTests(unittest.TestCase):
    def test_parse_args_uses_scene_defaults(self):
        args = parse_args(["--output-dir", "out"])

        self.assertEqual(args.output_dir, "out")
        self.assertEqual(args.env_id, "PickClutterYCB-v1")
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.camera, "base_camera")
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 1024)
        self.assertEqual(args.mask_mode, "task-target")
        self.assertEqual(args.camera_eye, [-0.30, 0.0, 0.55])
        self.assertEqual(args.camera_target, [0.05, 0.0, 0.08])


if __name__ == "__main__":
    unittest.main()
