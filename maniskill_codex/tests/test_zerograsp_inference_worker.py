from __future__ import annotations

import json
import unittest
from unittest import mock

from maniskill_codex.zerograsp_inference_worker import (
    READY_PREFIX,
    RESPONSE_PREFIX,
    emit,
    parse_args,
)


class ZeroGraspInferenceWorkerTest(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = parse_args([])

        self.assertFalse(args.enable_collision_detection)
        self.assertIsNone(args.device)

    def test_emit_writes_one_prefixed_json_line(self) -> None:
        with mock.patch("builtins.print") as print_mock:
            emit(RESPONSE_PREFIX, {"request_id": 3, "ok": True})

        line = print_mock.call_args.args[0]
        self.assertTrue(line.startswith(RESPONSE_PREFIX))
        self.assertEqual(
            json.loads(line[len(RESPONSE_PREFIX) :]),
            {"request_id": 3, "ok": True},
        )
        self.assertTrue(print_mock.call_args.kwargs["flush"])
        self.assertTrue(READY_PREFIX.startswith("ZEROGRASP_WORKER_"))


if __name__ == "__main__":
    unittest.main()
