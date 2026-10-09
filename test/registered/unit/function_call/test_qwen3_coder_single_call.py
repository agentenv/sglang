"""qwen3_coder + tool_choice=auto + parallel_tool_calls=False: at most one call (yeto S17 N17)."""

import unittest

from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


class TestQwen3CoderSingleCall(unittest.TestCase):
    def test_parallel_true_keeps_unconstrained_auto(self):
        self.assertIsNone(
            Qwen3CoderDetector().get_auto_tool_call_structural_tag(parallel_tool_calls=True)
        )

    def test_parallel_false_stops_after_first_call(self):
        tag = Qwen3CoderDetector().get_auto_tool_call_structural_tag(parallel_tool_calls=False)
        fmt = tag.format
        self.assertEqual(fmt.type, "triggered_tags")
        self.assertTrue(fmt.stop_after_first)
        self.assertEqual(fmt.triggers, ["<tool_call>"])
        self.assertEqual(fmt.tags[0].end, "</tool_call>")

    def test_thinking_mode_prefixes_reasoning(self):
        tag = Qwen3CoderDetector().get_auto_tool_call_structural_tag(
            thinking_mode=True, parallel_tool_calls=False
        )
        self.assertEqual(tag.format.type, "sequence")


if __name__ == "__main__":
    unittest.main()
