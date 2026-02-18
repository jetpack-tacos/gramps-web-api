"""Tests for chat grounding policy helpers."""

import unittest

from gramps_webapi.api.llm.grounding_policy import (
    decide_chat_grounding,
    normalize_search_grounding_mode,
)


class TestGroundingPolicy(unittest.TestCase):
    """Test Stage 1 grounding mode behavior."""

    def test_normalize_mode_defaults_to_auto(self):
        """Invalid and empty modes should normalize to auto."""
        self.assertEqual(normalize_search_grounding_mode(None), "auto")
        self.assertEqual(normalize_search_grounding_mode(""), "auto")
        self.assertEqual(normalize_search_grounding_mode("nonsense"), "auto")

    def test_mode_off_disables_grounding(self):
        """off mode must disable grounding."""
        decision = decide_chat_grounding("off")
        self.assertEqual(decision["mode"], "off")
        self.assertEqual(decision["decision_reason"], "mode_off")
        self.assertFalse(decision["grounding_attached"])

    def test_mode_on_enables_grounding(self):
        """on mode must enable grounding."""
        decision = decide_chat_grounding("on")
        self.assertEqual(decision["mode"], "on")
        self.assertEqual(decision["decision_reason"], "mode_on")
        self.assertTrue(decision["grounding_attached"])

    def test_mode_auto_uses_decision_function_result(self):
        """auto mode should flow through decision helper output."""
        decision = decide_chat_grounding("auto")
        self.assertEqual(decision["mode"], "auto")
        self.assertEqual(decision["decision_reason"], "auto_stage1_default")
        self.assertTrue(decision["grounding_attached"])
