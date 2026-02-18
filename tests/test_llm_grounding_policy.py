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
        decision = decide_chat_grounding("off", query="Tell me about my ancestors")
        self.assertEqual(decision["mode"], "off")
        self.assertEqual(decision["decision_reason"], "mode_off")
        self.assertFalse(decision["grounding_attached"])
        self.assertFalse(decision["should_refuse"])

    def test_mode_on_enables_grounding(self):
        """on mode must enable grounding."""
        decision = decide_chat_grounding("on", query="Tell me about my ancestors")
        self.assertEqual(decision["mode"], "on")
        self.assertEqual(decision["decision_reason"], "mode_on")
        self.assertTrue(decision["grounding_attached"])
        self.assertFalse(decision["should_refuse"])

    def test_mode_auto_tree_sufficient_disables_grounding(self):
        """auto mode should disable grounding for tree-sufficient prompts."""
        decision = decide_chat_grounding("auto", query="Who are the parents of John Smith?")
        self.assertEqual(decision["mode"], "auto")
        self.assertEqual(decision["decision_reason"], "tree_sufficient")
        self.assertFalse(decision["grounding_attached"])
        self.assertFalse(decision["should_refuse"])

    def test_mode_auto_context_gap_enables_grounding(self):
        """auto mode should enable grounding for context-gap prompts."""
        decision = decide_chat_grounding(
            "auto",
            query="Why did this branch migrate to Pennsylvania in the 1880s?",
        )
        self.assertEqual(decision["decision_reason"], "context_gap")
        self.assertTrue(decision["grounding_attached"])

    def test_mode_auto_cap_blocked_disables_grounding(self):
        """auto mode should block grounding after free-tier limit is reached."""
        decision = decide_chat_grounding(
            "auto",
            query="Why did this branch migrate to Pennsylvania in the 1880s?",
            current_grounded_prompts_count=5000,
            free_tier_limit=5000,
        )
        self.assertEqual(decision["decision_reason"], "cap_blocked")
        self.assertFalse(decision["grounding_attached"])

    def test_scope_out_refusal(self):
        """Out-of-scope prompts should be refused in any mode."""
        decision = decide_chat_grounding(
            "auto", query="Did Predator Badlands get good movie reviews?"
        )
        self.assertEqual(decision["decision_reason"], "scope_out")
        self.assertFalse(decision["grounding_attached"])
        self.assertTrue(decision["should_refuse"])
        self.assertIsNotNone(decision["refusal_message"])
