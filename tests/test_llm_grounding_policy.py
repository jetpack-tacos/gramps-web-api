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
        self.assertEqual(decision["decision_reason"], "cap_blocked_free_tier")
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

    def test_mode_auto_soft_cap_tightened(self):
        """At soft cap, non-high-confidence context gaps should be tightened."""
        decision = decide_chat_grounding(
            "auto",
            query="Give context for this family branch",
            current_grounded_prompts_count=4500,
            soft_cap=4000,
            hard_cap=5000,
        )
        self.assertEqual(decision["decision_reason"], "soft_cap_tightened")
        self.assertFalse(decision["grounding_attached"])

    def test_mode_auto_soft_cap_high_confidence_keeps_grounding(self):
        """At soft cap, high-confidence context gap prompts should still ground."""
        decision = decide_chat_grounding(
            "auto",
            query="What was life like in this place for this migration?",
            current_grounded_prompts_count=4500,
            soft_cap=4000,
            hard_cap=5000,
        )
        self.assertEqual(decision["decision_reason"], "context_gap_soft_cap")
        self.assertTrue(decision["grounding_attached"])

    def test_mode_auto_hard_cap_blocks(self):
        """Hard cap should block grounding in auto mode."""
        decision = decide_chat_grounding(
            "auto",
            query="why did they move from germany to pennsylvania?",
            current_grounded_prompts_count=6000,
            hard_cap=6000,
        )
        self.assertEqual(decision["decision_reason"], "cap_blocked_hard")
        self.assertFalse(decision["grounding_attached"])

    def test_threshold_crossing_is_deterministic(self):
        """Crossing soft/hard caps should change behavior deterministically."""
        query = "what was life like in this place for this migration?"
        cases = [
            (3999, "context_gap", True),
            (4000, "context_gap_soft_cap", True),
            (4999, "context_gap_soft_cap", True),
            (5000, "cap_blocked_hard", False),
            (5001, "cap_blocked_hard", False),
        ]
        for grounded_count, expected_reason, expected_attached in cases:
            decision = decide_chat_grounding(
                "auto",
                query=query,
                current_grounded_prompts_count=grounded_count,
                free_tier_limit=100000,
                soft_cap=4000,
                hard_cap=5000,
            )
            self.assertEqual(decision["decision_reason"], expected_reason)
            self.assertEqual(decision["grounding_attached"], expected_attached)
