"""Tests for grounding usage helper utilities."""

import unittest

from gramps_webapi.api.llm.grounding_usage import estimate_grounding_cost_usd


class TestGroundingUsage(unittest.TestCase):
    """Test cost estimation helpers."""

    def test_estimate_cost_under_free_tier(self):
        """No cost should be charged while under free tier."""
        cost = estimate_grounding_cost_usd(
            web_search_queries_count=1200,
            free_tier_limit=5000,
            cost_per_1000_queries_usd=14.0,
        )
        self.assertEqual(cost, 0.0)

    def test_estimate_cost_over_free_tier(self):
        """Cost should apply only to billable queries."""
        cost = estimate_grounding_cost_usd(
            web_search_queries_count=6500,
            free_tier_limit=5000,
            cost_per_1000_queries_usd=14.0,
        )
        self.assertEqual(cost, 21.0)

    def test_estimate_cost_handles_invalid_input(self):
        """Invalid values should fail safely."""
        cost = estimate_grounding_cost_usd(
            web_search_queries_count="bad-value",
            free_tier_limit="bad-value",
            cost_per_1000_queries_usd="bad-value",
        )
        self.assertEqual(cost, 0.0)
