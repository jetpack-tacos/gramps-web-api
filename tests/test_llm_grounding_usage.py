"""Tests for grounding usage helper utilities."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from sqlalchemy.exc import IntegrityError

from gramps_webapi.api.llm.grounding_usage import (
    estimate_grounding_cost_usd,
    get_current_month_grounding_usage,
    record_monthly_grounding_usage,
)
from gramps_webapi.app import create_app
from gramps_webapi.auth import SearchGroundingUsageMonthly, user_db
from gramps_webapi.const import ENV_CONFIG_FILE, TEST_AUTH_CONFIG


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


class TestGroundingUsageRetryPath(unittest.TestCase):
    """Stage 0 verification tests for retry behavior."""

    def setUp(self):
        with patch.dict(
            "os.environ",
            {
                ENV_CONFIG_FILE: TEST_AUTH_CONFIG,
                "USER_DB_URI": "sqlite://",
                "GRAMPSWEB_USER_DB_URI": "sqlite://",
            },
        ):
            self.app = create_app(
                config={"TESTING": True, "RATELIMIT_ENABLED": False},
                config_from_env=False,
            )
        self.ctx = self.app.test_request_context()
        self.ctx.push()
        user_db.create_all()

    def tearDown(self):
        user_db.session.remove()
        user_db.drop_all()
        self.ctx.pop()

    def test_record_monthly_grounding_usage_retry_commit_path_is_single_increment(self):
        """A commit retry must produce a single monthly counter increment."""
        now = datetime(2026, 2, 18, 12, 30, tzinfo=timezone.utc)
        original_commit = user_db.session.commit
        commit_calls = {"count": 0}

        def flaky_commit():
            commit_calls["count"] += 1
            if commit_calls["count"] == 1:
                raise IntegrityError(
                    "insert into search_grounding_usage_monthly (...)",
                    {},
                    Exception("simulated unique race"),
                )
            return original_commit()

        with patch.object(user_db.session, "commit", side_effect=flaky_commit):
            snapshot = record_monthly_grounding_usage(
                grounding_attached=True,
                web_search_query_count=3,
                now=now,
            )

        self.assertEqual(commit_calls["count"], 2)
        self.assertEqual(snapshot["grounded_prompts_count"], 1)
        self.assertEqual(snapshot["web_search_queries_count"], 3)

        usage = get_current_month_grounding_usage(now=now)
        self.assertEqual(usage["grounded_prompts_count"], 1)
        self.assertEqual(usage["web_search_queries_count"], 3)
        self.assertEqual(
            user_db.session.query(SearchGroundingUsageMonthly).count(),
            1,
        )
