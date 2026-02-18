"""Tests for grounding threshold alert and month-reset behavior."""

from datetime import date, datetime, timezone
import unittest
from unittest.mock import patch

from gramps_webapi.api.llm.grounding_usage import (
    get_current_month_grounding_usage,
    get_utc_month_start,
    maybe_record_threshold_alerts,
    record_monthly_grounding_usage,
)
from gramps_webapi.app import create_app
from gramps_webapi.auth import SearchGroundingUsageMonthly, user_db
from gramps_webapi.const import ENV_CONFIG_FILE, TEST_AUTH_CONFIG


class TestGroundingAlerts(unittest.TestCase):
    """Stage 3 guardrail verification tests."""

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

    def test_soft_cap_alert_fires_once_per_month(self):
        now = datetime(2026, 2, 15, 9, 0, tzinfo=timezone.utc)
        period_start = get_utc_month_start(now=now)

        user_db.session.add(
            SearchGroundingUsageMonthly(
                period_start=period_start,
                grounded_prompts_count=4200,
                web_search_queries_count=3500,
            )
        )
        user_db.session.commit()

        first = maybe_record_threshold_alerts(
            period_start=period_start,
            grounded_prompts_count=4200,
            soft_cap=4000,
            hard_cap=5000,
            now=now,
        )
        self.assertEqual(first["triggered"], ["soft_cap"])
        self.assertIsNotNone(first["soft_cap_alert_sent_at"])
        self.assertIsNone(first["hard_cap_alert_sent_at"])

        second = maybe_record_threshold_alerts(
            period_start=period_start,
            grounded_prompts_count=4300,
            soft_cap=4000,
            hard_cap=5000,
            now=now,
        )
        self.assertEqual(second["triggered"], [])

    def test_hard_cap_alert_fires_once_per_month(self):
        now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
        period_start = get_utc_month_start(now=now)

        user_db.session.add(
            SearchGroundingUsageMonthly(
                period_start=period_start,
                grounded_prompts_count=5200,
                web_search_queries_count=7000,
            )
        )
        user_db.session.commit()

        first = maybe_record_threshold_alerts(
            period_start=period_start,
            grounded_prompts_count=5200,
            soft_cap=0,
            hard_cap=5000,
            now=now,
        )
        self.assertEqual(first["triggered"], ["hard_cap"])
        self.assertIsNotNone(first["hard_cap_alert_sent_at"])

        second = maybe_record_threshold_alerts(
            period_start=period_start,
            grounded_prompts_count=5300,
            soft_cap=0,
            hard_cap=5000,
            now=now,
        )
        self.assertEqual(second["triggered"], [])

    def test_monthly_usage_resets_across_month_boundary(self):
        feb = datetime(2026, 2, 20, 8, 0, tzinfo=timezone.utc)
        mar = datetime(2026, 3, 2, 8, 0, tzinfo=timezone.utc)

        record_monthly_grounding_usage(
            grounding_attached=True,
            web_search_query_count=3,
            now=feb,
        )
        feb_usage = get_current_month_grounding_usage(now=feb)
        mar_usage = get_current_month_grounding_usage(now=mar)

        self.assertEqual(feb_usage["period_start"], date(2026, 2, 1).isoformat())
        self.assertEqual(feb_usage["grounded_prompts_count"], 1)
        self.assertEqual(feb_usage["web_search_queries_count"], 3)

        self.assertEqual(mar_usage["period_start"], date(2026, 3, 1).isoformat())
        self.assertEqual(mar_usage["grounded_prompts_count"], 0)
        self.assertEqual(mar_usage["web_search_queries_count"], 0)
