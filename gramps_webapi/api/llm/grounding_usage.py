"""Helpers for recording monthly web-grounding usage."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError

from ...auth import SearchGroundingUsageMonthly, user_db


def get_utc_month_start(now: datetime | None = None) -> date:
    """Return the first day of the current UTC month."""
    dt = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return date(dt.year, dt.month, 1)


def estimate_grounding_cost_usd(
    web_search_queries_count: int,
    free_tier_limit: Any,
    cost_per_1000_queries_usd: Any,
) -> float:
    """Estimate monthly grounding cost in USD."""
    try:
        queries = max(0, int(web_search_queries_count))
    except (TypeError, ValueError):
        queries = 0

    try:
        free_limit = int(free_tier_limit)
        if free_limit < 0:
            free_limit = 0
    except (TypeError, ValueError):
        free_limit = 0

    try:
        rate = float(cost_per_1000_queries_usd)
        if rate < 0:
            rate = 0.0
    except (TypeError, ValueError):
        rate = 0.0

    billable_queries = max(0, queries - free_limit)
    return round((billable_queries / 1000.0) * rate, 6)


def record_monthly_grounding_usage(
    grounding_attached: bool,
    web_search_query_count: int,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Upsert monthly usage counters and return the updated snapshot."""
    grounded_delta = 1 if grounding_attached else 0
    query_delta = max(0, int(web_search_query_count))
    period_start = get_utc_month_start(now=now)

    if grounded_delta == 0 and query_delta == 0:
        return None

    for _ in range(2):
        usage = (
            user_db.session.query(SearchGroundingUsageMonthly)
            .filter_by(period_start=period_start)
            .first()
        )
        if usage is None:
            usage = SearchGroundingUsageMonthly(
                period_start=period_start,
                grounded_prompts_count=grounded_delta,
                web_search_queries_count=query_delta,
            )
            user_db.session.add(usage)
        else:
            usage.grounded_prompts_count += grounded_delta
            usage.web_search_queries_count += query_delta

        try:
            user_db.session.commit()
            return {
                "period_start": period_start.isoformat(),
                "grounded_prompts_count": usage.grounded_prompts_count,
                "web_search_queries_count": usage.web_search_queries_count,
                "grounded_prompts_delta": grounded_delta,
                "web_search_queries_delta": query_delta,
            }
        except IntegrityError:
            user_db.session.rollback()

    # Final read in case a concurrent writer won the race twice.
    usage = (
        user_db.session.query(SearchGroundingUsageMonthly)
        .filter_by(period_start=period_start)
        .first()
    )
    if usage is None:
        raise RuntimeError("Failed to record grounding usage for current month")
    return {
        "period_start": period_start.isoformat(),
        "grounded_prompts_count": usage.grounded_prompts_count,
        "web_search_queries_count": usage.web_search_queries_count,
        "grounded_prompts_delta": grounded_delta,
        "web_search_queries_delta": query_delta,
    }


def get_current_month_grounding_usage(
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return current UTC month usage counters."""
    period_start = get_utc_month_start(now=now)
    usage = (
        user_db.session.query(SearchGroundingUsageMonthly)
        .filter_by(period_start=period_start)
        .first()
    )
    if usage is None:
        grounded_prompts_count = 0
        web_search_queries_count = 0
    else:
        grounded_prompts_count = int(usage.grounded_prompts_count or 0)
        web_search_queries_count = int(usage.web_search_queries_count or 0)
    return {
        "period_start": period_start.isoformat(),
        "grounded_prompts_count": grounded_prompts_count,
        "web_search_queries_count": web_search_queries_count,
    }


def maybe_record_threshold_alerts(
    period_start: str | date | None,
    grounded_prompts_count: int,
    soft_cap: Any,
    hard_cap: Any,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record threshold alerts once per month per threshold."""
    if period_start is None:
        return {"triggered": []}

    if isinstance(period_start, str):
        parsed_period = date.fromisoformat(period_start)
    else:
        parsed_period = period_start

    try:
        grounded_count = int(grounded_prompts_count)
    except (TypeError, ValueError):
        grounded_count = 0

    def _parse_cap(value: Any) -> int | None:
        try:
            cap = int(value)
        except (TypeError, ValueError):
            return None
        if cap <= 0:
            return None
        return cap

    soft_limit = _parse_cap(soft_cap)
    hard_limit = _parse_cap(hard_cap)

    usage = (
        user_db.session.query(SearchGroundingUsageMonthly)
        .filter_by(period_start=parsed_period)
        .first()
    )
    if usage is None:
        return {"triggered": []}

    dt = now or datetime.now(timezone.utc)
    triggered: list[str] = []

    if (
        soft_limit is not None
        and grounded_count >= soft_limit
        and usage.soft_cap_alert_sent_at is None
    ):
        usage.soft_cap_alert_sent_at = dt
        triggered.append("soft_cap")

    if (
        hard_limit is not None
        and grounded_count >= hard_limit
        and usage.hard_cap_alert_sent_at is None
    ):
        usage.hard_cap_alert_sent_at = dt
        triggered.append("hard_cap")

    if triggered:
        user_db.session.commit()

    return {
        "triggered": triggered,
        "soft_cap_alert_sent_at": (
            usage.soft_cap_alert_sent_at.isoformat()
            if usage.soft_cap_alert_sent_at is not None
            else None
        ),
        "hard_cap_alert_sent_at": (
            usage.hard_cap_alert_sent_at.isoformat()
            if usage.hard_cap_alert_sent_at is not None
            else None
        ),
    }
