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
