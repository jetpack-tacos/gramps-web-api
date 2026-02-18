#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      Gramps Web contributors
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#

"""Resources for search grounding admin controls and diagnostics."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

from flask import abort, jsonify
from webargs import fields, validate

from ...auth import SearchGroundingUsageMonthly, config_set, user_db
from ...auth.const import PERM_EDIT_OTHER_USER, PERM_EDIT_SETTINGS, PERM_VIEW_SETTINGS
from ..auth import has_permissions, require_permissions
from ..llm.grounding_policy import (
    VALID_SEARCH_GROUNDING_MODES,
    normalize_search_grounding_mode,
)
from ..llm.grounding_usage import (
    estimate_grounding_cost_usd,
    get_current_month_grounding_usage,
)
from ..util import get_config, use_args
from . import ProtectedResource

SEARCH_GROUNDING_MODE_KEY = "SEARCH_GROUNDING_MODE"
SEARCH_GROUNDING_FREE_TIER_LIMIT_KEY = "SEARCH_GROUNDING_FREE_TIER_LIMIT"
SEARCH_GROUNDING_SOFT_CAP_KEY = "SEARCH_GROUNDING_SOFT_CAP"
SEARCH_GROUNDING_HARD_CAP_KEY = "SEARCH_GROUNDING_HARD_CAP"
SEARCH_GROUNDING_COST_KEY = "SEARCH_GROUNDING_COST_PER_1000_QUERIES_USD"

GROUNDING_DECISION_REASONS = [
    "mode_off",
    "mode_on",
    "scope_out",
    "tree_sufficient",
    "context_gap",
    "context_gap_soft_cap",
    "soft_cap_tightened",
    "cap_blocked_free_tier",
    "cap_blocked_hard",
]


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _coerce_non_negative_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, parsed)


def _require_read_permission() -> None:
    if has_permissions([PERM_VIEW_SETTINGS]):
        return
    if has_permissions([PERM_EDIT_OTHER_USER]):
        return
    abort(403)


def _to_utc_iso(timestamp: datetime | None) -> str | None:
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        dt = timestamp.replace(tzinfo=timezone.utc)
    else:
        dt = timestamp.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _get_next_month_reset_timestamp(period_start: date) -> str:
    if period_start.month == 12:
        dt = datetime(period_start.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        dt = datetime(
            period_start.year, period_start.month + 1, 1, tzinfo=timezone.utc
        )
    return dt.isoformat().replace("+00:00", "Z")


def _get_free_tier_status(queries: int, free_tier_limit: int) -> str:
    if free_tier_limit <= 0:
        return "exhausted" if queries > 0 else "normal"
    if queries >= free_tier_limit:
        return "exhausted"
    near_limit_threshold = max(1, math.ceil(free_tier_limit * 0.1))
    remaining = max(free_tier_limit - queries, 0)
    if remaining <= near_limit_threshold:
        return "near_limit"
    return "normal"


def _get_effective_settings() -> dict[str, Any]:
    return {
        "mode": normalize_search_grounding_mode(get_config(SEARCH_GROUNDING_MODE_KEY)),
        "free_tier_limit": _coerce_non_negative_int(
            get_config(SEARCH_GROUNDING_FREE_TIER_LIMIT_KEY), 5000
        ),
        "soft_cap": _coerce_non_negative_int(
            get_config(SEARCH_GROUNDING_SOFT_CAP_KEY), 5000
        ),
        "hard_cap": _coerce_non_negative_int(
            get_config(SEARCH_GROUNDING_HARD_CAP_KEY), 5000
        ),
        "cost_per_1000_queries_usd": _coerce_non_negative_float(
            get_config(SEARCH_GROUNDING_COST_KEY), 14.0
        ),
    }


def _get_alert_timestamps(period_start: date) -> dict[str, str | None]:
    usage = (
        user_db.session.query(SearchGroundingUsageMonthly)
        .filter_by(period_start=period_start)
        .first()
    )
    if usage is None:
        return {
            "soft_cap_alert_sent_at": None,
            "hard_cap_alert_sent_at": None,
        }
    return {
        "soft_cap_alert_sent_at": _to_utc_iso(usage.soft_cap_alert_sent_at),
        "hard_cap_alert_sent_at": _to_utc_iso(usage.hard_cap_alert_sent_at),
    }


def get_search_grounding_snapshot(
    include_metadata_schema: bool = False,
) -> dict[str, Any]:
    """Build current effective settings, usage counters, and status indicators."""
    settings = _get_effective_settings()
    usage = get_current_month_grounding_usage()
    period_start = date.fromisoformat(usage["period_start"])
    grounded_prompts_count = int(usage.get("grounded_prompts_count") or 0)
    web_search_queries_count = int(usage.get("web_search_queries_count") or 0)

    free_tier_limit = int(settings["free_tier_limit"])
    if free_tier_limit > 0:
        free_tier_used = min(web_search_queries_count, free_tier_limit)
        free_tier_remaining = max(free_tier_limit - web_search_queries_count, 0)
    else:
        free_tier_used = 0
        free_tier_remaining = 0

    alerts = _get_alert_timestamps(period_start)
    estimated_cost_usd = estimate_grounding_cost_usd(
        web_search_queries_count=web_search_queries_count,
        free_tier_limit=free_tier_limit,
        cost_per_1000_queries_usd=settings["cost_per_1000_queries_usd"],
    )

    payload: dict[str, Any] = {
        "effective_mode": settings["mode"],
        "caps": {
            "free_tier_limit": free_tier_limit,
            "soft_cap": int(settings["soft_cap"]),
            "hard_cap": int(settings["hard_cap"]),
        },
        "pricing": {
            "cost_per_1000_queries_usd": float(settings["cost_per_1000_queries_usd"]),
        },
        "usage": {
            "period_start_utc": period_start.isoformat(),
            "resets_at_utc": _get_next_month_reset_timestamp(period_start),
            "grounded_prompts_count": grounded_prompts_count,
            "web_search_queries_count": web_search_queries_count,
            "free_tier_used": free_tier_used,
            "free_tier_remaining": free_tier_remaining,
            "free_tier_status": _get_free_tier_status(
                web_search_queries_count, free_tier_limit
            ),
        },
        "alerts": {
            "soft_cap_reached": bool(
                settings["soft_cap"] > 0 and grounded_prompts_count >= settings["soft_cap"]
            ),
            "hard_cap_reached": bool(
                settings["hard_cap"] > 0 and grounded_prompts_count >= settings["hard_cap"]
            ),
            "soft_cap_alert_sent_at": alerts["soft_cap_alert_sent_at"],
            "hard_cap_alert_sent_at": alerts["hard_cap_alert_sent_at"],
        },
        "estimated_cost_usd_month": estimated_cost_usd,
    }

    if include_metadata_schema:
        payload["metadata_schema_expectations"] = {
            "path": "metadata.grounding",
            "required_fields": [
                "mode",
                "decision_reason",
                "attached",
                "web_search_query_count",
                "estimated_cost_usd_month",
                "alerts_triggered",
            ],
            "optional_fields": [
                "web_search_queries",
            ],
            "mode_values": sorted(VALID_SEARCH_GROUNDING_MODES),
            "decision_reason_values": GROUNDING_DECISION_REASONS,
        }

    return payload


class SearchGroundingSettingsResource(ProtectedResource):
    """Get/update search grounding policy controls."""

    def get(self):
        """Return effective settings and current counters."""
        _require_read_permission()
        return jsonify(get_search_grounding_snapshot()), 200

    @use_args(
        {
            "mode": fields.String(
                required=True, validate=validate.OneOf(sorted(VALID_SEARCH_GROUNDING_MODES))
            ),
            "soft_cap": fields.Integer(required=True, validate=validate.Range(min=0)),
            "hard_cap": fields.Integer(required=True, validate=validate.Range(min=0)),
        },
        location="json",
    )
    def put(self, args):
        """Update grounding mode and cap settings."""
        require_permissions([PERM_EDIT_SETTINGS])
        soft_cap = int(args["soft_cap"])
        hard_cap = int(args["hard_cap"])
        if hard_cap > 0 and soft_cap > hard_cap:
            return jsonify({"error": {"message": "soft_cap cannot exceed hard_cap"}}), 400

        config_set(key=SEARCH_GROUNDING_MODE_KEY, value=args["mode"])
        config_set(key=SEARCH_GROUNDING_SOFT_CAP_KEY, value=str(soft_cap))
        config_set(key=SEARCH_GROUNDING_HARD_CAP_KEY, value=str(hard_cap))
        return jsonify(get_search_grounding_snapshot()), 200


class SearchGroundingDiagnosticsResource(ProtectedResource):
    """Read-only diagnostics for grounding policy state."""

    def get(self):
        """Return effective policy, usage counters, and metadata expectations."""
        _require_read_permission()
        return jsonify(get_search_grounding_snapshot(include_metadata_schema=True)), 200
