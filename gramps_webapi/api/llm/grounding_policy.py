"""Policy helpers for deciding chat grounding behavior."""

from __future__ import annotations

import re
from typing import Any


VALID_SEARCH_GROUNDING_MODES = {"off", "auto", "on"}
OUT_OF_SCOPE_REFUSAL_MESSAGE = (
    "I can only help with genealogy and family-history questions. "
    "If you connect your question to a person, place, time period, or event in your tree, "
    "I can help with that."
)

GENEALOGY_SCOPE_KEYWORDS = (
    "genealogy",
    "family tree",
    "ancestor",
    "ancestors",
    "descendant",
    "descendants",
    "lineage",
    "surname",
    "birth",
    "death",
    "marriage",
    "spouse",
    "parents",
    "parent",
    "children",
    "child",
    "sibling",
    "grandfather",
    "grandmother",
    "grandparent",
    "census",
    "immigration",
    "emigration",
    "migration",
    "parish",
    "record",
    "obituary",
    "grave",
    "cemetery",
    "occupation",
    "tree",
)

CONTEXT_GAP_KEYWORDS = (
    "historical context",
    "context",
    "what was life like",
    "why did",
    "why was",
    "because",
    "migration pattern",
    "migration patterns",
    "moved from",
    "moved to",
    "war",
    "famine",
    "plague",
    "epidemic",
    "economic",
    "industry",
    "occupation history",
    "place name change",
    "border change",
    "history of",
)

HIGH_CONFIDENCE_CONTEXT_GAP_KEYWORDS = (
    "historical context",
    "what was life like",
    "why did",
    "migration pattern",
    "migration patterns",
    "moved from",
    "moved to",
    "place name change",
    "border change",
)

OUT_OF_SCOPE_HINTS = (
    "movie",
    "film",
    "reviews",
    "box office",
    "sports",
    "nfl",
    "nba",
    "mlb",
    "nhl",
    "weather",
    "forecast",
    "stock",
    "crypto",
    "bitcoin",
    "recipe",
    "restaurant",
    "game",
    "video game",
    "programming",
    "code bug",
)


def _looks_like_person_name(query: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", query))


def _is_in_genealogy_scope(query: str) -> bool:
    normalized = query.lower().strip()
    if not normalized:
        return True

    has_genealogy_keyword = any(word in normalized for word in GENEALOGY_SCOPE_KEYWORDS)
    has_out_of_scope_hint = any(word in normalized for word in OUT_OF_SCOPE_HINTS)
    has_person_name = _looks_like_person_name(query)

    if has_out_of_scope_hint and not has_genealogy_keyword and not has_person_name:
        return False
    return True


def _is_context_gap_query(query: str) -> bool:
    normalized = query.lower().strip()
    if not normalized:
        return False
    return any(word in normalized for word in CONTEXT_GAP_KEYWORDS)


def _is_high_confidence_context_gap_query(query: str) -> bool:
    normalized = query.lower().strip()
    if not normalized:
        return False
    return any(word in normalized for word in HIGH_CONFIDENCE_CONTEXT_GAP_KEYWORDS)


def _normalize_limit(value: Any) -> int | None:
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    if limit <= 0:
        return None
    return limit


def normalize_search_grounding_mode(raw_mode: Any) -> str:
    """Normalize and validate the configured grounding mode."""
    mode = str(raw_mode or "auto").strip().lower()
    if mode not in VALID_SEARCH_GROUNDING_MODES:
        return "auto"
    return mode


def decide_chat_grounding(
    raw_mode: Any,
    query: str,
    current_grounded_prompts_count: int = 0,
    free_tier_limit: Any = None,
    soft_cap: Any = None,
    hard_cap: Any = None,
) -> dict[str, Any]:
    """Return grounding decision details for chat requests."""
    mode = normalize_search_grounding_mode(raw_mode)
    in_scope = _is_in_genealogy_scope(query)

    if not in_scope:
        return {
            "mode": mode,
            "grounding_attached": False,
            "decision_reason": "scope_out",
            "should_refuse": True,
            "refusal_message": OUT_OF_SCOPE_REFUSAL_MESSAGE,
        }

    if mode == "off":
        return {
            "mode": mode,
            "grounding_attached": False,
            "decision_reason": "mode_off",
            "should_refuse": False,
            "refusal_message": None,
        }
    if mode == "on":
        return {
            "mode": mode,
            "grounding_attached": True,
            "decision_reason": "mode_on",
            "should_refuse": False,
            "refusal_message": None,
        }

    free_limit = _normalize_limit(free_tier_limit)
    if (
        free_limit is not None
        and int(current_grounded_prompts_count) >= free_limit
    ):
        return {
            "mode": mode,
            "grounding_attached": False,
            "decision_reason": "cap_blocked_free_tier",
            "should_refuse": False,
            "refusal_message": None,
        }

    hard_limit = _normalize_limit(hard_cap)
    if (
        hard_limit is not None
        and int(current_grounded_prompts_count) >= hard_limit
    ):
        return {
            "mode": mode,
            "grounding_attached": False,
            "decision_reason": "cap_blocked_hard",
            "should_refuse": False,
            "refusal_message": None,
        }

    soft_limit = _normalize_limit(soft_cap)
    in_soft_tightening = (
        soft_limit is not None
        and int(current_grounded_prompts_count) >= soft_limit
    )

    if _is_context_gap_query(query):
        if in_soft_tightening and not _is_high_confidence_context_gap_query(query):
            decision_reason = "soft_cap_tightened"
            grounding_attached = False
        elif in_soft_tightening:
            decision_reason = "context_gap_soft_cap"
            grounding_attached = True
        else:
            decision_reason = "context_gap"
            grounding_attached = True
    else:
        decision_reason = "tree_sufficient"
        grounding_attached = False

    return {
        "mode": mode,
        "grounding_attached": grounding_attached,
        "decision_reason": decision_reason,
        "should_refuse": False,
        "refusal_message": None,
    }
