"""Policy helpers for deciding chat grounding behavior."""

from __future__ import annotations

from typing import Any


VALID_SEARCH_GROUNDING_MODES = {"off", "auto", "on"}


def normalize_search_grounding_mode(raw_mode: Any) -> str:
    """Normalize and validate the configured grounding mode."""
    mode = str(raw_mode or "auto").strip().lower()
    if mode not in VALID_SEARCH_GROUNDING_MODES:
        return "auto"
    return mode


def decide_chat_grounding(raw_mode: Any) -> dict[str, Any]:
    """Return Stage 1 grounding decision details for chat requests."""
    mode = normalize_search_grounding_mode(raw_mode)
    if mode == "off":
        return {
            "mode": mode,
            "grounding_attached": False,
            "decision_reason": "mode_off",
        }
    if mode == "on":
        return {
            "mode": mode,
            "grounding_attached": True,
            "decision_reason": "mode_on",
        }
    return {
        "mode": mode,
        "grounding_attached": True,
        "decision_reason": "auto_stage1_default",
    }
