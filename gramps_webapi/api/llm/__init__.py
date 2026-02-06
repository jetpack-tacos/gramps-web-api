"""Functions for working with large language models (LLMs)."""

from __future__ import annotations

import re
from typing import Any

from flask import current_app
from google.genai import types

from ..util import abort_with_message, get_logger
from .agent import run_agent
from .deps import AgentDeps


def sanitize_answer(answer: str) -> str:
    """Sanitize the LLM answer."""
    # some models convert relative URLs to absolute URLs with placeholder domains
    answer = answer.replace("https://www.example.com", "")
    answer = answer.replace("https://example.com", "")
    answer = answer.replace("http://example.com", "")

    # Remove forbidden markdown formatting that some models add despite instructions
    # Remove bold: **text** -> text
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    # Remove headers: ### text -> text (at start of line)
    answer = re.sub(r"^#+\s+", "", answer, flags=re.MULTILINE)
    # Remove bullet points: - text or * text -> text (at start of line)
    answer = re.sub(r"^[-*]\s+", "", answer, flags=re.MULTILINE)
    # Remove horizontal rules: --- or *** or ___ (at start of line)
    answer = re.sub(r"^[-*_]{3,}\s*$", "", answer, flags=re.MULTILINE)

    return answer


def extract_metadata_from_result(response: types.GenerateContentResponse) -> dict[str, Any]:
    """Extract metadata from Gemini GenerateContentResponse.

    Args:
        response: GenerateContentResponse from Gemini

    Returns:
        Dictionary containing run metadata including usage info
    """
    metadata: dict[str, Any] = {}

    if response.usage_metadata:
        metadata["usage"] = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }

    return metadata


def _convert_history_to_gemini(history: list[dict]) -> list[types.Content]:
    """Convert client-sent history to Gemini Content objects.

    Args:
        history: List of dicts with 'role' and 'message' keys

    Returns:
        List of Gemini Content objects
    """
    contents: list[types.Content] = []
    for message in history:
        if "role" not in message or "message" not in message:
            raise ValueError(f"Invalid message format: {message}")
        role = message["role"].lower()
        if role in ["ai", "system", "assistant", "model"]:
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=message["message"])],
                )
            )
        elif role != "error":  # skip error messages
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message["message"])],
                )
            )
    return contents


def answer_with_agent(
    prompt: str,
    tree: str,
    include_private: bool,
    user_id: str,
    history: list | None = None,
):
    """Answer a prompt using the Gemini agent.

    Args:
        prompt: The user's question/prompt
        tree: The tree identifier
        include_private: Whether to include private information
        user_id: The user identifier
        history: Optional chat history

    Returns:
        GenerateContentResponse from Gemini
    """
    logger = get_logger()

    config = current_app.config
    model_name = config.get("LLM_MODEL")
    max_context_length = config.get("LLM_MAX_CONTEXT_LENGTH", 50000)
    system_prompt_override = config.get("LLM_SYSTEM_PROMPT")

    if not model_name:
        raise ValueError("No LLM model specified")

    deps = AgentDeps(
        tree=tree,
        include_private=include_private,
        max_context_length=max_context_length,
        user_id=user_id,
    )

    gemini_history: list[types.Content] = []
    if history:
        gemini_history = _convert_history_to_gemini(history)

    try:
        logger.debug("Running Gemini agent with prompt: '%s'", prompt)
        response = run_agent(
            prompt=prompt,
            deps=deps,
            model_name=model_name,
            system_prompt_override=system_prompt_override,
            history=gemini_history,
        )
        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    response_text += part.text
        logger.debug("Gemini response: '%s'", response_text[:200] if response_text else "")
        return response
    except ValueError as e:
        logger.error("Gemini configuration error: %s", e)
        abort_with_message(500, "Error communicating with the AI model")
    except Exception as e:
        logger.error("Unexpected error in Gemini agent: %s", e)
        abort_with_message(500, "Unexpected error.")
