"""Functions for working with large language models (LLMs)."""

from __future__ import annotations

import os
import re
from typing import Any

from flask import current_app
from google import genai
from google.genai import types

from ..util import abort_with_message, get_logger
from .agent import run_agent
from .deps import AgentDeps


INSIGHT_SYSTEM_PROMPT = """You are a genealogy historian analyzing a person's record in the context of their family. You will receive a person's full record along with their parents, siblings, spouse(s), and children.

Your job is to generate 3-5 short observations that are SURPRISING and INTERESTING — things the user would never notice by just reading the facts on the page.

RULES:
- NEVER restate facts that are already on the page (birth date, death date, number of children). The user can already see those.
- NEVER do basic arithmetic ("He lived to age 77"). That's obvious.
- NEVER fabricate or assume facts not present in the data you received. Every claim must be grounded in the records provided. If a sibling's cause of death is not in the data, do not guess it. If an occupation is not listed, do not invent one. You may connect real dates/places to well-known historical events, but clearly distinguish between what the records show and what you are inferring from historical context.
- DO cross-reference this person with their relatives. Compare ages, dates, places, names. Look for patterns and coincidences.
- DO connect dates and places to historical events when you can make a specific, plausible connection (not generic history). Frame these as context, not as established facts about this person (e.g., "This was during the period when..." not "He experienced...").
- DO notice naming patterns — children named after grandparents, deceased siblings' names being reused, patronymic traditions.
- DO notice occupation changes across a person's life or across generations.
- DO notice when siblings had very different or very similar life paths.
- DO note data quality issues, but only if you can say something specific ("No source for birth — the county courthouse burned in 1897").

PRIORITIES (most interesting first):
1. Coincidences between this person and their family (same names, same places, same life events)
2. Connections between their life events and specific historical events
3. Patterns across generations (naming, occupation, migration, family size)
4. What their occupation or community meant in that specific time and place
5. Data quality notes (only if specific and actionable)

FORMAT:
- Write 3-5 short paragraphs in plain prose.
- Use person links in markdown format where relevant: [Name](/person/GRAMPS_ID)
- Do not use bullet points, numbered lists, headers, or bold text.
- Each paragraph should be one observation — a mini "Did you know...?" moment."""


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


def generate_insight(
    person_context: str,
    tree: str,
    include_private: bool,
    user_id: str,
) -> tuple[str, dict[str, Any]]:
    """Generate AI insight for a person using Gemini (no tools, single-shot).

    Args:
        person_context: Pre-formatted plain text context for the person + family
        tree: The tree identifier
        include_private: Whether to include private information
        user_id: The user identifier

    Returns:
        Tuple of (insight_text, metadata_dict)
    """
    logger = get_logger()

    config = current_app.config
    model_name = config.get("LLM_MODEL")

    if not model_name:
        raise ValueError("No LLM model specified")

    api_key = os.environ.get("GEMINI_API_KEY") or config.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key configured")

    client = genai.Client(api_key=api_key)

    try:
        logger.debug(
            "Generating insight for person context (%d chars)", len(person_context)
        )
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=person_context)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=INSIGHT_SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )

        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    response_text += part.text

        metadata = extract_metadata_from_result(response)
        logger.debug(
            "Insight generated: '%s'",
            response_text[:200] if response_text else "",
        )

        # Sanitize the response
        response_text = sanitize_answer(response_text)

        return response_text, metadata

    except ValueError as e:
        logger.error("Gemini configuration error during insight generation: %s", e)
        abort_with_message(500, "Error communicating with the AI model")
    except Exception as e:
        logger.error("Unexpected error during insight generation: %s", e)
        abort_with_message(500, "Unexpected error generating insight.")
