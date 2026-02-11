"""Functions for working with large language models (LLMs)."""

from __future__ import annotations

import os
import re
from typing import Any

from flask import current_app
from google import genai
from google.genai import types

from ..util import abort_with_message, get_logger
from .agent import run_agent, _ToolContext
from .deps import AgentDeps
from .tools import (
    get_tree_statistics,
    find_coincidences_and_clusters,
    get_person_full_details,
    analyze_migration_patterns,
)


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
        logger.info("Gemini response (%d chars): %s", len(response_text), response_text[:500])
        if len(response_text) > 500:
            logger.info("Full Gemini response: %s", response_text)
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


NUGGET_SYSTEM_PROMPT = """You are a genealogy historian. You will receive pre-gathered data about a family tree: statistics and interesting patterns/coincidences.

Your job is to turn this raw data into exactly 30 short, fascinating nuggets (1-2 sentences each, under 40 words).

RULES:
- Base nuggets ONLY on the data provided. Do NOT fabricate names, dates, or facts.
- Each nugget should highlight something surprising, dramatic, or humanizing.
- Include the Gramps ID of the most relevant person in [brackets] at the end.
- If a finding doesn't have a clear person ID, use [TREE] instead.
- Number each nugget 1-30.

GOOD NUGGETS:
- "Three Olson sisters all married men from the same Norwegian parish within two years of each other. [I0234]"
- "Johan was listed as a drayman in 1895 but an automobile mechanic by 1912 — one of the earliest to make the switch. [I0567]"

BAD NUGGETS (do not write these):
- "The oldest person lived to 94." (boring statistic)
- "Many people were born in Sweden." (too vague)
- "The family tree has 5,235 people." (just a number)

FORMAT:
1. Nugget text [GRAMPS_ID]
2. Nugget text [GRAMPS_ID]
...
30. Nugget text [GRAMPS_ID]"""


def generate_nuggets(
    tree: str,
    include_private: bool,
    user_id: str,
) -> tuple[str, dict[str, Any]]:
    """Generate interesting nuggets using single-shot Gemini call (no tools).

    Pre-gathers tree context by calling analytical tools directly in Python,
    then sends that context to Gemini in one shot. This avoids the agent
    tool-calling loop which takes 2+ minutes and times out Gunicorn.

    Args:
        tree: The tree identifier
        include_private: Whether to include private information
        user_id: The user identifier

    Returns:
        Tuple of (nuggets_text, metadata_dict)
    """
    logger = get_logger()

    config = current_app.config
    model_name = config.get("LLM_MODEL")

    if not model_name:
        raise ValueError("No LLM model specified")

    api_key = os.environ.get("GEMINI_API_KEY") or config.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key configured")

    # Pre-gather tree context by calling tools directly (no agent loop)
    deps = AgentDeps(
        tree=tree,
        include_private=include_private,
        max_context_length=config.get("LLM_MAX_CONTEXT_LENGTH", 50000),
        user_id=user_id,
    )
    ctx = _ToolContext(deps)

    logger.info("Gathering tree context for nugget generation...")
    stats_text = get_tree_statistics(ctx)
    logger.info("Tree statistics gathered (%d chars)", len(stats_text))

    clusters_text = find_coincidences_and_clusters(ctx, category="all", max_results=15)
    logger.info("Coincidences/clusters gathered (%d chars)", len(clusters_text))

    context = f"TREE STATISTICS:\n{stats_text}\n\nINTERESTING PATTERNS AND COINCIDENCES:\n{clusters_text}"

    client = genai.Client(api_key=api_key)

    try:
        logger.info("Sending single-shot nugget generation request to Gemini...")
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=context)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=NUGGET_SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )

        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    response_text += part.text

        metadata = extract_metadata_from_result(response)
        logger.info("Nuggets generated (%d chars): %s", len(response_text), response_text[:500])

        return response_text, metadata

    except ValueError as e:
        logger.error("Gemini configuration error during nugget generation: %s", e)
        abort_with_message(500, "Error communicating with the AI model")
    except Exception as e:
        logger.error("Unexpected error during nugget generation: %s", e)
        abort_with_message(500, "Unexpected error generating nuggets.")


BLOG_SYSTEM_PROMPT = """You are a family historian writing an engaging blog post for a genealogy website. You will receive rich data about specific people, patterns, and events from the family tree.

Your job is to write a compelling, narrative blog post (500-800 words) that brings ancestors to life.

WRITING STYLE:
- Write like a storytelling historian, not a data report
- Open with a hook that draws the reader in
- Weave specific names, dates, and places into the narrative
- Connect family events to the broader historical context of the time
- Use person links in markdown format: [Name](/person/GRAMPS_ID)
- End with an invitation for family members to explore further or share memories
- Make ancestors feel like real people with real lives, not just records

RULES:
- Base the story ONLY on the data provided. Do NOT fabricate names, dates, or facts.
- You MAY add historical context about the time period and places mentioned — this is encouraged.
- Clearly distinguish between what the records show and what you are inferring from history.
- Do NOT use numbered lists or bullet points in the blog post itself.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
TITLE: [A compelling, specific title — not generic]

CONTENT:
[The blog post content here. Use markdown links to people: [Name](/person/GRAMPS_ID)]"""


def generate_blog_post(
    tree: str,
    include_private: bool,
    user_id: str,
    previous_titles: list[str] | None = None,
    previously_featured_ids: set[str] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Generate a blog post using data-driven topic selection and single-shot Gemini call.

    Discovers interesting patterns in the tree, picks the most narratively rich
    finding, gathers detailed person records for the people involved, and sends
    everything to Gemini in a single call with Google Search for historical context.

    Args:
        tree: The tree identifier
        include_private: Whether to include private information
        user_id: The user identifier
        previous_titles: Titles of existing blog posts (for diversity)
        previously_featured_ids: Gramps IDs already featured in existing posts

    Returns:
        Tuple of (title, content, metadata_dict)
    """
    logger = get_logger()

    config = current_app.config
    model_name = config.get("LLM_MODEL")

    if not model_name:
        raise ValueError("No LLM model specified")

    api_key = os.environ.get("GEMINI_API_KEY") or config.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key configured")

    # Pre-gather tree context by calling tools directly
    deps = AgentDeps(
        tree=tree,
        include_private=include_private,
        max_context_length=config.get("LLM_MAX_CONTEXT_LENGTH", 50000),
        user_id=user_id,
    )
    ctx = _ToolContext(deps)

    import re as _re
    import random

    logger.info("Gathering context for blog post generation...")

    # Step 1: Gather all findings from analytical tools
    clusters_text = find_coincidences_and_clusters(ctx, category="all", max_results=10)
    logger.info("Coincidences/clusters gathered (%d chars)", len(clusters_text))

    stats_text = get_tree_statistics(ctx)
    logger.info("Tree statistics gathered (%d chars)", len(stats_text))

    migration_text = analyze_migration_patterns(ctx)
    logger.info("Migration patterns gathered (%d chars)", len(migration_text))

    # Step 2: Split findings into individual items and select a random subset
    # Each finding is separated by double newline in the tool output
    all_findings = []
    for text in [clusters_text, migration_text]:
        items = [f.strip() for f in text.split("\n\n") if f.strip()]
        all_findings.extend(items)

    featured = previously_featured_ids or set()

    # Score findings: prefer those mentioning unfeatured people or events
    def finding_score(finding):
        ids_in_finding = set(_re.findall(r'\b(I\d{4,5})\b', finding))
        # Penalize findings where all people are already featured
        if ids_in_finding and ids_in_finding.issubset(featured):
            return 0
        # Bonus for findings with unfeatured people
        unfeatured_count = len(ids_in_finding - featured)
        return unfeatured_count + 1  # +1 so event-only findings still score

    scored = [(finding_score(f), f) for f in all_findings]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top-scoring findings, then randomly pick 2-3 from the top half
    top_half = scored[:max(len(scored) // 2, 3)]
    random.shuffle(top_half)
    selected_findings = [f for _, f in top_half[:3]]

    logger.info(
        "Selected %d findings from %d total (avoiding %d previously featured people)",
        len(selected_findings), len(all_findings), len(featured),
    )

    # Step 3: Fetch person details only for people mentioned in selected findings
    selected_text = "\n\n".join(selected_findings)
    gramps_ids = list(set(_re.findall(r'\b(I\d{4,5})\b', selected_text)))
    random.shuffle(gramps_ids)

    person_details = []
    for gid in gramps_ids[:5]:
        details = get_person_full_details(ctx, gramps_id=gid)
        if details and "Error" not in details and "private" not in details.lower():
            person_details.append(f"--- PERSON {gid} ---\n{details}")
    logger.info("Gathered details for %d people from selected findings", len(person_details))

    # Step 4: Build the context — selected findings drive the topic
    diversity_text = ""
    if previous_titles:
        diversity_text += (
            "\n\nPREVIOUS BLOG POST TITLES (write about something DIFFERENT):\n"
            + "\n".join(f"- {t}" for t in previous_titles[-15:])
        )

    context = (
        f"TREE STATISTICS (for background):\n{stats_text}\n\n"
        f"SELECTED FINDINGS TO WRITE ABOUT:\n{selected_text}\n\n"
        f"DETAILED PERSON RECORDS (for people mentioned in findings):\n"
        + "\n\n".join(person_details)
        + diversity_text
        + "\n\nWrite a blog post based on the SELECTED FINDINGS above. "
        "The findings may be about events, patterns, or phenomena — not just people. "
        "Weave in the person details and historical context to bring the story to life."
    )

    client = genai.Client(api_key=api_key)

    try:
        logger.info("Sending blog generation request to Gemini (with Google Search)...")
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=context)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=BLOG_SYSTEM_PROMPT,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.8,
            ),
        )

        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    response_text += part.text

        metadata = extract_metadata_from_result(response)
        logger.info("Blog post generated (%d chars): %s", len(response_text), response_text[:300])

        # Parse TITLE and CONTENT
        title = "Family History Insights"
        content = response_text

        if "TITLE:" in response_text:
            parts = response_text.split("TITLE:", 1)
            if len(parts) > 1:
                title_section = parts[1]
                if "CONTENT:" in title_section:
                    title_parts = title_section.split("CONTENT:", 1)
                    title = title_parts[0].strip()
                    content = title_parts[1].strip()
                else:
                    lines = title_section.strip().split('\n', 1)
                    title = lines[0].strip()
                    content = lines[1].strip() if len(lines) > 1 else ""

        return title, content, metadata

    except ValueError as e:
        logger.error("Gemini configuration error during blog generation: %s", e)
        abort_with_message(500, "Error communicating with the AI model")
    except Exception as e:
        logger.error("Unexpected error during blog generation: %s", e)
        abort_with_message(500, "Unexpected error generating blog post.")
