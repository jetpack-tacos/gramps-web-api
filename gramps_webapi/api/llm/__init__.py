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


INSIGHT_SYSTEM_PROMPT = """You're looking at a person's genealogy record along with their family context (parents, siblings, spouses, children). Your job is to spot the stuff that's actually interesting — the things a user would never notice scrolling through dates and places.

Write 3-5 short observations. Think "cocktail party anecdote" not "encyclopedia entry." Be witty when the material calls for it. A little dry humor goes a long way.

WHAT TO LOOK FOR:
- Cross-reference with relatives: compare ages, dates, places, names. Coincidences are gold.
- Historical context: connect dates/places to specific events (not generic history). Frame as context, not biography ("This was smack in the middle of..." not "He experienced...").
- Naming patterns: kids named after grandparents, necronyms, patronymics.
- Occupation shifts across a life or across generations.
- Siblings with wildly different or eerily similar life paths.
- Data quality quirks, but only if specific ("No birth source — the courthouse burned in 1897, which explains a lot").

RULES:
- NEVER restate facts already visible on the page (birth date, death date, number of children).
- NEVER do basic arithmetic ("He lived to age 77"). The user has a calculator.
- NEVER fabricate facts. Every claim must be grounded in the data provided. You may infer historical context but clearly distinguish inference from record.
- Use person links: [Name](/person/GRAMPS_ID)
- No bullet points, numbered lists, headers, or bold text.
- Each paragraph = one observation, written in plain prose with personality."""


CONNECTIONS_SYSTEM_PROMPT = """You are writing a "connections" narrative for one person in a family tree.

You will receive:
- The viewed person's Gramps ID
- A set of findings that already mention the viewed person or their immediate family

Write 2-3 short paragraphs explaining the most interesting connections to other people, places, time periods, or recurring family patterns.

RULES:
- Base every claim only on the supplied findings.
- Keep it lively and specific, not generic.
- Always use person links in this exact format: [Name](/person/GRAMPS_ID)
- Do not use bullet points, numbered lists, headers, or bold text.
- Do not repeat the same finding in different words."""


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


def _is_mock_object(value: Any) -> bool:
    """Return True if this value looks like a unittest.mock object."""
    return value.__class__.__module__.startswith("unittest.mock")


def _extract_web_search_queries(
    response: types.GenerateContentResponse,
) -> list[str]:
    """Extract Google Search queries from Gemini grounding metadata."""
    if not response.candidates:
        return []

    candidate = response.candidates[0]
    grounding_metadata = getattr(candidate, "grounding_metadata", None)
    if grounding_metadata is None or _is_mock_object(grounding_metadata):
        return []

    raw_queries = getattr(grounding_metadata, "web_search_queries", None)
    if raw_queries is None or _is_mock_object(raw_queries):
        return []

    if isinstance(raw_queries, list):
        query_entries = raw_queries
    else:
        try:
            query_entries = list(raw_queries)
        except TypeError:
            return []

    queries: list[str] = []
    for entry in query_entries:
        if isinstance(entry, str):
            query = entry
        elif isinstance(entry, dict):
            query = entry.get("text") or entry.get("query") or ""
        else:
            query = getattr(entry, "text", None) or getattr(entry, "query", None) or ""
        if query:
            queries.append(str(query))
    return queries


def extract_grounding_stats_from_result(
    response: types.GenerateContentResponse,
) -> dict[str, Any]:
    """Extract web-grounding statistics from a Gemini response."""
    queries = _extract_web_search_queries(response)
    return {
        "web_search_query_count": len(queries),
        "web_search_queries": queries,
    }


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

    grounding = extract_grounding_stats_from_result(response)
    metadata["grounding"] = {
        "web_search_query_count": grounding["web_search_query_count"],
    }
    if grounding["web_search_queries"]:
        metadata["grounding"]["web_search_queries"] = grounding["web_search_queries"]

    return metadata


def extract_text_from_response(response: types.GenerateContentResponse) -> str:
    """Extract the text content from a Gemini response."""
    if not response.candidates or not response.candidates[0].content.parts:
        return ""
    return "".join(part.text for part in response.candidates[0].content.parts if part.text)


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
    grounding_enabled: bool = True,
):
    """Answer a prompt using the Gemini agent.

    Args:
        prompt: The user's question/prompt
        tree: The tree identifier
        include_private: Whether to include private information
        user_id: The user identifier
        history: Optional chat history
        grounding_enabled: Whether to attach Google Search grounding

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
            grounding_enabled=grounding_enabled,
        )
        response_text = extract_text_from_response(response)
        logger.info("Gemini response (%d chars): %s", len(response_text), response_text[:500])
        if len(response_text) > 500:
            logger.info("Full Gemini response: %s", response_text)
        return response
    except ValueError as e:
        logger.error("Gemini configuration error: %s", e)
        raise RuntimeError("Error communicating with the AI model") from e
    except Exception as e:
        logger.error("Unexpected error in Gemini agent: %s", e)
        raise RuntimeError("Unexpected error.") from e


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

        response_text = extract_text_from_response(response)

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


def generate_person_connections(
    person_gramps_id: str,
    findings_text: str,
    tree: str,
    include_private: bool,
    user_id: str,
) -> tuple[str, dict[str, Any]]:
    """Generate AI "person connections" narrative via single-shot Gemini."""
    logger = get_logger()

    config = current_app.config
    model_name = config.get("LLM_MODEL")

    if not model_name:
        raise ValueError("No LLM model specified")

    api_key = os.environ.get("GEMINI_API_KEY") or config.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key configured")

    context = (
        f"VIEWED PERSON: {person_gramps_id}\n\n"
        "SCOPED FINDINGS (already filtered to this person/immediate family):\n"
        f"{findings_text}"
    )

    client = genai.Client(api_key=api_key)

    try:
        logger.debug(
            "Generating person connections for %s (%d chars findings)",
            person_gramps_id,
            len(findings_text),
        )
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=context)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=CONNECTIONS_SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )

        response_text = extract_text_from_response(response)
        metadata = extract_metadata_from_result(response)
        response_text = sanitize_answer(response_text)
        return response_text, metadata
    except ValueError as e:
        logger.error("Gemini configuration error during connections generation: %s", e)
        abort_with_message(500, "Error communicating with the AI model")
    except Exception as e:
        logger.error("Unexpected error during connections generation: %s", e)
        abort_with_message(500, "Unexpected error generating person connections.")


NUGGET_SYSTEM_PROMPT = """You will receive data about a family tree. Turn it into exactly 30 short, punchy nuggets (1-2 sentences, under 40 words each).

These nuggets should make someone stop scrolling. Think "fun facts" at the bottom of a cereal box, but for genealogy nerds. Be witty. Be specific. Make the reader want to click through to the person's page.

RULES:
- Base nuggets ONLY on the data provided. Do NOT fabricate names, dates, or facts.
- Include the Gramps ID of the most relevant person in [brackets] at the end.
- If a finding doesn't have a clear person ID, use [TREE] instead.
- Number each nugget 1-30.

GOOD NUGGETS:
- "Three Olson sisters all married men from the same Norwegian parish within two years. Either it was love or an extremely efficient travel agent. [I0234]"
- "Johan was a drayman in 1895 and an auto mechanic by 1912 — a career pivot that basically skipped 400 years of technology. [I0567]"
- "19 family members died in 1349. The Black Death didn't send a save-the-date, but it showed up anyway. [TREE]"

BAD NUGGETS (do not write these):
- "The oldest person lived to 94." (boring)
- "Many people were born in Sweden." (vague)
- "The family tree has 5,235 people." (just a number)
- "This shows the resilience of the human spirit." (empty platitude)

FORMAT:
1. Nugget text [GRAMPS_ID]
2. Nugget text [GRAMPS_ID]
...
30. Nugget text [GRAMPS_ID]"""


def generate_nuggets(
    tree: str,
    include_private: bool,
    user_id: str,
    person_subset: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate interesting nuggets using single-shot Gemini call (no tools).

    Pre-gathers tree context by calling analytical tools directly in Python,
    then sends that context to Gemini in one shot. This avoids the agent
    tool-calling loop which takes 2+ minutes and times out Gunicorn.

    Args:
        tree: The tree identifier
        include_private: Whether to include private information
        user_id: The user identifier
        person_subset: If provided, only include findings about these Gramps IDs

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

    clusters_text = find_coincidences_and_clusters(ctx, category="all", max_results=15, person_subset=person_subset)
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

        response_text = extract_text_from_response(response)

        metadata = extract_metadata_from_result(response)
        logger.info("Nuggets generated (%d chars): %s", len(response_text), response_text[:500])

        # Sanitize the response
        response_text = sanitize_answer(response_text)

        return response_text, metadata

    except ValueError as e:
        logger.error("Gemini configuration error during nugget generation: %s", e)
        abort_with_message(500, "Error communicating with the AI model")
    except Exception as e:
        logger.error("Unexpected error during nugget generation: %s", e)
        abort_with_message(500, "Unexpected error generating nuggets.")


BLOG_SYSTEM_PROMPT = """You are a writer for a genealogy blog. Your tone is curious, witty, and conversational — like a friend who just discovered something wild in the archives and can't wait to tell you about it.

You will receive data about people, patterns, and events from a family tree. Write a blog post (500-800 words) that makes the reader say "wait, really?"

TONE:
- Think "favorite history podcast" not "museum placard"
- Lead with whatever is most surprising, weird, or funny
- Be specific and concrete — names, dates, places, odd details
- Historical context is great, but keep it punchy, not textbook-y
- Vary your tone post to post: some can be wry, some dramatic, some playful
- It's fine to be warm sometimes, but don't default to sentimental
- DO NOT end with a generic call-to-action like "share your memories" or "explore further"
- End with a bang — a final surprising detail, a wry observation, or an unanswered question

PERSON LINKS:
- ALWAYS link every person you mention: [Full Name](/person/GRAMPS_ID)
- Every post MUST mention at least 3 specific people by name with links
- Even when writing about events or patterns, name the actual people who lived through them
- Use the Gramps IDs from the data provided — never invent IDs

RULES:
- Base the story ONLY on the data provided. Do NOT fabricate names, dates, or facts.
- You MAY add historical context — this is encouraged. But distinguish inference from record.
- No numbered lists or bullet points in the post body.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
TITLE: [A specific, intriguing title — not generic]

CONTENT:
[The blog post content here. Use markdown links: [Name](/person/GRAMPS_ID)]"""


def generate_blog_post(
    tree: str,
    include_private: bool,
    user_id: str,
    previous_titles: list[str] | None = None,
    previously_featured_ids: set[str] | None = None,
    person_subset: set[str] | None = None,
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
        person_subset: If provided, only include findings about these Gramps IDs

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
    clusters_text = find_coincidences_and_clusters(ctx, category="all", max_results=10, person_subset=person_subset)
    logger.info("Coincidences/clusters gathered (%d chars)", len(clusters_text))

    stats_text = get_tree_statistics(ctx)
    logger.info("Tree statistics gathered (%d chars)", len(stats_text))

    migration_text = analyze_migration_patterns(ctx, person_subset=person_subset)
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
        ids_in_finding = set(
            _re.findall(r'\b(I\d+)\b', finding)
            + _re.findall(r'/person/(I\d+)', finding)
        )
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
    # Match both bare IDs (I0001) and markdown links (/person/I0001)
    gramps_ids = list(set(
        _re.findall(r'\b(I\d+)\b', selected_text)
        + _re.findall(r'/person/(I\d+)', selected_text)
    ))
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

        response_text = extract_text_from_response(response)

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


THIS_DAY_SYSTEM_PROMPT = """You're creating a daily "This Day in Your Family" digest — a quick, fun read that makes users smile when they open the app each morning.

You'll receive events from the family tree that happened on today's calendar date across different years (births, deaths, marriages, etc.). Turn these into 3-5 short, punchy blurbs that feel personal and interesting.

TONE:
- Warm and conversational, like a friend sharing family history
- Highlight whatever is surprising, touching, or just plain interesting about each event
- Vary your approach: some can be wry, some warm, some just factual with a twist
- Keep each blurb to 1-2 sentences max (under 40 words)
- Historical context is welcome, but keep it punchy

PERSON LINKS:
- ALWAYS link every person you mention: [Full Name](/person/GRAMPS_ID)
- Use the exact Gramps IDs from the data provided

RULES:
- Base blurbs ONLY on the data provided
- You MAY add historical context to enrich the story
- No numbered lists, headers, or bold text — just flowing paragraphs
- If events span very different eras, you can note the range: "From 1723 to 1987, this date saw..."

Write 3-5 blurbs, each as its own paragraph."""


def generate_this_day(
    month: int,
    day: int,
    tree: str,
    include_private: bool,
    user_id: str,
) -> tuple[str, dict[str, Any]]:
    """Generate 'This Day in Your Family' digest using single-shot Gemini call.

    Finds all events in the tree matching the given month and day (across all years),
    then sends them to Gemini for a narrative treatment.

    Args:
        month: Month number (1-12)
        day: Day number (1-31)
        tree: The tree identifier
        include_private: Whether to include private information
        user_id: The user identifier

    Returns:
        Tuple of (narrative_text, metadata_dict)
    """
    logger = get_logger()

    config = current_app.config
    model_name = config.get("LLM_MODEL")

    if not model_name:
        raise ValueError("No LLM model specified")

    api_key = os.environ.get("GEMINI_API_KEY") or config.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key configured")

    # Pre-gather events matching this month/day
    deps = AgentDeps(
        tree=tree,
        include_private=include_private,
        max_context_length=config.get("LLM_MAX_CONTEXT_LENGTH", 50000),
        user_id=user_id,
    )

    from ..util import get_db_outside_request
    from gramps.gen.lib.date import gregorian

    db_handle = None
    try:
        db_handle = get_db_outside_request(
            tree=tree,
            view_private=include_private,
            readonly=True,
            user_id=user_id,
        )

        logger.info("Gathering events for month=%d, day=%d", month, day)

        # Pre-build reverse index: event_handle -> [(name, gramps_id), ...]
        event_to_people = {}
        for ph in db_handle.iter_person_handles():
            p = db_handle.get_person_from_handle(ph)
            if not include_private and p.private:
                continue
            for eref in p.get_event_ref_list():
                event_to_people.setdefault(eref.ref, []).append(
                    (p.get_primary_name().get_name(), p.get_gramps_id())
                )

        event_to_family_people = {}
        for fh in db_handle.iter_family_handles():
            fam = db_handle.get_family_from_handle(fh)
            if not include_private and fam.private:
                continue
            for eref in fam.get_event_ref_list():
                people = []
                for spouse_handle in [fam.get_father_handle(), fam.get_mother_handle()]:
                    if spouse_handle:
                        spouse = db_handle.get_person_from_handle(spouse_handle)
                        if include_private or not spouse.private:
                            people.append(
                                (spouse.get_primary_name().get_name(), spouse.get_gramps_id())
                            )
                if people:
                    event_to_family_people.setdefault(eref.ref, []).extend(people)

        # Collect all events matching this month/day
        matching_events = []

        for event_handle in db_handle.iter_event_handles():
            event = db_handle.get_event_from_handle(event_handle)
            if not include_private and event.private:
                continue

            date = event.get_date_object()
            if not date or not date.is_valid():
                continue

            # Convert to gregorian calendar and check if month and day match
            greg_date = gregorian(date)
            if greg_date.get_month() == month and greg_date.get_day() == day:
                event_type = event.get_type().string
                year = greg_date.get_year()

                # Find people associated with this event
                person_names = []
                person_ids = []

                # Look up people for this event from pre-built indexes
                for name, gid in event_to_people.get(event_handle, []):
                    person_names.append(name)
                    person_ids.append(gid)

                for name, gid in event_to_family_people.get(event_handle, []):
                    person_names.append(name)
                    person_ids.append(gid)

                if person_names:  # Only include events with associated people
                    matching_events.append({
                        'type': event_type,
                        'year': year,
                        'people': list(zip(person_names, person_ids)),
                    })

        logger.info("Found %d events matching %02d-%02d", len(matching_events), month, day)

        if not matching_events:
            # No events on this exact date — find the nearest date with events
            logger.info("No events found for %02d-%02d, searching for nearby dates", month, day)

            # Simple fallback: just return a message
            fallback_text = f"Nothing recorded on {month:02d}-{day:02d} in your family tree. Check back tomorrow for another date!"
            return fallback_text, {}

        # Sort by year
        matching_events.sort(key=lambda x: x['year'])

        # Build context for Gemini
        context_parts = [f"Events that happened on {month:02d}-{day:02d} across different years:\n"]

        for evt in matching_events:
            people_str = ", ".join([f"{name} ({gid})" for name, gid in evt['people']])
            context_parts.append(
                f"- {evt['year']}: {evt['type']} — {people_str}"
            )

        context = "\n".join(context_parts)
        logger.info("Context for Gemini (%d chars): %s", len(context), context[:300])

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=context)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=THIS_DAY_SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )

        response_text = extract_text_from_response(response)

        metadata = extract_metadata_from_result(response)
        logger.info("This Day digest generated (%d chars): %s", len(response_text), response_text[:300])

        # Sanitize the response
        response_text = sanitize_answer(response_text)

        return response_text, metadata

    except ValueError as e:
        logger.error("Gemini configuration error during this-day generation: %s", e)
        abort_with_message(500, "Error communicating with the AI model")
    except Exception as e:
        logger.error("Unexpected error during this-day generation: %s", e)
        abort_with_message(500, "Unexpected error generating this-day digest.")
    finally:
        if db_handle:
            db_handle.close()
