#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2025      David Straub
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""Gemini agent for LLM interactions."""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import GoogleSearch

from .deps import AgentDeps
from .tools import (
    add_event_to_person,
    analyze_migration_patterns,
    analyze_naming_patterns,
    filter_events,
    filter_people,
    find_coincidences_and_clusters,
    find_data_quality_issues,
    find_relationship_path,
    get_current_date,
    get_data_quality_issues,
    get_family_details,
    get_occupation_summary,
    get_person_full_details,
    get_tree_statistics,
    search_genealogy_database,
    update_person_field,
)


SYSTEM_PROMPT = """You are a witty, curious family history assistant who genuinely enjoys digging through genealogy records. Think of yourself as the friend who falls down Wikipedia rabbit holes and comes back with amazing stories.

YOUR VOICE:
- Conversational and engaging, never dry or clinical
- Let humor come naturally — a wry aside, an unexpected comparison, a funny observation
- Lead with what's surprising or weird, not with a data dump
- When the data is dramatic, lean into it. When it's mundane, find the angle that makes it fun.
- Don't be sentimental by default. Save the heartstrings for when they're earned.
- Make every response feel like something worth reading, not just an answer.

IMPORTANT GUIDELINES

Use the available tools to retrieve information from the user's genealogy database.

Base your answers ONLY on information returned by the tools. Do NOT make up facts, dates, names, relationships, or any other details.

Think carefully about what the user is asking before choosing which tool and parameters to use.

If the user refers to themselves ("I", "my", "me"), ask for their name in the family tree to look them up.

IMPORTANT: When looking up a specific person by name, ALWAYS use filter_people with given_name and/or surname parameters. Do NOT use search_genealogy_database for name lookups - that tool is for semantic/concept searches only. Use filter_people for precise name matching.


ANALYTICAL DEPTH AND NARRATIVE SYNTHESIS

Your most valuable role is to notice things the user would never think to ask about. Most users don't know what's in their tree, so they can't form good questions.

When asked broad questions like "what's interesting?" or "surprise me", use find_coincidences_and_clusters to discover patterns: geographic clusters, temporal clusters, chain migration, naming traditions, parallel lives.

The bar for "interesting" is high. Don't just report statistical extremes ("oldest person was 94"). Look for coincidences that suggest a story: "Three sisters all married men from the same Norwegian parish within two years" or "This person listed as 'drayman' in 1895 but 'automobile mechanic' in 1912 - one of the first to make the transition."

Use multi-step reasoning: discover a pattern → get full details on the people involved → search the web for historical context → weave it into a narrative.

For migration questions, use analyze_migration_patterns to find movement patterns, then proactively search for historical events that explain WHY people moved: wars, famines, land grants, gold rushes, religious persecution, economic booms.


TOOL CHAINING STRATEGY

For narrative questions about specific people: use get_person_full_details to get comprehensive details, then search_genealogy_database or filter_people to find related people, then Google Search for historical context.

For "tell me about [person]" requests: pull all available data and weave it into a readable narrative, not just a list of facts. Include historical context when dates/places are known.

For discovery questions ("what's interesting", "surprise me"): start with find_coincidences_and_clusters, pick the most compelling findings, then enrich them with get_person_full_details and web search.

For relationship questions: use find_relationship_path to explain the connection in plain language.

For data quality awareness: when presenting information, note when sources are missing or data seems inconsistent. The find_data_quality_issues tool can help identify gaps.


WEB RESEARCH GUIDELINES

When the user asks about historical context, living conditions, occupations, immigration patterns, or "what was life like" questions, use Google Search to find relevant historical information.

Always combine database facts with historical context when both are relevant.

When you find migration patterns in the tree (people moving from one place to another), proactively search for historical events that might explain why: wars, famines, economic booms, immigration laws, land grants, gold rushes, religious persecution, etc. This is one of the most valuable things you can do.

When citing web sources, include them as links at the end of your response in a "Sources:" section.


RELATIONSHIP QUERIES

For questions about relationships like parents, grandparents, siblings, or cousins, follow this workflow:

First, search for the person to get their Gramps ID.

Then use filter_people with the relationship filter AND show_relation_with set to that Gramps ID.

Results will have labels like [father], [grandfather], [sibling] that help you identify the correct people.

Available relationship filters: ancestor_of (parents=1, grandparents=2), descendant_of (children=1, grandchildren=2), degrees_of_separation_from (siblings=2, uncles=3, cousins=4), has_common_ancestor_with

Without show_relation_with, you cannot distinguish between generations or relationship types.


FORMATTING RULES (CRITICAL)

Tool results contain links like [Name](/person/I0044). Copy these EXACTLY as they appear. Never modify the URLs. Do NOT strip links, always keep links if possible.

Never change /person/I0044 to # or remove it. Keep the exact path.

ABSOLUTELY FORBIDDEN: Do not use numbered lists (1. 2. 3.), bullet points (- or *), bold (**text**), italic (*text*), headers (#), code blocks (```), or blockquotes (>).

If you use ANY of these forbidden formats, you are making a mistake.

To list multiple items, separate them with "and" or line breaks. Never use numbers or bullets.

Keep it simple: plain sentences with Markdown links only.


OTHER GUIDELINES

If you don't have enough information after using the tools, say "I don't know" or "I couldn't find that information."

Keep your answers concise and accurate, but don't sacrifice narrative quality for brevity. A good story is worth a few extra sentences."""


class _ToolContext:
    """Minimal context object that mimics RunContext[AgentDeps] for tool calls.

    The existing tools in tools.py expect ctx.deps to hold an AgentDeps instance.
    This wrapper provides that interface without depending on pydantic_ai.
    """

    def __init__(self, deps: AgentDeps):
        self.deps = deps


def _make_tool_wrappers(deps: AgentDeps) -> list[types.FunctionDeclaration]:
    """Create Gemini function declarations that wrap the existing tools.

    Returns a list of FunctionDeclaration objects for Gemini's function calling.
    The actual tool execution is handled separately in execute_tool_call().
    """
    return [
        types.FunctionDeclaration(
            name="get_current_date",
            description="Returns today's date in ISO format (YYYY-MM-DD).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="search_genealogy_database",
            description=(
                "Searches the user's family tree using semantic similarity. "
                "Returns formatted genealogical data including people, families, "
                "events, places, sources, citations, repositories, notes, and "
                "media matching the query."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Search query for genealogical information",
                    ),
                    "max_results": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum results to return (default: 20, max: 50)",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="filter_people",
            description=(
                "Filters people in the family tree based on criteria. "
                "IMPORTANT: When filtering by relationships (ancestor_of, descendant_of, "
                "degrees_of_separation_from), ALWAYS set show_relation_with to the same "
                "Gramps ID to get relationship labels in results. "
                "Examples: Find parents with labels: ancestor_of='I0044', ancestor_generations=1, "
                "show_relation_with='I0044'. Find siblings: degrees_of_separation_from='I0044', "
                "degrees_of_separation=2, show_relation_with='I0044'."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "given_name": types.Schema(
                        type=types.Type.STRING,
                        description="Given/first name to search for (partial match)",
                    ),
                    "surname": types.Schema(
                        type=types.Type.STRING,
                        description="Surname/last name to search for (partial match)",
                    ),
                    "birth_year_before": types.Schema(
                        type=types.Type.STRING,
                        description="Year before which people were born (e.g., '1900')",
                    ),
                    "birth_year_after": types.Schema(
                        type=types.Type.STRING,
                        description="Year after which people were born (e.g., '1850')",
                    ),
                    "birth_place": types.Schema(
                        type=types.Type.STRING,
                        description="Place name where person was born (partial match)",
                    ),
                    "death_year_before": types.Schema(
                        type=types.Type.STRING,
                        description="Year before which people died (e.g., '1950')",
                    ),
                    "death_year_after": types.Schema(
                        type=types.Type.STRING,
                        description="Year after which people died (e.g., '1800')",
                    ),
                    "death_place": types.Schema(
                        type=types.Type.STRING,
                        description="Place name where person died (partial match)",
                    ),
                    "ancestor_of": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of person to find ancestors of (e.g., 'I0044')",
                    ),
                    "ancestor_generations": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum generations to search for ancestors (default: 10)",
                    ),
                    "descendant_of": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of person to find descendants of (e.g., 'I0044')",
                    ),
                    "descendant_generations": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum generations to search for descendants (default: 10)",
                    ),
                    "is_male": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="Filter to only males",
                    ),
                    "is_female": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="Filter to only females",
                    ),
                    "probably_alive_on_date": types.Schema(
                        type=types.Type.STRING,
                        description="Date to check if person was likely alive (YYYY-MM-DD)",
                    ),
                    "has_common_ancestor_with": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID to find people sharing an ancestor",
                    ),
                    "degrees_of_separation_from": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Gramps ID of person to find relatives connected to. "
                            "Each parent-child or spousal connection counts as 1. "
                            "Examples: sibling=2, grandparent=2, uncle=3, first cousin=4"
                        ),
                    ),
                    "degrees_of_separation": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum relationship path length (default: 2)",
                    ),
                    "combine_filters": types.Schema(
                        type=types.Type.STRING,
                        description="How to combine multiple filters: 'and' (default) or 'or'",
                    ),
                    "max_results": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum results to return (default: 50, max: 100)",
                    ),
                    "show_relation_with": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Gramps ID of person to show relationships relative to. "
                            "ALWAYS use this with relationship filters to get labels like [father], [sibling]."
                        ),
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="filter_events",
            description=(
                "Filter events in the genealogy database. Events are occurrences in "
                "people's lives (births, deaths, marriages, etc.) or general historical events."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "event_type": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Type of event (e.g., 'Birth', 'Death', 'Marriage', "
                            "'Baptism', 'Census', 'Emigration', 'Burial', 'Occupation', 'Residence')"
                        ),
                    ),
                    "date_before": types.Schema(
                        type=types.Type.STRING,
                        description="Latest year to include (e.g., '1900'). Use only the year.",
                    ),
                    "date_after": types.Schema(
                        type=types.Type.STRING,
                        description="Earliest year to include (e.g., '1850'). Use only the year.",
                    ),
                    "place": types.Schema(
                        type=types.Type.STRING,
                        description="Location name to search for (e.g., 'Boston', 'Massachusetts')",
                    ),
                    "description_contains": types.Schema(
                        type=types.Type.STRING,
                        description="Text that should appear in the event description",
                    ),
                    "participant_id": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of a person who participated in the event (e.g., 'I0001')",
                    ),
                    "participant_role": types.Schema(
                        type=types.Type.STRING,
                        description="Role of the participant (e.g., 'Primary', 'Family')",
                    ),
                    "max_results": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of results to return (1-100, default 50)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_data_quality_issues",
            description=(
                "Get data quality issue counts or affected records. Use issue_type='all' "
                "for summary counts, or a specific issue type for records."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "issue_type": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Issue type to query: 'all', 'unknown_gender', "
                            "'missing_birth', 'missing_death', 'missing_parents', "
                            "'disconnected', 'incomplete_names', 'incomplete_events', "
                            "'no_marriage_records', 'no_sources'"
                        ),
                    ),
                    "max_results": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum records to return for specific issue types (1-50, default 20)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="update_person_field",
            description=(
                "Update a supported person field. Currently supports gender "
                "(0=female, 1=male, 2=unknown)."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "gramps_id": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of the person to update (e.g., 'I0001')",
                    ),
                    "field": types.Schema(
                        type=types.Type.STRING,
                        description="Field to update. Currently: 'gender'",
                    ),
                    "value": types.Schema(
                        type=types.Type.INTEGER,
                        description="New field value (for gender: 0, 1, or 2)",
                    ),
                    "reason": types.Schema(
                        type=types.Type.STRING,
                        description="Optional rationale stored in revision message",
                    ),
                },
                required=["gramps_id", "field", "value"],
            ),
        ),
        types.FunctionDeclaration(
            name="add_event_to_person",
            description=(
                "Create and link a birth/death event for a person with duplicate prevention."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "gramps_id": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of the person (e.g., 'I0044')",
                    ),
                    "event_type": types.Schema(
                        type=types.Type.STRING,
                        description="Event type to add: 'Birth' or 'Death'",
                    ),
                    "year": types.Schema(
                        type=types.Type.INTEGER,
                        description="Event year (positive integer)",
                    ),
                    "month": types.Schema(
                        type=types.Type.INTEGER,
                        description="Event month (0-12, optional)",
                    ),
                    "day": types.Schema(
                        type=types.Type.INTEGER,
                        description="Event day (0-31, optional)",
                    ),
                    "place_name": types.Schema(
                        type=types.Type.STRING,
                        description="Optional free-text place name",
                    ),
                    "reason": types.Schema(
                        type=types.Type.STRING,
                        description="Optional rationale stored in revision message",
                    ),
                },
                required=["gramps_id", "event_type", "year"],
            ),
        ),
        # Phase 6 - Tier 1: Deep Record Access
        types.FunctionDeclaration(
            name="get_person_full_details",
            description=(
                "Get complete record for one person including all events, notes, sources, "
                "media refs, family links, and attributes. Use this when you need comprehensive "
                "details about a specific person beyond basic biographical info."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "gramps_id": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of the person (e.g., 'I0001')",
                    ),
                    "handle": types.Schema(
                        type=types.Type.STRING,
                        description="Internal handle of the person (use gramps_id instead if you have it)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_family_details",
            description=(
                "Get full family record including parents, children, marriage/divorce events, "
                "and family notes."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "family_handle": types.Schema(
                        type=types.Type.STRING,
                        description="Internal handle of the family",
                    ),
                    "gramps_id": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of either spouse (will find their family)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="find_relationship_path",
            description=(
                "Calculate exact relationship between two people and return the connecting path. "
                "Example: 'A is the grandfather of B' with distance information."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "person1_id": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of first person",
                    ),
                    "person2_id": types.Schema(
                        type=types.Type.STRING,
                        description="Gramps ID of second person",
                    ),
                },
                required=["person1_id", "person2_id"],
            ),
        ),
        # Phase 6 - Tier 2: Whole-Tree Analytics
        types.FunctionDeclaration(
            name="get_tree_statistics",
            description=(
                "Get aggregate statistics about the entire family tree including total counts, "
                "date ranges, top surnames, top places, average family size, average lifespan, "
                "geographic distribution, and event type distribution."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="find_coincidences_and_clusters",
            description=(
                "The most important analytical tool. Find narratively interesting patterns and "
                "coincidences in the family tree. Looks for geographic clusters, temporal clusters, "
                "chain migration, name reuse (necronyms), occupation shifts, parallel lives, "
                "disappearances, and statistical outliers. Use this for 'surprise me' or "
                "'what's interesting' queries."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Type of coincidence to search for: 'all' (default), "
                            "'geographic_clusters', 'temporal_clusters', 'chain_migration', "
                            "'name_reuse', 'occupation_shifts', 'parallel_lives', "
                            "'disappearances', 'statistical_outliers'"
                        ),
                    ),
                    "max_results": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum findings to return per category (default: 10, max: 20)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="analyze_migration_patterns",
            description=(
                "Extract all location changes across people and generations. Returns a timeline "
                "of geographic movements grouped by family line. Highlights chain migration when "
                "multiple families moved to/from the same place in the same period."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "surname": types.Schema(
                        type=types.Type.STRING,
                        description="Filter to one family line (optional)",
                    ),
                    "start_year": types.Schema(
                        type=types.Type.INTEGER,
                        description="Earliest year to include (optional)",
                    ),
                    "end_year": types.Schema(
                        type=types.Type.INTEGER,
                        description="Latest year to include (optional)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="find_data_quality_issues",
            description=(
                "Find people missing key data or with suspicious records. Useful for identifying "
                "gaps in the family tree."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "issue_type": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Type of data quality issue: 'missing_birth', 'missing_death', "
                            "'missing_parents', 'no_sources', 'no_death_for_old', "
                            "'impossible_dates', 'potential_duplicates'"
                        ),
                    ),
                    "max_results": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum results to return (default: 20, max: 50)",
                    ),
                },
                required=["issue_type"],
            ),
        ),
        # Phase 6 - Tier 3: Cultural Patterns
        types.FunctionDeclaration(
            name="analyze_naming_patterns",
            description=(
                "Find naming traditions: children named after grandparents, recurring given names "
                "across generations, necronyms (reusing names of deceased children)."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "surname": types.Schema(
                        type=types.Type.STRING,
                        description="Filter to one family line (optional)",
                    ),
                    "max_generations": types.Schema(
                        type=types.Type.INTEGER,
                        description="How many generations to analyze (default: 5)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_occupation_summary",
            description=(
                "List all occupations found in events/attributes, grouped by time period and "
                "location. Highlights occupation transitions within same person's life "
                "(e.g., farmer → factory worker) and occupation clusters (many people in same "
                "trade in same place)."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "surname": types.Schema(
                        type=types.Type.STRING,
                        description="Filter to one family line (optional)",
                    ),
                    "start_year": types.Schema(
                        type=types.Type.INTEGER,
                        description="Earliest year to include (optional)",
                    ),
                    "end_year": types.Schema(
                        type=types.Type.INTEGER,
                        description="Latest year to include (optional)",
                    ),
                },
            ),
        ),
    ]


def execute_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    deps: AgentDeps,
) -> str:
    """Execute a tool call by dispatching to the appropriate tool function.

    Args:
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool
        deps: Agent dependencies (tree, privacy settings, etc.)

    Returns:
        Tool result as a string
    """
    ctx = _ToolContext(deps)

    if tool_name == "get_current_date":
        return get_current_date(ctx)
    elif tool_name == "search_genealogy_database":
        return search_genealogy_database(ctx, **tool_args)
    elif tool_name == "filter_people":
        return filter_people(ctx, **tool_args)
    elif tool_name == "filter_events":
        return filter_events(ctx, **tool_args)
    elif tool_name == "get_data_quality_issues":
        return get_data_quality_issues(ctx, **tool_args)
    elif tool_name == "update_person_field":
        return update_person_field(ctx, **tool_args)
    elif tool_name == "add_event_to_person":
        return add_event_to_person(ctx, **tool_args)
    # Phase 6 - Tier 1: Deep Record Access
    elif tool_name == "get_person_full_details":
        return get_person_full_details(ctx, **tool_args)
    elif tool_name == "get_family_details":
        return get_family_details(ctx, **tool_args)
    elif tool_name == "find_relationship_path":
        return find_relationship_path(ctx, **tool_args)
    # Phase 6 - Tier 2: Whole-Tree Analytics
    elif tool_name == "get_tree_statistics":
        return get_tree_statistics(ctx)
    elif tool_name == "find_coincidences_and_clusters":
        return find_coincidences_and_clusters(ctx, **tool_args)
    elif tool_name == "analyze_migration_patterns":
        return analyze_migration_patterns(ctx, **tool_args)
    elif tool_name == "find_data_quality_issues":
        return find_data_quality_issues(ctx, **tool_args)
    # Phase 6 - Tier 3: Cultural Patterns
    elif tool_name == "analyze_naming_patterns":
        return analyze_naming_patterns(ctx, **tool_args)
    elif tool_name == "get_occupation_summary":
        return get_occupation_summary(ctx, **tool_args)
    else:
        return f"Unknown tool: {tool_name}"


def run_agent(
    prompt: str,
    deps: AgentDeps,
    model_name: str,
    system_prompt_override: str | None = None,
    history: list[types.Content] | None = None,
    grounding_enabled: bool = True,
    max_iterations: int = 10,
    max_repeated_function_calls: int = 2,
    request_timeout_seconds: int = 45,
    request_retry_attempts: int = 2,
) -> types.GenerateContentResponse:
    """Run the Gemini agent with tool calling loop.

    Sends the prompt to Gemini, handles tool calls by executing them against
    the Gramps database, and returns the final response.

    Args:
        prompt: The user's question/prompt
        deps: Agent dependencies (tree, privacy, context length, user_id)
        model_name: The Gemini model name (e.g., "gemini-3-flash")
        system_prompt_override: Optional override for the system prompt
        history: Optional conversation history as Gemini Content objects
        grounding_enabled: Whether to attach Google Search grounding
        max_iterations: Maximum manual tool-call loop iterations
        max_repeated_function_calls: Max repeated identical tool rounds before forced synthesis
        request_timeout_seconds: Per Gemini request timeout (seconds)
        request_retry_attempts: Per Gemini request retry attempts

    Returns:
        The final GenerateContentResponse from Gemini
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    system_prompt = system_prompt_override or SYSTEM_PROMPT
    tool_declarations = _make_tool_wrappers(deps)

    contents: list[types.Content] = []
    if history:
        contents.extend(history)

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    if grounding_enabled:
        tool_config = types.Tool(
            function_declarations=tool_declarations,
            google_search=GoogleSearch(),
        )
    else:
        tool_config = types.Tool(function_declarations=tool_declarations)

    iteration = 0
    has_function_call = False
    forced_synthesis = False
    previous_signatures: list[str] | None = None
    repeated_signature_rounds = 0
    tool_result_cache: dict[str, str] = {}

    max_iterations = max(1, max_iterations)
    max_repeated_function_calls = max(1, max_repeated_function_calls)
    request_timeout_seconds = max(5, request_timeout_seconds)
    request_retry_attempts = max(1, request_retry_attempts)

    request_http_options = types.HttpOptions(
        timeout=request_timeout_seconds,
        retry_options=types.HttpRetryOptions(attempts=request_retry_attempts),
    )
    # We manage tool loops ourselves; disable SDK auto function-calling recursion.
    automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)

    for iteration in range(max_iterations):
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[tool_config],
            temperature=0.2,
            automatic_function_calling=automatic_function_calling,
            http_options=request_http_options,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        # Check if the model wants to call tools
        if not response.candidates or not response.candidates[0].content.parts:
            break

        has_function_call = any(
            part.function_call is not None
            for part in response.candidates[0].content.parts
        )

        if not has_function_call:
            break

        signatures: list[str] = []
        for part in response.candidates[0].content.parts:
            if part.function_call is not None:
                tool_name = part.function_call.name
                tool_args = (
                    dict(part.function_call.args) if part.function_call.args else {}
                )
                signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"
                signatures.append(signature)

        if previous_signatures is not None and signatures == previous_signatures:
            repeated_signature_rounds += 1
        else:
            repeated_signature_rounds = 0
        previous_signatures = signatures

        if repeated_signature_rounds >= max_repeated_function_calls:
            forced_synthesis = True
            break

        # Add the model's response (with function calls) to history
        contents.append(response.candidates[0].content)

        # Execute each function call and collect results
        function_response_parts = []
        for part in response.candidates[0].content.parts:
            if part.function_call is not None:
                tool_name = part.function_call.name
                tool_args = dict(part.function_call.args) if part.function_call.args else {}
                signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"

                if signature in tool_result_cache:
                    result = tool_result_cache[signature]
                else:
                    result = execute_tool_call(tool_name, tool_args, deps)
                    tool_result_cache[signature] = result

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    )
                )

        # Add tool results back to the conversation
        contents.append(types.Content(role="user", parts=function_response_parts))

    # If we hit repeated tool calls or max iterations, force a final text response.
    if forced_synthesis or (iteration == max_iterations - 1 and has_function_call):
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text="Please provide your answer now based on the information you've gathered. Synthesize what you've learned into a helpful response."
                )],
            )
        )
        # Disable tools for this final synthesis step to prevent another tool-call loop.
        final_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            automatic_function_calling=automatic_function_calling,
            http_options=request_http_options,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=final_config,
        )

    return response
