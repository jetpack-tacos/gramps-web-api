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

import os
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import GoogleSearch, Tool

from .deps import AgentDeps
from .tools import (
    filter_events,
    filter_people,
    get_current_date,
    search_genealogy_database,
)


SYSTEM_PROMPT = """You are an assistant for answering questions about a user's family history.

IMPORTANT GUIDELINES

Use the available tools to retrieve information from the user's genealogy database.

Base your answers ONLY on information returned by the tools. Do NOT make up facts, dates, names, relationships, or any other details.

Think carefully about what the user is asking before choosing which tool and parameters to use.

If the user refers to themselves ("I", "my", "me"), ask for their name in the family tree to look them up.


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

Keep your answers concise and accurate."""


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
    else:
        return f"Unknown tool: {tool_name}"


def run_agent(
    prompt: str,
    deps: AgentDeps,
    model_name: str,
    system_prompt_override: str | None = None,
    history: list[types.Content] | None = None,
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

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[
            types.Tool(
                function_declarations=tool_declarations,
                google_search=GoogleSearch(),
            ),
        ],
        temperature=0.2,
    )

    max_iterations = 10
    for _ in range(max_iterations):
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

        # Add the model's response (with function calls) to history
        contents.append(response.candidates[0].content)

        # Execute each function call and collect results
        function_response_parts = []
        for part in response.candidates[0].content.parts:
            if part.function_call is not None:
                tool_name = part.function_call.name
                tool_args = dict(part.function_call.args) if part.function_call.args else {}

                result = execute_tool_call(tool_name, tool_args, deps)

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    )
                )

        # Add tool results back to the conversation
        contents.append(types.Content(role="user", parts=function_response_parts))

    return response
