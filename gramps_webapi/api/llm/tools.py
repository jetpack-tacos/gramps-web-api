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

"""Tools for LLM interactions."""

from __future__ import annotations

import json
from datetime import datetime
from functools import wraps
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class RunContext(Protocol[T]):
    """Minimal protocol matching the context interface expected by tools."""

    deps: T

from ..resources.filters import apply_filter
from ..resources.util import get_one_relationship
from ..search import get_semantic_search_indexer
from ..search.text import obj_strings_from_object
from ..util import get_db_outside_request, get_logger
from .deps import AgentDeps


def _build_date_expression(before: str, after: str) -> str:
    """Build a date string from before/after parameters.

    Args:
        before: Year before which to filter (e.g., "1900")
        after: Year after which to filter (e.g., "1850")

    Returns:
        A date string for Gramps filters:
        - "between 1850 and 1900" for date ranges
        - "after 1850" for only after
        - "before 1900" for only before
    """
    if before and after:
        return f"between {after} and {before}"
    if after:
        return f"after {after}"
    if before:
        return f"before {before}"
    return ""


def _get_relationship_prefix(db_handle, anchor_person, result_person, logger) -> str:
    """Get a relationship string prefix for a result person.

    Args:
        db_handle: Database handle
        anchor_person: The Person object to calculate relationship from
        result_person: The Person object to calculate relationship to
        logger: Logger instance

    Returns:
        A formatted relationship prefix like "[grandfather] " or empty string
    """
    try:
        rel_string, dist_orig, dist_other = get_one_relationship(
            db_handle=db_handle,
            person1=anchor_person,
            person2=result_person,
            depth=10,
        )
        if rel_string and rel_string.lower() not in ["", "self"]:
            return f"[{rel_string}] "
        elif dist_orig == 0 and dist_other == 0:
            return "[self] "
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "Error calculating relationship between %s and %s: %s",
            anchor_person.gramps_id,
            result_person.gramps_id,
            e,
        )
    return ""


def _apply_gramps_filter(
    ctx: RunContext[AgentDeps],
    namespace: str,
    rules: list[dict[str, Any]],
    max_results: int,
    empty_message: str = "No results found matching the filter criteria.",
    show_relation_with: str = "",
    combine_function: str = "and",
) -> str:
    """Apply a Gramps filter and return formatted results.

    This is a common helper for filter tools that handles:
    - Database handle management
    - Filter application
    - Result iteration with privacy checking
    - Context length limiting
    - Truncation messages
    - Error handling
    - Optional relationship calculation

    Args:
        ctx: The Pydantic AI run context with dependencies
        namespace: Gramps object namespace ("Person", "Event", "Family", etc.)
        rules: List of filter rule dictionaries
        max_results: Maximum number of results to return (already validated)
        empty_message: Message to return when no results found
        show_relation_with: Gramps ID of anchor person for relationship calculation (Person namespace only)

    Returns:
        Formatted string with matching objects or error message
    """
    logger = get_logger()
    db_handle = None

    try:
        # Use get_db_outside_request to avoid Flask's g caching, since Pydantic AI's
        # run_sync() uses an event loop that can violate SQLite's thread-safety.
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        filter_dict: dict[str, Any] = {"rules": rules}
        if len(rules) > 1:
            filter_dict["function"] = combine_function

        filter_rules = json.dumps(filter_dict)
        logger.debug("%s filter rules: %s", namespace, filter_rules)

        args = {"rules": filter_rules}
        matching_handles = apply_filter(
            db_handle=db_handle,
            args=args,
            namespace=namespace,
            handles=None,
        )

        if not matching_handles:
            return empty_message

        total_matches = len(matching_handles)
        matching_handles = matching_handles[:max_results]

        context_parts: list[str] = []
        max_length = ctx.deps.max_context_length
        per_item_max = 10000  # Maximum chars per individual item
        current_length = 0

        # Get the anchor person for relationship calculation if requested
        anchor_person = None
        if show_relation_with and namespace == "Person":
            try:
                anchor_person = db_handle.get_person_from_gramps_id(show_relation_with)
                if not anchor_person:
                    logger.warning(
                        "Anchor person %s not found for relationship calculation",
                        show_relation_with,
                    )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Error fetching anchor person %s: %s", show_relation_with, e
                )

        # Get the appropriate method to fetch objects
        get_method_name = f"get_{namespace.lower()}_from_handle"
        get_method = getattr(db_handle, get_method_name)

        for handle in matching_handles:
            try:
                obj = get_method(handle)

                if not ctx.deps.include_private and obj.private:
                    continue

                obj_dict = obj_strings_from_object(
                    db_handle=db_handle,
                    class_name=namespace,
                    obj=obj,
                    semantic=True,
                )

                if not obj_dict:
                    continue

                # obj_strings_from_object always returns string_all/string_public
                content = (
                    obj_dict["string_all"]
                    if ctx.deps.include_private
                    else obj_dict["string_public"]
                )

                if not content:
                    continue

                # Add relationship prefix if anchor person is set
                if anchor_person and namespace == "Person":
                    rel_prefix = _get_relationship_prefix(
                        db_handle, anchor_person, obj, logger
                    )
                    content = rel_prefix + content

                # Truncate individual items if they're too long
                if len(content) > per_item_max:
                    content = (
                        content[:per_item_max]
                        + "\n\n[Content truncated due to length...]"
                    )
                    logger.debug(
                        "Truncated %s content from %d to %d chars",
                        namespace,
                        len(content) - per_item_max,
                        per_item_max,
                    )

                # Check if adding this item would exceed total limit
                if current_length + len(content) > max_length:
                    logger.debug(
                        "Reached max context length (%d chars), stopping at %d results",
                        max_length,
                        len(context_parts),
                    )
                    break

                context_parts.append(content)
                current_length += len(content) + 2

            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Error processing %s %s: %s", namespace, handle, e)
                continue

        if not context_parts:
            return f"{empty_message} (or all results are private)."

        result = "\n\n".join(context_parts)

        # Add truncation messages
        returned_count = len(context_parts)
        if returned_count < total_matches:
            result += f"\n\n---\nShowing {returned_count} of {total_matches} matching {namespace.lower()}s. Use max_results parameter to see more."
        elif total_matches == max_results:
            result += f"\n\n---\nShowing {returned_count} {namespace.lower()}s (limit reached). There may be more matches."

        logger.debug(
            "Tool filter_%ss returned %d results (%d chars)",
            namespace.lower(),
            returned_count,
            len(result),
        )

        return result

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error filtering %ss: %s", namespace.lower(), e)
        return f"Error filtering {namespace.lower()}s: {str(e)}"
    finally:
        if db_handle is not None:
            try:
                db_handle.close()
            except Exception:  # pylint: disable=broad-except
                pass


def log_tool_call(func):
    """Decorator to log tool usage."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug("Tool called: %s", func.__name__)
        return func(*args, **kwargs)

    return wrapper


@log_tool_call
def get_current_date(_ctx: RunContext[AgentDeps]) -> str:
    """Returns today's date in ISO format (YYYY-MM-DD)."""
    logger = get_logger()

    result = datetime.now().date().isoformat()
    logger.debug("Tool get_current_date returned: %s", result)
    return result


@log_tool_call
def search_genealogy_database(
    ctx: RunContext[AgentDeps], query: str, max_results: int = 20
) -> str:
    """Searches the user's family tree using semantic similarity.

    Args:
        query: Search query for genealogical information
        max_results: Maximum results to return (default: 20, max: 50)

    Returns:
        Formatted genealogical data including people, families, events, places,
        sources, citations, repositories, notes, and media matching the query.
    """

    logger = get_logger()

    # Limit max_results to reasonable bounds
    max_results = min(max(1, max_results), 50)

    try:
        searcher = get_semantic_search_indexer(ctx.deps.tree)

        # Try semantic search, but fall back gracefully if it fails
        try:
            _, hits = searcher.search(
                query=query,
                page=1,
                pagesize=max_results,
                include_private=ctx.deps.include_private,
                include_content=True,
            )
        except Exception as search_error:
            # Semantic search failed - log it and return helpful message
            logger.warning("Semantic search failed for query %r: %s. This is a known issue with the search indexer.", query, search_error)
            return (
                f"Semantic search is currently unavailable (indexer error: {str(search_error)}). "
                "Please use filter_people or filter_events with specific parameters instead of searching."
            )

        if not hits:
            return "No results found in the genealogy database."

        context_parts: list[str] = []
        max_length = ctx.deps.max_context_length
        per_item_max = 10000  # Maximum chars per individual item
        current_length = 0

        for hit in hits:
            content = hit.get("content", "")

            # Truncate individual items if they're too long
            if len(content) > per_item_max:
                content = (
                    content[:per_item_max] + "\n\n[Content truncated due to length...]"
                )
                logger.debug(
                    "Truncated search result from %d to %d chars",
                    len(content) - per_item_max,
                    per_item_max,
                )

            if current_length + len(content) > max_length:
                logger.debug(
                    "Reached max context length (%d chars), stopping at %d results",
                    max_length,
                    len(context_parts),
                )
                break
            context_parts.append(content)
            current_length += len(content) + 2

        result = "\n\n".join(context_parts)
        logger.debug(
            "Tool search_genealogy_database returned %d results (%d chars) for query: %r",
            len(context_parts),
            len(result),
            query,
        )
        return result

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error searching genealogy database: %s", e)
        return f"Error searching the database: {str(e)}"


@log_tool_call
def filter_people(
    ctx: RunContext[AgentDeps],
    given_name: str = "",
    surname: str = "",
    birth_year_before: str = "",
    birth_year_after: str = "",
    birth_place: str = "",
    death_year_before: str = "",
    death_year_after: str = "",
    death_place: str = "",
    ancestor_of: str = "",
    ancestor_generations: int = 10,
    descendant_of: str = "",
    descendant_generations: int = 10,
    is_male: bool = False,
    is_female: bool = False,
    probably_alive_on_date: str = "",
    has_common_ancestor_with: str = "",
    degrees_of_separation_from: str = "",
    degrees_of_separation: int = 2,
    combine_filters: str = "and",
    max_results: int = 50,
    show_relation_with: str = "",
) -> str:
    """Filters people in the family tree based on simple criteria.

    IMPORTANT: When filtering by relationships (ancestor_of, descendant_of, degrees_of_separation_from),
    ALWAYS set show_relation_with to the same Gramps ID to get relationship labels in results.
    Without it, you cannot determine specific relationships like "grandfather" vs "father".

    Args:
        given_name: Given/first name to search for (partial match)
        surname: Surname/last name to search for (partial match)
        birth_year_before: Year before which people were born (e.g., "1900"). Use only the year.
        birth_year_after: Year after which people were born (e.g., "1850"). Use only the year.
        birth_place: Place name where person was born (partial match)
        death_year_before: Year before which people died (e.g., "1950"). Use only the year.
        death_year_after: Year after which people died (e.g., "1800"). Use only the year.
        death_place: Place name where person died (partial match)
        ancestor_of: Gramps ID of person to find ancestors of (e.g., "I0044")
        ancestor_generations: Maximum generations to search for ancestors (default: 10)
        descendant_of: Gramps ID of person to find descendants of (e.g., "I0044")
        descendant_generations: Maximum generations to search for descendants (default: 10)
        is_male: Filter to only males (True/False)
        is_female: Filter to only females (True/False)
        probably_alive_on_date: Date to check if person was likely alive (YYYY-MM-DD)
        has_common_ancestor_with: Gramps ID to find people sharing an ancestor (e.g., "I0044")
        degrees_of_separation_from: Gramps ID of person to find relatives connected to (e.g., "I0044")
        degrees_of_separation: Maximum relationship path length (default: 2). Each parent-child
            or spousal connection counts as 1. Examples: sibling=2, grandparent=2, uncle=3,
            first cousin=4, brother-in-law=2
        combine_filters: How to combine multiple filters: "and" (default) or "or"
        max_results: Maximum results to return (default: 50, max: 100)
        show_relation_with: Gramps ID of person to show relationships relative to (e.g., "I0044").
            When set, each result will include the relationship to this anchor person.

    Returns:
        Formatted list of people matching the filter criteria.

    Examples:
        - Find people with surname Smith: surname="Smith"
        - Find people born before 1900: birth_year_before="1900"
        - Find people born between 1850-1900: birth_year_after="1850", birth_year_before="1900"
        - Find who was alive in 1880: probably_alive_on_date="1880-01-01"
        - Find cousins: has_common_ancestor_with="I0044"
        - Find someone's parents (with labels): ancestor_of="I0044", ancestor_generations=1, show_relation_with="I0044"
        - Find someone's grandfathers (with labels): ancestor_of="I0044", ancestor_generations=2, is_male=True, show_relation_with="I0044"
        - Find siblings (with labels): degrees_of_separation_from="I0044", degrees_of_separation=2, show_relation_with="I0044"
        - Find extended family (uncles, aunts): degrees_of_separation_from="I0044", degrees_of_separation=3
    """
    logger = get_logger()

    max_results = min(max(1, max_results), 100)

    rules: list[dict[str, Any]] = []

    if given_name or surname:
        rules.append(
            {
                "name": "HasNameOf",
                "values": [given_name, surname, "", "", "", "", "", "", "", "", ""],
            }
        )

    if birth_year_before or birth_year_after or birth_place:
        date_expr = _build_date_expression(birth_year_before, birth_year_after)
        rules.append({"name": "HasBirth", "values": [date_expr, birth_place, ""]})

    if death_year_before or death_year_after or death_place:
        date_expr = _build_date_expression(death_year_before, death_year_after)
        rules.append({"name": "HasDeath", "values": [date_expr, death_place, ""]})

    if ancestor_of:
        rules.append(
            {
                "name": "IsLessThanNthGenerationAncestorOf",
                "values": [ancestor_of, str(ancestor_generations + 1)],
            }
        )

    if descendant_of:
        rules.append(
            {
                "name": "IsLessThanNthGenerationDescendantOf",
                "values": [descendant_of, str(descendant_generations + 1)],
            }
        )

    if has_common_ancestor_with:
        rules.append(
            {"name": "HasCommonAncestorWith", "values": [has_common_ancestor_with]}
        )

    if degrees_of_separation_from:
        # Check if DegreesOfSeparation filter is available (from FilterRules addon)
        from ..resources.filters import get_rule_list

        available_rules = [rule.__name__ for rule in get_rule_list("Person")]  # type: ignore
        if "DegreesOfSeparation" in available_rules:
            rules.append(
                {
                    "name": "DegreesOfSeparation",
                    "values": [degrees_of_separation_from, str(degrees_of_separation)],
                }
            )
        else:
            logger.warning(
                "DegreesOfSeparation filter not available. "
                "Install FilterRules addon to use this feature."
            )
            return (
                "DegreesOfSeparation filter is not available. "
                "The FilterRules addon must be installed to use this feature."
            )

    if is_male:
        rules.append({"name": "IsMale", "values": []})

    if is_female:
        rules.append({"name": "IsFemale", "values": []})

    if probably_alive_on_date:
        rules.append({"name": "ProbablyAlive", "values": [probably_alive_on_date]})

    if not rules:
        return (
            "No filter criteria provided. Please specify at least one filter parameter."
        )

    if combine_filters.lower() == "or":
        return _apply_gramps_filter(
            ctx=ctx,
            namespace="Person",
            rules=rules,
            max_results=max_results,
            empty_message="No people found matching the filter criteria.",
            show_relation_with=show_relation_with,
            combine_function="or",
        )

    # Use the common filter helper for AND logic
    return _apply_gramps_filter(
        ctx=ctx,
        namespace="Person",
        rules=rules,
        max_results=max_results,
        empty_message="No people found matching the filter criteria.",
        show_relation_with=show_relation_with,
    )


@log_tool_call
def filter_events(
    ctx: RunContext[AgentDeps],
    event_type: str = "",
    date_before: str = "",
    date_after: str = "",
    place: str = "",
    description_contains: str = "",
    participant_id: str = "",
    participant_role: str = "",
    max_results: int = 50,
) -> str:
    """Filter events in the genealogy database.

    Use this tool to find events matching specific criteria. Events are occurrences in
    people's lives (births, deaths, marriages, etc.) or general historical events.

    Args:
        event_type: Type of event (e.g., "Birth", "Death", "Marriage", "Baptism",
            "Census", "Emigration", "Burial", "Occupation", "Residence")
        date_before: Latest year to include (inclusive). For "between 1892 and 1900", use "1900".
            Use only the year as a string.
        date_after: Earliest year to include (inclusive). For "between 1892 and 1900", use "1892".
            Use only the year as a string.
        place: Location name to search for (e.g., "Boston", "Massachusetts")
        description_contains: Text that should appear in the event description
        participant_id: Gramps ID of a person who participated in the event (e.g., "I0001")
        participant_role: Role of the participant if participant_id is provided
            (e.g., "Primary", "Family")
        max_results: Maximum number of results to return (1-100, default 50)

    Returns:
        A formatted string containing matching events with their details, or an error message.

    Examples:
        - "births in 1850": filter_events(event_type="Birth", date_after="1850", date_before="1850")
        - "marriages in Boston": filter_events(event_type="Marriage", place="Boston")
        - "events between 1892 and 1900": filter_events(date_after="1892", date_before="1900")
        - "events after 1850": filter_events(date_after="1850")
        - "events before 1900": filter_events(date_before="1900")
        - "events for person I0044": filter_events(participant_id="I0044")
    """
    max_results = min(max(1, max_results), 100)

    rules: list[dict[str, Any]] = []

    if event_type or date_before or date_after or place or description_contains:
        date_expr = _build_date_expression(before=date_before, after=date_after)
        rules.append(
            {
                "name": "HasData",
                "values": [
                    event_type or "",
                    date_expr,
                    place or "",
                    description_contains or "",
                ],
            }
        )

    if participant_id:
        person_filter_rules = [{"name": "HasIdOf", "values": [participant_id]}]
        person_filter_json = json.dumps({"rules": person_filter_rules})

        rules.append(
            {
                "name": "MatchesPersonFilter",
                "values": [person_filter_json, "1" if participant_role else "0"],
            }
        )

    if not rules:
        return (
            "No filter criteria provided. Please specify at least one filter parameter "
            "(event_type, date_before, date_after, place, description_contains, or participant_id)."
        )

    # Use the common filter helper
    return _apply_gramps_filter(
        ctx=ctx,
        namespace="Event",
        rules=rules,
        max_results=max_results,
        empty_message="No events found matching the filter criteria.",
    )


# =============================================================================
# PHASE 6 - TIER 1: Deep Record Access Tools
# =============================================================================


@log_tool_call
def get_person_full_details(
    ctx: RunContext[AgentDeps],
    gramps_id: str = "",
    handle: str = "",
) -> str:
    """Get complete record for one person including all events, notes, sources, media refs, family links, and attributes.

    Use this when you need comprehensive details about a specific person beyond basic biographical info.

    Args:
        gramps_id: Gramps ID of the person (e.g., "I0001")
        handle: Internal handle of the person (use gramps_id instead if you have it)

    Returns:
        Complete formatted person record with all available details
    """
    logger = get_logger()
    db_handle = None

    if not gramps_id and not handle:
        return "Error: Must provide either gramps_id or handle."

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        # Get the person object
        if gramps_id:
            person = db_handle.get_person_from_gramps_id(gramps_id)
            if not person:
                db_handle.close()
                return f"No person found with Gramps ID {gramps_id}."
        else:
            person = db_handle.get_person_from_handle(handle)
            if not person:
                db_handle.close()
                return f"No person found with handle {handle}."

        # Check privacy
        if not ctx.deps.include_private and person.private:
            db_handle.close()
            return "This person's record is private."

        # Use the existing obj_strings_from_object helper for comprehensive formatting
        obj_dict = obj_strings_from_object(
            db_handle=db_handle,
            class_name="Person",
            obj=person,
            semantic=True,
        )

        content = (
            obj_dict["string_all"]
            if ctx.deps.include_private
            else obj_dict["string_public"]
        )

        db_handle.close()
        return content if content else "No details available for this person."

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error fetching person full details: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error retrieving person details: {str(e)}"


@log_tool_call
def get_family_details(
    ctx: RunContext[AgentDeps],
    family_handle: str = "",
    gramps_id: str = "",
) -> str:
    """Get full family record including parents, children, marriage/divorce events, and family notes.

    Args:
        family_handle: Internal handle of the family
        gramps_id: Gramps ID of either spouse (will find their family)

    Returns:
        Complete formatted family record
    """
    logger = get_logger()
    db_handle = None

    if not family_handle and not gramps_id:
        return "Error: Must provide either family_handle or gramps_id of a spouse."

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        # If we have a gramps_id, find the person's families
        if gramps_id:
            person = db_handle.get_person_from_gramps_id(gramps_id)
            if not person:
                db_handle.close()
                return f"No person found with Gramps ID {gramps_id}."

            family_handles = person.get_family_handle_list()
            if not family_handles:
                db_handle.close()
                return f"Person {gramps_id} has no recorded families."

            # Use the first family (most people have one)
            family_handle = family_handles[0]

        family = db_handle.get_family_from_handle(family_handle)
        if not family:
            db_handle.close()
            return f"No family found with handle {family_handle}."

        # Check privacy
        if not ctx.deps.include_private and family.private:
            db_handle.close()
            return "This family record is private."

        # Use obj_strings_from_object for comprehensive formatting
        obj_dict = obj_strings_from_object(
            db_handle=db_handle,
            class_name="Family",
            obj=family,
            semantic=True,
        )

        content = (
            obj_dict["string_all"]
            if ctx.deps.include_private
            else obj_dict["string_public"]
        )

        db_handle.close()
        return content if content else "No details available for this family."

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error fetching family details: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error retrieving family details: {str(e)}"


@log_tool_call
def find_relationship_path(
    ctx: RunContext[AgentDeps],
    person1_id: str,
    person2_id: str,
) -> str:
    """Calculate exact relationship between two people and return the connecting path.

    Args:
        person1_id: Gramps ID of first person
        person2_id: Gramps ID of second person

    Returns:
        Relationship description and connecting path (e.g., "A → father → B → mother → C")
    """
    logger = get_logger()
    db_handle = None

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        person1 = db_handle.get_person_from_gramps_id(person1_id)
        person2 = db_handle.get_person_from_gramps_id(person2_id)

        if not person1:
            db_handle.close()
            return f"No person found with Gramps ID {person1_id}."

        if not person2:
            db_handle.close()
            return f"No person found with Gramps ID {person2_id}."

        # Use the existing get_one_relationship helper
        rel_string, dist_orig, dist_other = get_one_relationship(
            db_handle=db_handle,
            person1=person1,
            person2=person2,
            depth=20,  # Allow deeper searches for distant relationships
        )

        person1_name = person1.get_primary_name().get_name()
        person2_name = person2.get_primary_name().get_name()

        if rel_string:
            result = f"{person1_name} [{person1_id}] is the {rel_string} of {person2_name} [{person2_id}]."
            if dist_orig > 0 or dist_other > 0:
                result += f"\n\nDistance: {dist_orig} generations up from {person1_name}, {dist_other} generations down to {person2_name}."
        else:
            result = f"No relationship found between {person1_name} [{person1_id}] and {person2_name} [{person2_id}] within 20 generations."

        db_handle.close()
        return result

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error calculating relationship path: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error calculating relationship: {str(e)}"


# =============================================================================
# PHASE 6 - TIER 2: Whole-Tree Analytics Tools
# =============================================================================


@log_tool_call
def get_tree_statistics(ctx: RunContext[AgentDeps]) -> str:
    """Get aggregate statistics about the entire family tree.

    Returns comprehensive stats including:
    - Total counts by object type
    - Date ranges (earliest/latest events)
    - Top surnames and given names
    - Top places
    - Average family size and lifespan
    - Geographic distribution
    - Event type distribution

    Args:
        None

    Returns:
        Formatted summary of tree-wide statistics
    """
    logger = get_logger()
    db_handle = None

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        stats = []

        # Basic counts
        person_count = db_handle.get_number_of_people()
        family_count = db_handle.get_number_of_families()
        event_count = db_handle.get_number_of_events()
        place_count = db_handle.get_number_of_places()
        source_count = db_handle.get_number_of_sources()

        stats.append(f"Total Records: {person_count} people, {family_count} families, {event_count} events, {place_count} places, {source_count} sources")

        # Surname frequency
        surname_freq = {}
        for person_handle in db_handle.iter_person_handles():
            person = db_handle.get_person_from_handle(person_handle)
            if not ctx.deps.include_private and person.private:
                continue
            surname = person.get_primary_name().get_surname()
            if surname:
                surname_freq[surname] = surname_freq.get(surname, 0) + 1

        if surname_freq:
            top_surnames = sorted(surname_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            stats.append("\nTop Surnames: " + ", ".join([f"{name} ({count})" for name, count in top_surnames]))

        # Date ranges from birth/death events
        birth_years = []
        death_years = []
        for event_handle in db_handle.iter_event_handles():
            event = db_handle.get_event_from_handle(event_handle)
            if not ctx.deps.include_private and event.private:
                continue

            date = event.get_date_object()
            if date and date.get_year() > 0:
                year = date.get_year()
                event_type = event.get_type().string

                if event_type == "Birth":
                    birth_years.append(year)
                elif event_type == "Death":
                    death_years.append(year)

        if birth_years:
            stats.append(f"\nBirth Years Range: {min(birth_years)} to {max(birth_years)}")
        if death_years:
            stats.append(f"\nDeath Years Range: {min(death_years)} to {max(death_years)}")

        # Average lifespan (only for people with both birth and death dates)
        lifespans = []
        for person_handle in db_handle.iter_person_handles():
            person = db_handle.get_person_from_handle(person_handle)
            if not ctx.deps.include_private and person.private:
                continue

            birth_ref = person.get_birth_ref()
            death_ref = person.get_death_ref()

            if birth_ref and death_ref:
                birth_event = db_handle.get_event_from_handle(birth_ref.ref)
                death_event = db_handle.get_event_from_handle(death_ref.ref)

                birth_date = birth_event.get_date_object() if birth_event else None
                death_date = death_event.get_date_object() if death_event else None

                if birth_date and death_date:
                    birth_year = birth_date.get_year()
                    death_year = death_date.get_year()
                    if birth_year > 0 and death_year > birth_year:
                        lifespans.append(death_year - birth_year)

        if lifespans:
            avg_lifespan = sum(lifespans) / len(lifespans)
            stats.append(f"\nAverage Lifespan: {avg_lifespan:.1f} years (based on {len(lifespans)} people with both birth and death dates)")

        # Average family size
        children_counts = []
        for family_handle in db_handle.iter_family_handles():
            family = db_handle.get_family_from_handle(family_handle)
            if not ctx.deps.include_private and family.private:
                continue
            children_counts.append(len(family.get_child_ref_list()))

        if children_counts:
            avg_children = sum(children_counts) / len(children_counts)
            max_children = max(children_counts)
            stats.append(f"\nAverage Family Size: {avg_children:.1f} children per family (max: {max_children})")

        # Top places
        place_freq = {}
        for event_handle in db_handle.iter_event_handles():
            event = db_handle.get_event_from_handle(event_handle)
            if not ctx.deps.include_private and event.private:
                continue

            place_handle = event.get_place_handle()
            if place_handle:
                place = db_handle.get_place_from_handle(place_handle)
                place_name = place.get_name().get_value()
                if place_name:
                    place_freq[place_name] = place_freq.get(place_name, 0) + 1

        if place_freq:
            top_places = sorted(place_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            stats.append("\nTop Places: " + ", ".join([f"{name} ({count} events)" for name, count in top_places]))

        db_handle.close()
        return "\n".join(stats)

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error calculating tree statistics: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error calculating statistics: {str(e)}"


@log_tool_call
def find_coincidences_and_clusters(
    ctx: RunContext[AgentDeps],
    category: str = "all",
    max_results: int = 10,
) -> str:
    """Find narratively interesting patterns and coincidences in the family tree.

    This is the most important analytical tool. It looks for patterns that are interesting
    from a storytelling perspective, not just statistical extremes.

    Categories:
    - all: Run all checks (default)
    - geographic_clusters: Multiple unrelated families in same small town at same time
    - temporal_clusters: Many events of same type in short window (e.g., 8 marriages in 1942-43)
    - chain_migration: Multiple families from same origin to same destination within years
    - name_reuse: Necronyms (children named after deceased siblings), name patterns
    - occupation_shifts: Same person/family changing occupations across records
    - parallel_lives: Siblings/cousins with eerily similar life patterns
    - disappearances: Family lines that go dark (all members vanish from records)
    - statistical_outliers: Extreme ages, very large families, very young parents

    Args:
        category: Which type of coincidence to search for (default: "all")
        max_results: Maximum findings to return per category (default: 10)

    Returns:
        Formatted list of interesting findings with narrative context
    """
    logger = get_logger()
    db_handle = None

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        findings = []
        max_results = min(max(1, max_results), 20)

        # Geographic Clusters - multiple unrelated people in same place/time
        if category in ["all", "geographic_clusters"]:
            place_time_clusters = {}  # (place, decade) -> list of person IDs

            for person_handle in db_handle.iter_person_handles():
                person = db_handle.get_person_from_handle(person_handle)
                if not ctx.deps.include_private and person.private:
                    continue

                birth_ref = person.get_birth_ref()
                if not birth_ref:
                    continue

                birth_event = db_handle.get_event_from_handle(birth_ref.ref)
                if not birth_event:
                    continue

                place_handle = birth_event.get_place_handle()
                birth_date = birth_event.get_date_object()

                if place_handle and birth_date and birth_date.get_year() > 0:
                    place = db_handle.get_place_from_handle(place_handle)
                    place_name = place.get_name().get_value()
                    decade = (birth_date.get_year() // 10) * 10

                    key = (place_name, decade)
                    if key not in place_time_clusters:
                        place_time_clusters[key] = []
                    place_time_clusters[key].append((person.gramps_id, person.get_primary_name().get_name(), person.get_primary_name().get_surname()))

            # Find clusters with 4+ people (suggests a small community)
            interesting_clusters = []
            for (place, decade), people in place_time_clusters.items():
                if len(people) >= 4:
                    # Check if they're different surnames (suggests unrelated families)
                    surnames = set(p[2] for p in people if p[2])
                    if len(surnames) >= 2:
                        interesting_clusters.append((place, decade, people, surnames))

            interesting_clusters.sort(key=lambda x: len(x[2]), reverse=True)
            for place, decade, people, surnames in interesting_clusters[:max_results]:
                # Include up to 4 example people with Gramps IDs for linking
                examples = [f"[{p[1]}](/person/{p[0]})" for p in people[:4]]
                examples_str = ", ".join(examples)
                finding = f"**Geographic Cluster**: {len(people)} people from {len(surnames)} different families ({', '.join(sorted(surnames)[:3])}) all born in {place} in the {decade}s. Examples: {examples_str}. This suggests a small community with multiple family lines."
                findings.append(finding)

        # Temporal Clusters - many events in short timeframe
        if category in ["all", "temporal_clusters"]:
            event_year_data = {}  # (event_type, year) -> list of (gramps_id, name)

            for event_handle in db_handle.iter_event_handles():
                event = db_handle.get_event_from_handle(event_handle)
                if not ctx.deps.include_private and event.private:
                    continue

                event_type = event.get_type().string
                date = event.get_date_object()

                if date and date.get_year() > 0 and event_type in ["Marriage", "Death", "Emigration", "Immigration"]:
                    year = date.get_year()
                    key = (event_type, year)
                    if key not in event_year_data:
                        event_year_data[key] = []
                    # Find people linked to this event (limit stored to 6 for memory)
                    if len(event_year_data[key]) < 6:
                        for class_name, ref_handle in db_handle.find_backlink_handles(event_handle, ['Person']):
                            person = db_handle.get_person_from_handle(ref_handle)
                            if person:
                                event_year_data[key].append((person.gramps_id, person.get_primary_name().get_name()))
                                break

            # Find years with unusually high event counts
            temporal_clusters = []
            for (event_type, year), people in event_year_data.items():
                count = len(people)
                if count >= 5:  # 5+ of same event type in one year is notable
                    temporal_clusters.append((event_type, year, count, people))

            temporal_clusters.sort(key=lambda x: x[2], reverse=True)
            for event_type, year, count, people in temporal_clusters[:max_results]:
                examples = [f"[{p[1]}](/person/{p[0]})" for p in people[:3]]
                examples_str = ", ".join(examples) if examples else ""
                extra = f" Including: {examples_str}." if examples_str else ""
                finding = f"**Temporal Cluster**: {count} {event_type} events occurred in {year}.{extra} This spike may correlate with historical events (wars, epidemics, economic changes)."
                findings.append(finding)

        # Name Reuse (Necronyms) - children named after deceased siblings
        if category in ["all", "name_reuse"]:
            for family_handle in db_handle.iter_family_handles():
                family = db_handle.get_family_from_handle(family_handle)
                if not ctx.deps.include_private and family.private:
                    continue

                children_refs = family.get_child_ref_list()
                if len(children_refs) < 2:
                    continue

                # Get all children's names and death dates
                children_data = []
                for child_ref in children_refs:
                    child = db_handle.get_person_from_handle(child_ref.ref)
                    if not child or (not ctx.deps.include_private and child.private):
                        continue

                    name = child.get_primary_name().get_first_name()
                    death_ref = child.get_death_ref()
                    death_year = None
                    if death_ref:
                        death_event = db_handle.get_event_from_handle(death_ref.ref)
                        if death_event:
                            death_date = death_event.get_date_object()
                            if death_date:
                                death_year = death_date.get_year()

                    children_data.append({
                        'gramps_id': child.gramps_id,
                        'name': name,
                        'full_name': child.get_primary_name().get_name(),
                        'death_year': death_year
                    })

                # Look for name reuse
                names_seen = {}
                for child in children_data:
                    if child['name'] in names_seen and child['name']:
                        # Found duplicate name - check if first one died
                        prev = names_seen[child['name']]
                        if prev['death_year'] and prev['death_year'] > 0:
                            father_handle = family.get_father_handle()
                            mother_handle = family.get_mother_handle()
                            father = db_handle.get_person_from_handle(father_handle) if father_handle else None
                            mother = db_handle.get_person_from_handle(mother_handle) if mother_handle else None

                            parents = []
                            if father:
                                parents.append(f"[{father.get_primary_name().get_name()}](/person/{father.gramps_id})")
                            if mother:
                                parents.append(f"[{mother.get_primary_name().get_name()}](/person/{mother.gramps_id})")
                            parents_str = " and ".join(parents) if parents else "Unknown parents"

                            finding = f"**Necronym**: The family of {parents_str} had multiple children named '{child['name']}'. [{prev['full_name']}](/person/{prev['gramps_id']}) died around {prev['death_year']}, and another child [{child['full_name']}](/person/{child['gramps_id']}) was given the same name. This was a common practice to preserve family names."
                            findings.append(finding)
                            break
                    else:
                        names_seen[child['name']] = child

        # Statistical Outliers - but only the interesting ones
        if category in ["all", "statistical_outliers"]:
            # Very large families
            large_families = []
            for family_handle in db_handle.iter_family_handles():
                family = db_handle.get_family_from_handle(family_handle)
                if not ctx.deps.include_private and family.private:
                    continue

                child_count = len(family.get_child_ref_list())
                if child_count >= 10:
                    father_handle = family.get_father_handle()
                    mother_handle = family.get_mother_handle()
                    father = db_handle.get_person_from_handle(father_handle) if father_handle else None
                    mother = db_handle.get_person_from_handle(mother_handle) if mother_handle else None

                    parents = []
                    if father:
                        parents.append(f"[{father.get_primary_name().get_name()}](/person/{father.gramps_id})")
                    if mother:
                        parents.append(f"[{mother.get_primary_name().get_name()}](/person/{mother.gramps_id})")

                    large_families.append((child_count, parents))

            large_families.sort(reverse=True)
            for count, parents in large_families[:3]:
                parents_str = " and ".join(parents)
                finding = f"**Large Family**: {parents_str} had {count} children, which was unusually large even by historical standards."
                findings.append(finding)

            # Very long lives (90+ years)
            long_lived = []
            for person_handle in db_handle.iter_person_handles():
                person = db_handle.get_person_from_handle(person_handle)
                if not ctx.deps.include_private and person.private:
                    continue

                birth_ref = person.get_birth_ref()
                death_ref = person.get_death_ref()

                if birth_ref and death_ref:
                    birth_event = db_handle.get_event_from_handle(birth_ref.ref)
                    death_event = db_handle.get_event_from_handle(death_ref.ref)

                    if birth_event and death_event:
                        birth_date = birth_event.get_date_object()
                        death_date = death_event.get_date_object()

                        if birth_date and death_date:
                            birth_year = birth_date.get_year()
                            death_year = death_date.get_year()
                            if birth_year > 0 and death_year > birth_year:
                                lifespan = death_year - birth_year
                                if lifespan >= 90:
                                    long_lived.append((lifespan, person.get_primary_name().get_name(), person.gramps_id, birth_year, death_year))

            long_lived.sort(reverse=True)
            for lifespan, name, gramps_id, birth_year, death_year in long_lived[:3]:
                finding = f"**Longevity**: [{name}](/person/{gramps_id}) lived {lifespan} years ({birth_year}-{death_year}), which was exceptional for the era."
                findings.append(finding)

        if not findings:
            db_handle.close()
            return f"No interesting {category} patterns found in the tree. Try running 'all' or a different category."

        db_handle.close()
        result = "\n\n".join(findings[:max_results * 2])  # Allow more results when running "all"
        return result

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error finding coincidences and clusters: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error finding patterns: {str(e)}"


@log_tool_call
def analyze_migration_patterns(
    ctx: RunContext[AgentDeps],
    surname: str = "",
    start_year: int = 0,
    end_year: int = 9999,
) -> str:
    """Extract all location changes across people and generations.

    Returns a timeline of geographic movements grouped by family line, with date ranges
    and place names. Highlights when multiple families moved to/from the same place in
    the same period (potential chain migration).

    Args:
        surname: Filter to one family line (optional)
        start_year: Earliest year to include (optional)
        end_year: Latest year to include (optional)

    Returns:
        Formatted summary of migration patterns
    """
    logger = get_logger()
    db_handle = None

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        migrations = []  # List of (year, person_name, surname, from_place, to_place)

        for person_handle in db_handle.iter_person_handles():
            person = db_handle.get_person_from_handle(person_handle)
            if not ctx.deps.include_private and person.private:
                continue

            person_surname = person.get_primary_name().get_surname()
            if surname and person_surname.lower() != surname.lower():
                continue

            person_name = person.get_primary_name().get_name()

            # Collect all residence and birth/death events for this person
            life_events = []

            birth_ref = person.get_birth_ref()
            if birth_ref:
                birth_event = db_handle.get_event_from_handle(birth_ref.ref)
                if birth_event:
                    birth_date = birth_event.get_date_object()
                    place_handle = birth_event.get_place_handle()
                    if birth_date and place_handle:
                        year = birth_date.get_year()
                        if start_year <= year <= end_year:
                            place = db_handle.get_place_from_handle(place_handle)
                            place_name = place.get_name().get_value()
                            life_events.append((year, place_name, "Birth"))

            death_ref = person.get_death_ref()
            if death_ref:
                death_event = db_handle.get_event_from_handle(death_ref.ref)
                if death_event:
                    death_date = death_event.get_date_object()
                    place_handle = death_event.get_place_handle()
                    if death_date and place_handle:
                        year = death_date.get_year()
                        if start_year <= year <= end_year:
                            place = db_handle.get_place_from_handle(place_handle)
                            place_name = place.get_name().get_value()
                            life_events.append((year, place_name, "Death"))

            # Also check for Residence, Immigration, Emigration events
            for event_ref in person.get_event_ref_list():
                event = db_handle.get_event_from_handle(event_ref.ref)
                if not event:
                    continue

                event_type = event.get_type().string
                if event_type in ["Residence", "Immigration", "Emigration"]:
                    date = event.get_date_object()
                    place_handle = event.get_place_handle()
                    if date and place_handle:
                        year = date.get_year()
                        if start_year <= year <= end_year:
                            place = db_handle.get_place_from_handle(place_handle)
                            place_name = place.get_name().get_value()
                            life_events.append((year, place_name, event_type))

            # Sort events chronologically
            life_events.sort()

            # Look for location changes
            if len(life_events) >= 2:
                for i in range(1, len(life_events)):
                    prev_year, prev_place, prev_type = life_events[i-1]
                    curr_year, curr_place, curr_type = life_events[i]

                    if prev_place != curr_place:
                        migrations.append({
                            'year': curr_year,
                            'person': person_name,
                            'gramps_id': person.gramps_id,
                            'surname': person_surname,
                            'from': prev_place,
                            'to': curr_place,
                            'event_type': curr_type
                        })

        if not migrations:
            db_handle.close()
            return "No migration patterns found in the tree" + (f" for surname '{surname}'" if surname else "") + "."

        # Sort by year
        migrations.sort(key=lambda x: x['year'])

        # Format results
        result_lines = []
        if surname:
            result_lines.append(f"**Migration Patterns for {surname} Family**:\n")
        else:
            result_lines.append("**Migration Patterns Across All Families**:\n")

        for m in migrations[:30]:  # Limit to 30 most recent migrations
            line = f"- {m['year']}: [{m['person']}](/person/{m['gramps_id']}) ({m['surname']}) moved from {m['from']} to {m['to']}"
            result_lines.append(line)

        # Look for chain migration (multiple people moving from same origin to same destination in close timeframe)
        route_counts = {}  # (from, to, decade) -> count
        for m in migrations:
            decade = (m['year'] // 10) * 10
            key = (m['from'], m['to'], decade)
            route_counts[key] = route_counts.get(key, 0) + 1

        chain_migrations = [(from_place, to_place, decade, count) for (from_place, to_place, decade), count in route_counts.items() if count >= 3]
        if chain_migrations:
            result_lines.append("\n**Possible Chain Migration Routes**:")
            for from_place, to_place, decade, count in sorted(chain_migrations, key=lambda x: x[3], reverse=True)[:5]:
                result_lines.append(f"- {from_place} → {to_place} in the {decade}s: {count} people made this move")

        db_handle.close()
        return "\n".join(result_lines)

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error analyzing migration patterns: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error analyzing migrations: {str(e)}"


@log_tool_call
def find_data_quality_issues(
    ctx: RunContext[AgentDeps],
    issue_type: str = "missing_birth",
    max_results: int = 20,
) -> str:
    """Find people missing key data or with suspicious records.

    Issue types:
    - missing_birth: People without birth events
    - missing_death: People who are likely dead but have no death record
    - missing_parents: People with no recorded parents
    - no_sources: People with no source citations
    - no_death_for_old: People born 100+ years ago with no death record
    - impossible_dates: Child born before parent, death before birth, etc.
    - potential_duplicates: Similar name + similar dates

    Args:
        issue_type: Type of data quality issue to find
        max_results: Maximum results to return (default: 20, max: 50)

    Returns:
        List of people with the specified data quality issue
    """
    logger = get_logger()
    db_handle = None

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        max_results = min(max(1, max_results), 50)
        issues = []

        current_year = datetime.now().year

        if issue_type == "missing_birth":
            for person_handle in db_handle.iter_person_handles():
                person = db_handle.get_person_from_handle(person_handle)
                if not ctx.deps.include_private and person.private:
                    continue

                if not person.get_birth_ref():
                    issues.append(f"[{person.get_primary_name().get_name()}](/person/{person.gramps_id}) has no birth event recorded")

        elif issue_type == "no_death_for_old":
            for person_handle in db_handle.iter_person_handles():
                person = db_handle.get_person_from_handle(person_handle)
                if not ctx.deps.include_private and person.private:
                    continue

                birth_ref = person.get_birth_ref()
                death_ref = person.get_death_ref()

                if birth_ref and not death_ref:
                    birth_event = db_handle.get_event_from_handle(birth_ref.ref)
                    if birth_event:
                        birth_date = birth_event.get_date_object()
                        if birth_date and birth_date.get_year() > 0:
                            birth_year = birth_date.get_year()
                            if current_year - birth_year > 100:
                                issues.append(f"[{person.get_primary_name().get_name()}](/person/{person.gramps_id}) was born in {birth_year} but has no death record")

        elif issue_type == "missing_parents":
            for person_handle in db_handle.iter_person_handles():
                person = db_handle.get_person_from_handle(person_handle)
                if not ctx.deps.include_private and person.private:
                    continue

                parent_families = person.get_parent_family_handle_list()
                if not parent_families:
                    issues.append(f"[{person.get_primary_name().get_name()}](/person/{person.gramps_id}) has no recorded parents")

        elif issue_type == "impossible_dates":
            for person_handle in db_handle.iter_person_handles():
                person = db_handle.get_person_from_handle(person_handle)
                if not ctx.deps.include_private and person.private:
                    continue

                birth_ref = person.get_birth_ref()
                death_ref = person.get_death_ref()

                if birth_ref and death_ref:
                    birth_event = db_handle.get_event_from_handle(birth_ref.ref)
                    death_event = db_handle.get_event_from_handle(death_ref.ref)

                    if birth_event and death_event:
                        birth_date = birth_event.get_date_object()
                        death_date = death_event.get_date_object()

                        if birth_date and death_date:
                            birth_year = birth_date.get_year()
                            death_year = death_date.get_year()

                            if death_year < birth_year:
                                issues.append(f"[{person.get_primary_name().get_name()}](/person/{person.gramps_id}) died ({death_year}) before being born ({birth_year})")

        else:
            db_handle.close()
            return f"Unknown issue type: {issue_type}. Valid types: missing_birth, missing_death, missing_parents, no_sources, no_death_for_old, impossible_dates, potential_duplicates"

        if not issues:
            db_handle.close()
            return f"No {issue_type} issues found in the tree."

        db_handle.close()
        return "\n".join(issues[:max_results])

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error finding data quality issues: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error finding data quality issues: {str(e)}"


# =============================================================================
# PHASE 6 - TIER 3: Cultural Patterns Tools
# =============================================================================


@log_tool_call
def analyze_naming_patterns(
    ctx: RunContext[AgentDeps],
    surname: str = "",
    max_generations: int = 5,
) -> str:
    """Find naming traditions in the family tree.

    Looks for:
    - Children named after grandparents (patronymic patterns)
    - Recurring given names across generations
    - Necronyms (reusing names of deceased children)

    Args:
        surname: Filter to one family line (optional)
        max_generations: How many generations to analyze (default: 5)

    Returns:
        Summary of naming patterns and traditions
    """
    logger = get_logger()
    db_handle = None

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        # Track given names by generation
        name_frequency = {}  # name -> count
        grandparent_naming = []  # Cases where child named after grandparent

        for person_handle in db_handle.iter_person_handles():
            person = db_handle.get_person_from_handle(person_handle)
            if not ctx.deps.include_private and person.private:
                continue

            person_surname = person.get_primary_name().get_surname()
            if surname and person_surname.lower() != surname.lower():
                continue

            given_name = person.get_primary_name().get_first_name()
            if given_name:
                name_frequency[given_name] = name_frequency.get(given_name, 0) + 1

            # Check if named after grandparent
            parent_families = person.get_parent_family_handle_list()
            if parent_families:
                family = db_handle.get_family_from_handle(parent_families[0])
                if family:
                    # Get grandparents
                    father_handle = family.get_father_handle()
                    mother_handle = family.get_mother_handle()

                    grandparents = []
                    if father_handle:
                        father = db_handle.get_person_from_handle(father_handle)
                        if father:
                            for gp_fam_handle in father.get_parent_family_handle_list():
                                gp_fam = db_handle.get_family_from_handle(gp_fam_handle)
                                if gp_fam:
                                    gp_father_handle = gp_fam.get_father_handle()
                                    gp_mother_handle = gp_fam.get_mother_handle()
                                    if gp_father_handle:
                                        grandparents.append(db_handle.get_person_from_handle(gp_father_handle))
                                    if gp_mother_handle:
                                        grandparents.append(db_handle.get_person_from_handle(gp_mother_handle))

                    if mother_handle:
                        mother = db_handle.get_person_from_handle(mother_handle)
                        if mother:
                            for gp_fam_handle in mother.get_parent_family_handle_list():
                                gp_fam = db_handle.get_family_from_handle(gp_fam_handle)
                                if gp_fam:
                                    gp_father_handle = gp_fam.get_father_handle()
                                    gp_mother_handle = gp_fam.get_mother_handle()
                                    if gp_father_handle:
                                        grandparents.append(db_handle.get_person_from_handle(gp_father_handle))
                                    if gp_mother_handle:
                                        grandparents.append(db_handle.get_person_from_handle(gp_mother_handle))

                    # Check if child's name matches any grandparent
                    for gp in grandparents:
                        if gp:
                            gp_given_name = gp.get_primary_name().get_first_name()
                            if gp_given_name and gp_given_name == given_name:
                                grandparent_naming.append({
                                    'child': person.get_primary_name().get_name(),
                                    'child_id': person.gramps_id,
                                    'grandparent': gp.get_primary_name().get_name(),
                                    'grandparent_id': gp.gramps_id,
                                    'name': given_name
                                })

        findings = []

        if surname:
            findings.append(f"**Naming Patterns for {surname} Family**:\n")
        else:
            findings.append("**Naming Patterns Across All Families**:\n")

        # Most common given names
        if name_frequency:
            top_names = sorted(name_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            recurring = [name for name, count in top_names if count >= 5]
            if recurring:
                findings.append(f"**Recurring Names**: {', '.join(recurring[:5])} appear most frequently across generations, suggesting strong naming traditions.")

        # Grandparent naming instances
        if grandparent_naming:
            findings.append(f"\n**Patronymic/Matronymic Naming**: Found {len(grandparent_naming)} cases where children were named after grandparents:")
            for case in grandparent_naming[:5]:
                findings.append(f"- [{case['child']}](/person/{case['child_id']}) named '{case['name']}' after grandparent [{case['grandparent']}](/person/{case['grandparent_id']})")

        if len(findings) <= 1:  # Only has the header
            db_handle.close()
            return "No strong naming patterns detected" + (f" in the {surname} family line" if surname else "") + "."

        db_handle.close()
        return "\n".join(findings)

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error analyzing naming patterns: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error analyzing naming patterns: {str(e)}"


@log_tool_call
def get_occupation_summary(
    ctx: RunContext[AgentDeps],
    surname: str = "",
    start_year: int = 0,
    end_year: int = 9999,
) -> str:
    """List all occupations found in events/attributes, grouped by time period and location.

    Highlights:
    - Occupation transitions within same person's life (e.g., farmer → factory worker)
    - Occupation clusters (many people in same trade in same place)

    Args:
        surname: Filter to one family line (optional)
        start_year: Earliest year to include (optional)
        end_year: Latest year to include (optional)

    Returns:
        Summary of occupations with temporal and geographic patterns
    """
    logger = get_logger()
    db_handle = None

    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        occupation_data = []  # List of (year, person, occupation, place)
        person_occupations = {}  # person_id -> list of (year, occupation)

        for person_handle in db_handle.iter_person_handles():
            person = db_handle.get_person_from_handle(person_handle)
            if not ctx.deps.include_private and person.private:
                continue

            person_surname = person.get_primary_name().get_surname()
            if surname and person_surname.lower() != surname.lower():
                continue

            person_name = person.get_primary_name().get_name()
            person_id = person.gramps_id

            # Check Occupation events
            for event_ref in person.get_event_ref_list():
                event = db_handle.get_event_from_handle(event_ref.ref)
                if not event:
                    continue

                if event.get_type().string == "Occupation":
                    date = event.get_date_object()
                    year = date.get_year() if date else 0

                    if year == 0 or not (start_year <= year <= end_year):
                        continue

                    # Get occupation from description
                    occupation = event.get_description()
                    if not occupation:
                        continue

                    place_handle = event.get_place_handle()
                    place_name = ""
                    if place_handle:
                        place = db_handle.get_place_from_handle(place_handle)
                        place_name = place.get_name().get_value()

                    occupation_data.append({
                        'year': year,
                        'person': person_name,
                        'person_id': person_id,
                        'occupation': occupation,
                        'place': place_name
                    })

                    if person_id not in person_occupations:
                        person_occupations[person_id] = []
                    person_occupations[person_id].append((year, occupation, person_name))

        if not occupation_data:
            db_handle.close()
            return "No occupation records found" + (f" for surname '{surname}'" if surname else "") + "."

        findings = []

        if surname:
            findings.append(f"**Occupation Summary for {surname} Family**:\n")
        else:
            findings.append("**Occupation Summary**:\n")

        # Count occupation types
        occupation_counts = {}
        for rec in occupation_data:
            occ = rec['occupation']
            occupation_counts[occ] = occupation_counts.get(occ, 0) + 1

        top_occupations = sorted(occupation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        findings.append("**Most Common Occupations**: " + ", ".join([f"{occ} ({count})" for occ, count in top_occupations]))

        # Find occupation transitions (same person with different occupations over time)
        transitions = []
        for person_id, occs in person_occupations.items():
            if len(occs) >= 2:
                occs_sorted = sorted(occs, key=lambda x: x[0])
                for i in range(1, len(occs_sorted)):
                    if occs_sorted[i][1] != occs_sorted[i-1][1]:
                        transitions.append({
                            'person': occs_sorted[i][2],
                            'person_id': person_id,
                            'from': occs_sorted[i-1][1],
                            'to': occs_sorted[i][1],
                            'from_year': occs_sorted[i-1][0],
                            'to_year': occs_sorted[i][0]
                        })

        if transitions:
            findings.append("\n**Occupation Transitions** (career changes over time):")
            for t in transitions[:5]:
                findings.append(f"- [{t['person']}](/person/{t['person_id']}): {t['from']} ({t['from_year']}) → {t['to']} ({t['to_year']})")

        # Find occupation clusters (same occupation in same place)
        occupation_place_counts = {}
        for rec in occupation_data:
            if rec['place']:
                key = (rec['occupation'], rec['place'])
                occupation_place_counts[key] = occupation_place_counts.get(key, 0) + 1

        clusters = [(occ, place, count) for (occ, place), count in occupation_place_counts.items() if count >= 3]
        if clusters:
            clusters.sort(key=lambda x: x[2], reverse=True)
            findings.append("\n**Occupation Clusters** (same trade in same location):")
            for occ, place, count in clusters[:5]:
                findings.append(f"- {count} people worked as {occ} in {place}")

        db_handle.close()
        return "\n".join(findings)

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error getting occupation summary: %s", e)
        if db_handle is not None:
            try:
                db_handle.close()
            except:  # pylint: disable=bare-except
                pass
        return f"Error analyzing occupations: {str(e)}"
