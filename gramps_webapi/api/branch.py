"""Branch computation for per-user personalization.

Computes the set of people on a user's "side of the family" by walking UP
N generations from their home person to find root ancestors, then collecting
all descendants of those ancestors.
"""

import json
import logging
from collections import deque
from typing import Optional, Set

from .resources.filters import apply_filter

logger = logging.getLogger(__name__)


def compute_branch_ids(
    db_handle,
    home_person_gramps_id: str,
    generations_up: int = 5,
    descendant_depth: int = 20,
) -> Set[str]:
    """Compute the set of Gramps IDs in a user's branch.

    Algorithm:
    1. BFS upward from home person for `generations_up` generations
       to find the top-generation ancestors.
    2. For each top-generation ancestor, use Gramps' built-in
       IsLessThanNthGenerationDescendantOf filter to collect all descendants.
    3. Union all results into a single set of Gramps IDs.

    Args:
        db_handle: Open Gramps DB handle (readonly).
        home_person_gramps_id: Gramps ID of the home person (e.g. "I0001").
        generations_up: How many generations to walk up (default 5).
        descendant_depth: Max depth for descendant collection (default 20).

    Returns:
        Set of Gramps IDs for everyone in the branch.
    """
    home_person = db_handle.get_person_from_gramps_id(home_person_gramps_id)
    if not home_person:
        logger.warning("Home person %s not found", home_person_gramps_id)
        return set()

    # Step 1: BFS upward to find top-generation ancestors
    top_ancestors = _walk_up(db_handle, home_person, generations_up)

    if not top_ancestors:
        # If no ancestors found (orphan), treat the home person as the root
        top_ancestors = {home_person.handle}

    logger.info(
        "Branch computation for %s: found %d top ancestors at generation %d",
        home_person_gramps_id,
        len(top_ancestors),
        generations_up,
    )

    # Step 2: For each top ancestor, collect all descendants
    branch_handles: Set[str] = set()
    for ancestor_handle in top_ancestors:
        ancestor = db_handle.get_person_from_handle(ancestor_handle)
        if not ancestor:
            continue

        descendant_handles = _get_descendants(
            db_handle, ancestor.gramps_id, descendant_depth
        )
        branch_handles.update(descendant_handles)

    # Step 3: Convert handles to Gramps IDs
    branch_ids: Set[str] = set()
    for handle in branch_handles:
        person = db_handle.get_person_from_handle(handle)
        if person:
            branch_ids.add(person.gramps_id)

    logger.info(
        "Branch for %s: %d people from %d top ancestors",
        home_person_gramps_id,
        len(branch_ids),
        len(top_ancestors),
    )

    return branch_ids


def _walk_up(db_handle, person, generations: int) -> Set[str]:
    """BFS upward to find ancestors at the Nth generation.

    Returns handles of ancestors at exactly the top generation.
    If a lineage ends before reaching the top generation, the
    last-found ancestor in that line is included.
    """
    # Each entry: (handle, current_generation)
    current_level = {person.handle}
    top_ancestors: Set[str] = set()

    for gen in range(generations):
        next_level: Set[str] = set()
        for handle in current_level:
            p = db_handle.get_person_from_handle(handle)
            if not p:
                top_ancestors.add(handle)
                continue

            parent_families = p.get_parent_family_handle_list()
            if not parent_families:
                # Dead end — this person is a top ancestor for this line
                top_ancestors.add(handle)
                continue

            found_parent = False
            for fam_handle in parent_families:
                family = db_handle.get_family_from_handle(fam_handle)
                if not family:
                    continue
                for parent_handle in [
                    family.get_father_handle(),
                    family.get_mother_handle(),
                ]:
                    if parent_handle:
                        next_level.add(parent_handle)
                        found_parent = True

            if not found_parent:
                top_ancestors.add(handle)

        if not next_level:
            break
        current_level = next_level

    # Whatever is left at the top level are top ancestors
    top_ancestors.update(current_level)
    return top_ancestors


def _get_descendants(
    db_handle, ancestor_gramps_id: str, depth: int
) -> Set[str]:
    """Get all descendants of an ancestor using Gramps filter rules."""
    rules = [
        {
            "name": "IsLessThanNthGenerationDescendantOf",
            "values": [ancestor_gramps_id, str(depth + 1)],
        }
    ]
    filter_rules = json.dumps({"rules": rules})
    args = {"rules": filter_rules}

    try:
        matching_handles = apply_filter(
            db_handle=db_handle,
            args=args,
            namespace="Person",
            handles=None,
        )
        return set(matching_handles)
    except Exception as e:
        logger.error(
            "Error getting descendants of %s: %s", ancestor_gramps_id, e
        )
        return set()
