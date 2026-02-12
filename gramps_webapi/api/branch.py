"""Branch computation and user clustering for per-user personalization.

Computes the set of people on a user's "side of the family" by walking UP
N generations from their home person to find root ancestors, then collecting
all descendants of those ancestors.

Also handles clustering: users with highly overlapping branches (>= 85%
Jaccard similarity) are assigned the same cluster_id so they share
generated content (nuggets, blog posts) instead of duplicating API calls.
"""

import json
import logging
import uuid
from collections import deque
from typing import Optional, Set

from .resources.filters import apply_filter

logger = logging.getLogger(__name__)

JACCARD_THRESHOLD = 0.85  # Minimum similarity to share a cluster


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


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets: |A & B| / |A | B|."""
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def compute_cluster_for_user(user_id: str, tree: str) -> Optional[str]:
    """Assign a cluster_id to a user based on branch overlap with other users.

    Algorithm:
    1. Load this user's branch_ids.
    2. Load all other users on the same tree who have branch_ids.
    3. For each, compute Jaccard similarity.
    4. If any existing user has >= JACCARD_THRESHOLD similarity:
       - If that user already has a cluster_id, adopt it.
       - If not, create a new cluster_id and assign to both.
    5. If no match found, create a solo cluster for this user.

    This must be called inside a Flask app context with access to user_db.

    Args:
        user_id: The current user's ID.
        tree: The tree name.

    Returns:
        The cluster_id string (a UUID), or None if user has no branch_ids.
    """
    # Import here to avoid circular imports (auth -> api -> auth)
    from ..auth import User, user_db

    user = user_db.session.query(User).filter_by(id=user_id).scalar()
    if not user or not user.branch_ids:
        return None

    try:
        my_branch = set(json.loads(user.branch_ids))
    except (ValueError, TypeError):
        return None

    if not my_branch:
        return None

    # Load all other users on the same tree who have branches
    other_users = (
        user_db.session.query(User)
        .filter(User.tree == tree, User.id != user_id, User.branch_ids.isnot(None))
        .all()
    )

    best_match = None
    best_similarity = 0.0

    for other in other_users:
        try:
            other_branch = set(json.loads(other.branch_ids))
        except (ValueError, TypeError):
            continue

        sim = _jaccard_similarity(my_branch, other_branch)
        if sim >= JACCARD_THRESHOLD and sim > best_similarity:
            best_similarity = sim
            best_match = other

    if best_match:
        if best_match.cluster_id:
            # Join the existing cluster
            cluster_id = best_match.cluster_id
            logger.info(
                "User %s joining cluster %s (similarity %.2f with user %s)",
                user_id, cluster_id, best_similarity, best_match.id,
            )
        else:
            # Create a new cluster for both users
            cluster_id = str(uuid.uuid4())
            best_match.cluster_id = cluster_id
            logger.info(
                "Creating cluster %s for users %s and %s (similarity %.2f)",
                cluster_id, user_id, best_match.id, best_similarity,
            )
    else:
        # Solo cluster — no one else is similar enough
        cluster_id = str(uuid.uuid4())
        logger.info(
            "Creating solo cluster %s for user %s (no matches above %.0f%%)",
            cluster_id, user_id, JACCARD_THRESHOLD * 100,
        )

    user.cluster_id = cluster_id
    user_db.session.commit()
    return cluster_id
