"""Per-user branch API resource."""

import json
import logging

from flask import jsonify
from flask_jwt_extended import get_jwt_identity

from ...auth import User, user_db
from ..branch import compute_branch_ids
from ..util import abort_with_message, get_db_handle
from . import ProtectedResource

logger = logging.getLogger(__name__)


def _recompute_and_store_branch(user, db_handle):
    """Recompute branch IDs and store on user record.

    Args:
        user: SQLAlchemy User object (must be in session).
        db_handle: Open Gramps DB handle (readonly).

    Returns:
        Set of branch Gramps IDs.
    """
    if not user.home_person:
        user.branch_ids = None
        user_db.session.commit()
        return set()

    branch_ids = compute_branch_ids(db_handle, user.home_person)
    user.branch_ids = json.dumps(sorted(branch_ids))
    user_db.session.commit()
    return branch_ids


def _get_branch_summary(user, db_handle, branch_ids=None):
    """Build a branch summary response.

    Args:
        user: SQLAlchemy User object.
        db_handle: Open Gramps DB handle for looking up names.
        branch_ids: Pre-computed branch IDs (or None to load from user).

    Returns:
        Dict with branch summary.
    """
    if branch_ids is None:
        if user.branch_ids:
            branch_ids = set(json.loads(user.branch_ids))
        else:
            branch_ids = set()

    # Get sample names (up to 10) for the summary
    sample_names = []
    for gramps_id in sorted(branch_ids)[:10]:
        try:
            person = db_handle.get_person_from_gramps_id(gramps_id)
            if person:
                name = person.get_primary_name()
                full_name = f"{name.get_first_name()} {name.get_surname()}".strip()
                if full_name:
                    sample_names.append(
                        {"gramps_id": gramps_id, "name": full_name}
                    )
        except Exception:
            continue

    return {
        "home_person": user.home_person,
        "count": len(branch_ids),
        "sample_names": sample_names,
        "generations_up": 5,
    }


class UserBranchResource(ProtectedResource):
    """Resource for computing and retrieving the user's family branch."""

    def get(self):
        """Get the current user's branch summary."""
        user_id = get_jwt_identity()
        user = user_db.session.query(User).filter_by(id=user_id).scalar()
        if user is None:
            abort_with_message(401, "User not found")

        if not user.home_person:
            return jsonify({
                "home_person": None,
                "count": 0,
                "sample_names": [],
                "generations_up": 5,
            }), 200

        db_handle = get_db_handle()
        summary = _get_branch_summary(user, db_handle)
        return jsonify(summary), 200

    def post(self):
        """Recompute the user's branch from their home person."""
        user_id = get_jwt_identity()
        user = user_db.session.query(User).filter_by(id=user_id).scalar()
        if user is None:
            abort_with_message(401, "User not found")

        if not user.home_person:
            abort_with_message(400, "No home person set. Set one first via PUT /api/users/-/home-person/")

        db_handle = get_db_handle()
        branch_ids = _recompute_and_store_branch(user, db_handle)
        summary = _get_branch_summary(user, db_handle, branch_ids)
        return jsonify(summary), 200
