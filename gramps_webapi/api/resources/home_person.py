"""Per-user home person API resource."""

import logging

from flask import jsonify, request
from flask_jwt_extended import get_jwt_identity

from ...auth import user_db
from ...auth import User
from ..util import abort_with_message, get_db_handle
from .branch import _recompute_and_store_branch
from . import ProtectedResource

logger = logging.getLogger(__name__)


class UserHomePersonResource(ProtectedResource):
    """Resource for getting/setting the current user's home person."""

    def get(self):
        """Get the current user's home person gramps_id."""
        user_id = get_jwt_identity()
        user = user_db.session.query(User).filter_by(id=user_id).scalar()
        if user is None:
            abort_with_message(401, "User not found")
        return jsonify({"gramps_id": user.home_person}), 200

    def put(self):
        """Set the current user's home person gramps_id."""
        payload = request.json
        if not payload or "gramps_id" not in payload:
            abort_with_message(400, "Missing 'gramps_id' in request body")

        gramps_id = payload["gramps_id"]
        user_id = get_jwt_identity()
        user = user_db.session.query(User).filter_by(id=user_id).scalar()
        if user is None:
            abort_with_message(401, "User not found")

        user.home_person = gramps_id
        user_db.session.commit()

        # Recompute branch in background — non-blocking, errors are logged
        try:
            db_handle = get_db_handle()
            _recompute_and_store_branch(user, db_handle)
        except Exception as e:
            logger.warning("Branch recomputation failed: %s", e)

        return jsonify({"gramps_id": user.home_person}), 200
