"""Per-user home person API resource."""

from flask import abort, jsonify, request
from flask_jwt_extended import get_jwt_identity

from ...auth import get_name, user_db
from ...auth import User
from ..util import abort_with_message
from . import ProtectedResource


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
        return jsonify({"gramps_id": user.home_person}), 200
