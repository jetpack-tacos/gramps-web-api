#
# GenAI - Geocoding API Resource
#
# Provides endpoint for batch geocoding places.
#

"""Geocoding API resource."""

import logging

from flask import jsonify
from flask_jwt_extended import get_jwt_identity

from ...auth.const import PERM_EDIT_OBJ
from ..auth import has_permissions
from ..geocode import geocode_all_places_task
from ..tasks import AsyncResult, make_task_response, run_task
from ..util import get_tree_from_jwt, abort_with_message
from . import ProtectedResource
from .emit import GrampsJSONEncoder

logger = logging.getLogger(__name__)


class GeocodeResource(ProtectedResource, GrampsJSONEncoder):
    """Geocoding resource for batch geocoding places."""

    def post(self):
        """
        Trigger batch geocoding of all places.

        POST /api/places/geocode/

        Returns task ID for progress tracking.
        """
        # Check permissions
        if not has_permissions({PERM_EDIT_OBJ}):
            abort_with_message(403, "Edit permission required")

        tree = get_tree_from_jwt()
        user_id = get_jwt_identity()

        logger.info(f"Geocode request: tree={tree}, user={user_id}")

        # Run as Celery task
        task = run_task(
            geocode_all_places_task,
            tree=tree,
            user_id=user_id,
            skip_existing=True,
        )

        if isinstance(task, AsyncResult):
            logger.info(f"Geocode task dispatched: {task.id}")
            return make_task_response(task)

        # Synchronous execution (no Celery)
        return jsonify(task), 201
