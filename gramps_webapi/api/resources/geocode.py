#
# GenAI - Geocoding API Resource
#
# Provides endpoint for batch geocoding places.
#

"""Geocoding API resource."""

from flask import jsonify
from flask_jwt_extended import get_jwt_identity

from ...auth.const import PERM_EDIT_OBJ
from ..auth import has_permissions
from ..geocode import geocode_all_places_task
from ..tasks import AsyncResult, make_task_response, run_task
from ..util import get_tree_from_jwt, abort_with_message
from . import ProtectedResource
from .emit import GrampsJSONEncoder


class GeocodeResource(ProtectedResource, GrampsJSONEncoder):
    """Geocoding resource for batch geocoding places."""

    def post(self):
        """
        Trigger batch geocoding of all places.
        
        POST /api/places/geocode/
        
        Returns task ID for progress tracking.
        """
        import logging
        import traceback
        import os
        
        # Setup debug logging to file
        log_file = '/app/media/debug.log'
        logging.basicConfig(filename=log_file, level=logging.DEBUG)
        
        try:
            logging.info("Starting geocode request")
            
            # Check permissions
            if not has_permissions({PERM_EDIT_OBJ}):
                logging.warning("Permission denied")
                abort_with_message(403, "Edit permission required")
            
            tree = get_tree_from_jwt()
            user_id = get_jwt_identity()
            
            logging.info(f"Tree: {tree}, User: {user_id}")
            
            # Run as Celery task
            logging.info("Dispatching task...")
            task = run_task(
                geocode_all_places_task,
                tree=tree,
                user_id=user_id,
                skip_existing=True,
            )
            logging.info(f"Task dispatched: {task}")
            
            if isinstance(task, AsyncResult):
                return make_task_response(task)
            
            # Synchronous execution (no Celery)
            return jsonify(task), 201
            
        except Exception as e:
            error_msg = f"ERROR in geocode post: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            # Log to stdout as well just in case
            print(error_msg)
            # Return a JSON error instead of 500 HTML
            return {"error": str(e), "traceback": traceback.format_exc()}, 500
