#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      GenAI Genealogy
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#

"""Default person API resource."""

from flask import abort, jsonify, request

from ...auth.const import PERM_EDIT_OBJ
from ..auth import require_permissions
from ..util import get_db_handle
from . import ProtectedResource


class DefaultPersonResource(ProtectedResource):
    """Resource for getting/setting the default (home) person."""

    def get(self):
        """Get the current default person handle."""
        db_handle = get_db_handle()
        handle = db_handle.get_default_handle()
        return jsonify({"default_person": handle}), 200

    def put(self):
        """Set the default person handle."""
        require_permissions([PERM_EDIT_OBJ])
        payload = request.json
        if not payload or "handle" not in payload:
            abort(400, description="Missing 'handle' in request body")

        handle = payload["handle"]
        db_handle = get_db_handle(readonly=False)

        if not db_handle.has_person_handle(handle):
            abort(404, description="Person not found")

        db_handle.set_default_person_handle(handle)
        return jsonify({"default_person": handle}), 200
