#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      Jer Olson
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

"""AI-generated interesting nuggets endpoints."""

import uuid
import random

from flask import current_app, request
from flask_jwt_extended import get_jwt_identity
from gramps.gen.errors import HandleError

from ..llm import generate_nuggets
from ..util import (
    abort_with_message,
    check_quota_ai,
    get_db_outside_request,
    get_logger,
    get_tree_from_jwt_or_fail,
    update_usage_ai,
)
from . import ProtectedResource
from ...auth import Nugget, user_db
from ...auth.const import PERM_USE_CHAT, PERM_VIEW_PRIVATE
from ..auth import has_permissions, require_permissions




class NuggetsListResource(ProtectedResource):
    """List and generate nuggets."""

    def get(self):
        """Get random nuggets for display on home page."""
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()

        # Get limit from query params (default 5)
        limit = request.args.get('limit', 5, type=int)
        if limit > 20:
            limit = 20

        # Get nuggets for this tree, weighted by least shown
        nuggets = (
            Nugget.query
            .filter_by(tree=tree)
            .order_by(Nugget.display_count.asc(), Nugget.created_at.desc())
            .limit(limit * 2)  # Get more than needed for randomness
            .all()
        )

        if not nuggets:
            get_logger().info("No nuggets found for tree %s", tree)
            return {"data": []}, 200

        # Randomly select from the least-shown nuggets
        selected = random.sample(nuggets, min(limit, len(nuggets)))

        # Increment display count for selected nuggets
        for nugget in selected:
            nugget.display_count += 1
        user_db.session.commit()

        result = []
        for nugget in selected:
            result.append({
                "id": nugget.id,
                "content": nugget.content,
                "nugget_type": nugget.nugget_type,
                "target_handle": nugget.target_handle,
                "target_gramps_id": nugget.target_gramps_id,
                "created_at": nugget.created_at.isoformat() if nugget.created_at else None,
                "display_count": nugget.display_count,
                "click_count": nugget.click_count,
            })

        return {"data": result}, 200

    def post(self):
        """Generate new nuggets using AI."""
        require_permissions({PERM_USE_CHAT})
        check_quota_ai(requested=1)
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})

        config = current_app.config
        model_name = config.get("LLM_MODEL")
        if not model_name:
            abort_with_message(500, "No LLM model configured")

        logger = get_logger()
        logger.info("Generating nuggets for tree %s", tree)

        db_handle = None
        try:
            db_handle = get_db_outside_request(
                tree=tree,
                view_private=include_private,
                readonly=True,
                user_id=user_id,
            )

            # Single-shot Gemini call with pre-gathered context (no agent loop)
            answer_text, metadata = generate_nuggets(
                tree=tree,
                include_private=include_private,
                user_id=user_id,
            )

            if not answer_text:
                abort_with_message(500, "No nuggets generated")

            logger.info("Generated nuggets response: %d chars", len(answer_text))
            logger.info("First 500 chars: %s", answer_text[:500])

            # Parse the nuggets from the response
            # Expected format: "1. Nugget text [I1234]\n2. Nugget text [I5678]\n..."
            nuggets_created = []
            lines = answer_text.strip().split('\n')
            logger.info("Split into %d lines", len(lines))

            for line in lines:
                line = line.strip()
                if not line or not line[0].isdigit():
                    logger.debug("Skipping line (empty or doesn't start with digit): %s", line[:100] if line else "(empty)")
                    continue

                # Remove leading number and period/dot
                if '. ' in line:
                    line = line.split('. ', 1)[1]
                elif ') ' in line:
                    line = line.split(') ', 1)[1]

                # Extract Gramps ID if present
                gramps_id = None
                target_handle = None
                if '[' in line and ']' in line:
                    start = line.rfind('[')
                    end = line.rfind(']')
                    gramps_id = line[start+1:end]
                    content = line[:start].strip()

                    # Try to get the handle for this Gramps ID
                    try:
                        if gramps_id.startswith('I'):  # Person
                            person = db_handle.get_person_from_gramps_id(gramps_id)
                            if person:
                                target_handle = person.handle
                        elif gramps_id.startswith('E'):  # Event
                            event = db_handle.get_event_from_gramps_id(gramps_id)
                            if event:
                                target_handle = event.handle
                        elif gramps_id.startswith('F'):  # Family
                            family = db_handle.get_family_from_gramps_id(gramps_id)
                            if family:
                                target_handle = family.handle
                    except (HandleError, AttributeError):
                        logger.warning("Could not find handle for Gramps ID %s", gramps_id)
                else:
                    content = line

                # Determine nugget type
                nugget_type = 'general'
                if gramps_id:
                    if gramps_id.startswith('I'):
                        nugget_type = 'person'
                    elif gramps_id.startswith('E'):
                        nugget_type = 'event'
                    elif gramps_id.startswith('F'):
                        nugget_type = 'family'

                # Create the nugget
                nugget_id = str(uuid.uuid4())
                new_nugget = Nugget(
                    id=nugget_id,
                    tree=tree,
                    content=content,
                    nugget_type=nugget_type,
                    target_handle=target_handle,
                    target_gramps_id=gramps_id,
                    generated_by=user_id,
                    model=model_name,
                )
                user_db.session.add(new_nugget)
                nuggets_created.append({
                    "id": nugget_id,
                    "content": content,
                    "nugget_type": nugget_type,
                    "target_gramps_id": gramps_id,
                })

            user_db.session.commit()
            update_usage_ai(new=1)

            logger.info("Created %d nuggets for tree %s", len(nuggets_created), tree)

            return {"data": nuggets_created, "count": len(nuggets_created)}, 201

        finally:
            if db_handle:
                db_handle.close()


class NuggetResource(ProtectedResource):
    """Individual nugget operations."""

    def post(self, nugget_id):
        """Track click on a nugget."""
        tree = get_tree_from_jwt_or_fail()

        nugget = Nugget.query.filter_by(id=nugget_id, tree=tree).first()
        if not nugget:
            abort_with_message(404, "Nugget not found")

        nugget.click_count += 1
        user_db.session.commit()

        return {"data": {"click_count": nugget.click_count}}, 200

    def delete(self, nugget_id):
        """Delete a nugget."""
        require_permissions({PERM_USE_CHAT})
        tree = get_tree_from_jwt_or_fail()

        nugget = Nugget.query.filter_by(id=nugget_id, tree=tree).first()
        if not nugget:
            abort_with_message(404, "Nugget not found")

        user_db.session.delete(nugget)
        user_db.session.commit()

        return {}, 204
