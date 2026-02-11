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

"""'This Day in Your Family' daily digest endpoint."""

import uuid
import json
from datetime import datetime

from flask import current_app
from flask_jwt_extended import get_jwt_identity

from ..llm import generate_this_day
from ..util import (
    abort_with_message,
    check_quota_ai,
    get_logger,
    get_tree_from_jwt_or_fail,
    update_usage_ai,
)
from . import ProtectedResource
from ...auth import ThisDayCache, user_db
from ...auth.const import PERM_VIEW_PRIVATE
from ..auth import has_permissions


class ThisDayResource(ProtectedResource):
    """Get or generate 'This Day in Your Family' daily digest."""

    def get(self):
        """Get today's family history digest (cached or freshly generated)."""
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})

        # Get today's month and day
        today = datetime.now()
        month_day = f"{today.month:02d}-{today.day:02d}"

        logger = get_logger()
        logger.info("Fetching This Day digest for %s on tree %s", month_day, tree)

        # Check cache first
        cached = ThisDayCache.query.filter_by(tree=tree, month_day=month_day).first()

        if cached:
            logger.info("Returning cached This Day digest for %s", month_day)
            return {
                "month_day": month_day,
                "content": cached.content,
                "cached": True,
                "created_at": cached.created_at.isoformat() if cached.created_at else None,
            }, 200

        # No cache — generate fresh
        config = current_app.config
        model_name = config.get("LLM_MODEL")
        if not model_name:
            abort_with_message(500, "No LLM model configured")

        # Check quota before generating
        check_quota_ai(requested=1)

        logger.info("Generating fresh This Day digest for %s", month_day)

        try:
            content_text, metadata = generate_this_day(
                month=today.month,
                day=today.day,
                tree=tree,
                include_private=include_private,
                user_id=user_id,
            )

            if not content_text:
                abort_with_message(500, "No content generated")

            # Cache the result
            cache_id = str(uuid.uuid4())
            new_cache = ThisDayCache(
                id=cache_id,
                tree=tree,
                month_day=month_day,
                content=content_text,
            )
            user_db.session.add(new_cache)
            user_db.session.commit()

            update_usage_ai(new=1)

            logger.info("Cached new This Day digest for %s", month_day)

            return {
                "month_day": month_day,
                "content": content_text,
                "cached": False,
                "created_at": new_cache.created_at.isoformat() if new_cache.created_at else None,
            }, 200

        except Exception as e:
            logger.error("Error generating This Day digest: %s", e)
            abort_with_message(500, f"Error generating digest: {str(e)}")
