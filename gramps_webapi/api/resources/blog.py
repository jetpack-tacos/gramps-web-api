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

"""AI-generated blog post endpoints."""

from datetime import datetime

from flask import current_app, request
from flask_jwt_extended import get_jwt_identity
from gramps.gen.lib import Source, Note, Tag
from gramps.gen.db import DbTxn

from ..llm import generate_blog_post
from ..util import (
    abort_with_message,
    check_quota_ai,
    get_db_outside_request,
    get_logger,
    get_tree_from_jwt_or_fail,
    update_usage_ai,
)
from . import ProtectedResource
from ...auth.const import PERM_USE_CHAT, PERM_VIEW_PRIVATE, PERM_ADD_OBJ
from ..auth import has_permissions, require_permissions


def _get_last_blog_post_date(db_handle):
    """Get the date of the most recent blog post (Source with 'Blog' tag)."""
    blog_tag = None
    for tag_handle in db_handle.get_tag_handles():
        tag = db_handle.get_tag_from_handle(tag_handle)
        if tag.name == "Blog":
            blog_tag = tag
            break

    if not blog_tag:
        return None

    latest_timestamp = None
    for source_handle in db_handle.get_source_handles():
        source = db_handle.get_source_from_handle(source_handle)
        if blog_tag.handle in source.tag_list:
            if latest_timestamp is None or source.change > latest_timestamp:
                latest_timestamp = source.change

    if latest_timestamp:
        return datetime.fromtimestamp(latest_timestamp)
    return None


def _get_existing_blog_info(db_handle):
    """Get titles and featured person IDs from existing blog posts for diversity."""
    titles = []
    featured_ids = set()
    blog_tag = None

    for tag_handle in db_handle.get_tag_handles():
        tag = db_handle.get_tag_from_handle(tag_handle)
        if tag.name == "Blog":
            blog_tag = tag
            break

    if not blog_tag:
        return titles, featured_ids

    import re
    for source_handle in db_handle.get_source_handles():
        source = db_handle.get_source_from_handle(source_handle)
        if blog_tag.handle in source.tag_list:
            titles.append(source.title)
            # Extract person IDs from blog content
            for note_handle in source.get_note_list():
                note = db_handle.get_note_from_handle(note_handle)
                text = note.get()
                ids = re.findall(r'/person/(I\d{4,5})', text)
                featured_ids.update(ids)

    return titles, featured_ids


class BlogGenerateResource(ProtectedResource):
    """Generate a new AI blog post."""

    def post(self):
        """Generate a new blog post as a Source with Blog tag."""
        require_permissions({PERM_USE_CHAT, PERM_ADD_OBJ})
        check_quota_ai(requested=1)
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})

        config = current_app.config
        model_name = config.get("LLM_MODEL")
        if not model_name:
            abort_with_message(500, "No LLM model configured")

        logger = get_logger()
        logger.info("Generating blog post for tree %s", tree)

        db_handle = None
        try:
            db_handle = get_db_outside_request(
                tree=tree,
                view_private=include_private,
                readonly=False,
                user_id=user_id,
            )

            # Get existing titles and featured person IDs for diversity
            previous_titles, featured_ids = _get_existing_blog_info(db_handle)
            logger.info("Found %d existing blog posts featuring %d people", len(previous_titles), len(featured_ids))

            # Single-shot Gemini call with pre-gathered context
            title, content, metadata = generate_blog_post(
                tree=tree,
                include_private=include_private,
                user_id=user_id,
                previous_titles=previous_titles,
                previously_featured_ids=featured_ids,
            )

            if not content:
                abort_with_message(500, "No blog content generated")

            logger.info("Generated blog post '%s': %d chars", title, len(content))

            # Create a Source object for the blog post
            with DbTxn("Generate AI blog post", db_handle) as trans:
                # Create or get the Blog tag
                blog_tag = None
                for tag_handle in db_handle.get_tag_handles():
                    tag = db_handle.get_tag_from_handle(tag_handle)
                    if tag.name == "Blog":
                        blog_tag = tag
                        break

                if not blog_tag:
                    blog_tag = Tag()
                    blog_tag.set_name("Blog")
                    blog_tag_handle = db_handle.add_tag(blog_tag, trans)
                else:
                    blog_tag_handle = blog_tag.handle

                # Create the source
                source = Source()
                source.set_title(title)
                source.set_author(f"AI Generated ({model_name})")
                source.tag_list.append(blog_tag_handle)

                # Create a note with the blog content
                note = Note()
                note.set(content)
                note.set_type("Blog Post")
                note_handle = db_handle.add_note(note, trans)

                # Attach note to source
                source.add_note(note_handle)

                # Add the source
                source_handle = db_handle.add_source(source, trans)

            # Get the created source with its Gramps ID
            created_source = db_handle.get_source_from_handle(source_handle)
            gramps_id = created_source.gramps_id

            update_usage_ai(new=1)

            logger.info(
                "Created blog post: %s (Gramps ID: %s)",
                title,
                gramps_id,
            )

            return {
                "data": {
                    "gramps_id": gramps_id,
                    "title": title,
                    "content_length": len(content),
                }
            }, 201

        finally:
            if db_handle:
                db_handle.close()


class BlogCheckResource(ProtectedResource):
    """Check if a new blog post should be generated."""

    def get(self):
        """Check last blog post date and return whether to generate a new one."""
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})

        days_threshold = request.args.get('days', 7, type=int)

        db_handle = None
        try:
            db_handle = get_db_outside_request(
                tree=tree,
                view_private=include_private,
                readonly=True,
                user_id=user_id,
            )

            last_post_date = _get_last_blog_post_date(db_handle)

            if not last_post_date:
                return {
                    "data": {
                        "should_generate": True,
                        "reason": "No blog posts exist yet",
                        "last_post_date": None,
                    }
                }, 200

            days_since_last = (datetime.now() - last_post_date).days

            should_generate = days_since_last >= days_threshold

            return {
                "data": {
                    "should_generate": should_generate,
                    "days_since_last": days_since_last,
                    "last_post_date": last_post_date.isoformat(),
                    "threshold_days": days_threshold,
                }
            }, 200

        finally:
            if db_handle:
                db_handle.close()
