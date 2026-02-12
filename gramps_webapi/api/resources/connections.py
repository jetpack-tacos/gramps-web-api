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

"""AI person connections endpoint."""

import re
import uuid
from datetime import datetime, timezone

from flask import current_app
from flask_jwt_extended import get_jwt_identity

from ..auth import has_permissions, require_permissions
from ..llm import generate_person_connections
from ..llm.agent import _ToolContext
from ..llm.deps import AgentDeps
from ..llm.tools import find_coincidences_and_clusters
from ..util import (
    abort_with_message,
    check_quota_ai,
    get_db_outside_request,
    get_tree_from_jwt_or_fail,
    update_usage_ai,
)
from . import ProtectedResource
from ...auth import PersonConnection, user_db
from ...auth.const import PERM_USE_CHAT, PERM_VIEW_PRIVATE


_GRAMPS_ID_PATTERN = re.compile(r"\bI\d+\b")


def _person_from_gramps_id_or_404(db_handle, gramps_id: str):
    person = db_handle.get_person_from_gramps_id(gramps_id)
    if not person:
        abort_with_message(404, f"Person {gramps_id} not found")
    return person


def _get_immediate_family_ids(db_handle, person) -> set[str]:
    """Collect this person's immediate family Gramps IDs."""
    ids = {person.gramps_id}
    person_handle = person.handle

    for family_handle in person.parent_family_list:
        family = db_handle.get_family_from_handle(family_handle)
        if not family:
            continue

        for parent_handle in [family.father_handle, family.mother_handle]:
            if parent_handle:
                parent = db_handle.get_person_from_handle(parent_handle)
                if parent:
                    ids.add(parent.gramps_id)

        for child_ref in family.child_ref_list:
            if child_ref.ref == person_handle:
                continue
            sibling = db_handle.get_person_from_handle(child_ref.ref)
            if sibling:
                ids.add(sibling.gramps_id)

    for family_handle in person.family_list:
        family = db_handle.get_family_from_handle(family_handle)
        if not family:
            continue

        spouse_handle = None
        if family.father_handle == person_handle:
            spouse_handle = family.mother_handle
        elif family.mother_handle == person_handle:
            spouse_handle = family.father_handle

        if spouse_handle:
            spouse = db_handle.get_person_from_handle(spouse_handle)
            if spouse:
                ids.add(spouse.gramps_id)

        for child_ref in family.child_ref_list:
            child = db_handle.get_person_from_handle(child_ref.ref)
            if child:
                ids.add(child.gramps_id)

    return ids


def _filter_findings_for_scope(findings_text: str, scope_ids: set[str]) -> str:
    """Keep only findings that explicitly mention someone in scope."""
    if not findings_text:
        return ""

    findings = [block.strip() for block in findings_text.split("\n\n") if block.strip()]
    scoped = []
    for finding in findings:
        ids_in_finding = set(_GRAMPS_ID_PATTERN.findall(finding))
        if ids_in_finding & scope_ids:
            scoped.append(finding)
    return "\n\n".join(scoped[:8])


class PersonConnectionsResource(ProtectedResource):
    """Get AI-generated connection narrative for a person."""

    def get(self, gramps_id):
        """Get cached person connections or generate on first request."""
        require_permissions({PERM_USE_CHAT})

        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})

        config = current_app.config
        model_name = config.get("LLM_MODEL")
        if not model_name:
            abort_with_message(500, "No LLM model configured")

        db_handle = None
        try:
            db_handle = get_db_outside_request(
                tree=tree,
                view_private=include_private,
                readonly=True,
                user_id=user_id,
            )
            person = _person_from_gramps_id_or_404(db_handle, gramps_id)
            person_handle = person.handle
            scope_ids = _get_immediate_family_ids(db_handle, person)
        finally:
            if db_handle:
                db_handle.close()

        cached = PersonConnection.query.filter_by(
            tree=tree, person_handle=person_handle
        ).first()
        if cached:
            return {
                "data": {
                    "id": cached.id,
                    "person_handle": cached.person_handle,
                    "gramps_id": gramps_id,
                    "content": cached.content,
                    "model": cached.model,
                    "created_at": cached.created_at.isoformat() if cached.created_at else None,
                }
            }, 200

        check_quota_ai(requested=1)

        deps = AgentDeps(
            tree=tree,
            include_private=include_private,
            max_context_length=config.get("LLM_MAX_CONTEXT_LENGTH", 50000),
            user_id=user_id,
        )
        ctx = _ToolContext(deps)

        findings_text = find_coincidences_and_clusters(
            ctx,
            category="all",
            max_results=15,
            person_subset=scope_ids,
        )
        scoped_findings = _filter_findings_for_scope(findings_text, scope_ids)

        if not scoped_findings:
            content = (
                f"No strong pattern clusters found yet for [{gramps_id}](/person/{gramps_id}) "
                "and immediate family. Add more events, places, and relatives to reveal richer connections."
            )
        else:
            content, _metadata = generate_person_connections(
                person_gramps_id=gramps_id,
                findings_text=scoped_findings,
                tree=tree,
                include_private=include_private,
                user_id=user_id,
            )
            if not content:
                abort_with_message(500, "Failed to generate person connections")

        connection_id = str(uuid.uuid4())
        new_connection = PersonConnection(
            id=connection_id,
            tree=tree,
            person_handle=person_handle,
            content=content,
            generated_by=user_id,
            model=model_name,
            created_at=datetime.now(timezone.utc),
        )
        user_db.session.add(new_connection)
        user_db.session.commit()
        update_usage_ai(new=1)

        return {
            "data": {
                "id": connection_id,
                "person_handle": person_handle,
                "gramps_id": gramps_id,
                "content": content,
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        }, 200
