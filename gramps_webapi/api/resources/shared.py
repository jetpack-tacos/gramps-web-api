"""Shared discovery feed endpoints."""

import json
import uuid
from datetime import datetime, timezone

from flask_jwt_extended import get_jwt_identity
from webargs import fields

from ..util import (
    get_tree_from_jwt_or_fail,
    use_args,
    abort_with_message,
)
from . import ProtectedResource
from ...auth import user_db, SharedDiscovery, User
from ...auth.const import PERM_USE_CHAT
from ..auth import require_permissions


class SharedDiscoveriesResource(ProtectedResource):
    """Create and list shared discoveries for the current tree."""

    @use_args(
        {
            "content": fields.Str(required=True),
            "person_ids": fields.List(fields.Str(), load_default=[]),
        },
        location="json",
    )
    def post(self, args):
        """Save a shared discovery from chat content."""
        require_permissions({PERM_USE_CHAT})
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()

        content = (args["content"] or "").strip()
        if not content:
            abort_with_message(422, "content is required")

        person_ids = sorted({pid.strip() for pid in args.get("person_ids", []) if pid})

        shared = SharedDiscovery(
            id=str(uuid.uuid4()),
            user_id=user_id,
            tree=tree,
            content=content,
            person_ids=json.dumps(person_ids),
            created_at=datetime.now(timezone.utc),
        )
        user_db.session.add(shared)
        user_db.session.commit()

        return {
            "data": {
                "id": shared.id,
                "content": shared.content,
                "person_ids": person_ids,
                "created_at": shared.created_at.isoformat() if shared.created_at else None,
            }
        }, 201

    @use_args(
        {
            "page": fields.Int(load_default=1),
            "pagesize": fields.Int(load_default=20),
        },
        location="query",
    )
    def get(self, args):
        """Get shared discoveries for the current tree, newest first."""
        require_permissions({PERM_USE_CHAT})
        tree = get_tree_from_jwt_or_fail()

        page = max(1, args["page"])
        pagesize = min(100, max(1, args["pagesize"]))
        offset = (page - 1) * pagesize

        rows = (
            user_db.session.query(SharedDiscovery, User)
            .join(User, SharedDiscovery.user_id == User.id)
            .filter(SharedDiscovery.tree == tree)
            .order_by(SharedDiscovery.created_at.desc())
            .offset(offset)
            .limit(pagesize)
            .all()
        )

        discoveries = []
        for shared, user in rows:
            try:
                person_ids = json.loads(shared.person_ids or "[]")
            except (ValueError, TypeError):
                person_ids = []
            discoveries.append(
                {
                    "id": shared.id,
                    "content": shared.content,
                    "person_ids": person_ids if isinstance(person_ids, list) else [],
                    "created_at": shared.created_at.isoformat() if shared.created_at else None,
                    "shared_by": (user.fullname or user.name) if user else None,
                    "user_id": str(shared.user_id),
                }
            )

        return {"data": discoveries}, 200
