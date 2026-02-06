"""Conversation management endpoints for AI chat."""

import json
import uuid
from datetime import datetime

from flask_jwt_extended import get_jwt_identity
from webargs import fields

from ..util import (
    get_tree_from_jwt_or_fail,
    use_args,
    abort_with_message,
)
from . import ProtectedResource
from ...auth import user_db, Conversation, Message
from ...auth.const import PERM_USE_CHAT
from ..auth import require_permissions


class ConversationsResource(ProtectedResource):
    """List conversations for the current user."""

    @use_args(
        {
            "page": fields.Int(load_default=1),
            "pagesize": fields.Int(load_default=20),
        },
        location="query",
    )
    def get(self, args):
        """Get list of conversations."""
        require_permissions({PERM_USE_CHAT})
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()

        page = max(1, args["page"])
        pagesize = min(100, max(1, args["pagesize"]))
        offset = (page - 1) * pagesize

        conversations = (
            user_db.session.query(Conversation)
            .filter_by(user_id=user_id, tree=tree)
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(pagesize)
            .all()
        )

        result = []
        for conv in conversations:
            last_msg = (
                user_db.session.query(Message)
                .filter_by(conversation_id=conv.id)
                .order_by(Message.created_at.desc())
                .first()
            )
            msg_count = (
                user_db.session.query(Message)
                .filter_by(conversation_id=conv.id)
                .count()
            )
            result.append(
                {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                    "message_count": msg_count,
                    "last_message_preview": (
                        last_msg.content[:100] if last_msg else None
                    ),
                }
            )

        return result, 200


class ConversationResource(ProtectedResource):
    """Single conversation with messages."""

    def get(self, conversation_id):
        """Get a conversation with all messages."""
        require_permissions({PERM_USE_CHAT})
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()

        conv = user_db.session.query(Conversation).filter_by(
            id=conversation_id, user_id=user_id, tree=tree
        ).first()

        if not conv:
            abort_with_message(404, "Conversation not found")

        messages = (
            user_db.session.query(Message)
            .filter_by(conversation_id=conv.id)
            .order_by(Message.created_at.asc())
            .all()
        )

        return {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                }
                for msg in messages
            ],
        }, 200

    def delete(self, conversation_id):
        """Delete a conversation and all its messages."""
        require_permissions({PERM_USE_CHAT})
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()

        conv = user_db.session.query(Conversation).filter_by(
            id=conversation_id, user_id=user_id, tree=tree
        ).first()

        if not conv:
            abort_with_message(404, "Conversation not found")

        user_db.session.delete(conv)
        user_db.session.commit()

        return "", 204


def create_conversation(user_id: str, tree: str, title: str | None = None) -> Conversation:
    """Create a new conversation."""
    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_id,
        tree=tree,
        title=title,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    user_db.session.add(conv)
    user_db.session.commit()
    return conv


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> Message:
    """Add a message to a conversation and update its timestamp."""
    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata_json=json.dumps(metadata) if metadata else None,
        created_at=datetime.utcnow(),
    )
    user_db.session.add(msg)

    # Update conversation's updated_at
    conv = user_db.session.query(Conversation).filter_by(id=conversation_id).first()
    if conv:
        conv.updated_at = datetime.utcnow()

    user_db.session.commit()
    return msg


def get_conversation_history(conversation_id: str) -> list[dict]:
    """Get conversation messages formatted for the LLM history."""
    messages = (
        user_db.session.query(Message)
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [
        {"role": msg.role, "message": msg.content}
        for msg in messages
    ]


def auto_title(text: str) -> str:
    """Generate a title from the first user message."""
    title = text.strip()
    if len(title) > 60:
        title = title[:57]
        # Truncate at last word boundary
        last_space = title.rfind(" ")
        if last_space > 20:
            title = title[:last_space]
        title += "..."
    return title


def cleanup_old_conversations(user_id: str, tree: str, max_conversations: int = 100):
    """Delete oldest conversations beyond the limit."""
    count = (
        user_db.session.query(Conversation)
        .filter_by(user_id=user_id, tree=tree)
        .count()
    )
    if count <= max_conversations:
        return

    old_conversations = (
        user_db.session.query(Conversation)
        .filter_by(user_id=user_id, tree=tree)
        .order_by(Conversation.updated_at.desc())
        .offset(max_conversations)
        .all()
    )
    for conv in old_conversations:
        user_db.session.delete(conv)
    user_db.session.commit()
