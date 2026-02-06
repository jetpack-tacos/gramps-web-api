"""add conversations and messages tables for AI chat persistence

Revision ID: f1a2b3c4d5e6
Revises: 2082445b0769
Create Date: 2026-02-06 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from gramps_webapi.auth.sql_guid import GUID


# revision identifiers, used by Alembic.
revision = 'f1a2b3c4d5e6'
down_revision = '2082445b0769'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', GUID(), nullable=False),
        sa.Column('tree', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.Index('idx_conversations_user_tree', 'user_id', 'tree'),
        sa.Index('idx_conversations_updated', 'updated_at'),
    )

    op.create_table(
        'messages',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), nullable=False),
        sa.Column('role', sa.String(10), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.Index('idx_messages_conversation', 'conversation_id', 'created_at'),
    )


def downgrade():
    op.drop_table('messages')
    op.drop_table('conversations')
