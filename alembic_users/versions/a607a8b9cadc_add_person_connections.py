"""add person_connections table for AI-generated person connection narratives

Revision ID: a607a8b9cadc
Revises: f607a8b9cadb
Create Date: 2026-02-12 21:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from gramps_webapi.auth.sql_guid import GUID


# revision identifiers, used by Alembic.
revision = "a607a8b9cadc"
down_revision = "f607a8b9cadb"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "person_connections",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tree", sa.String(), nullable=False),
        sa.Column("person_handle", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("generated_by", GUID(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["generated_by"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("tree", "person_handle", name="uq_person_connections_tree_handle"),
    )
    op.create_index(
        "idx_person_connections_tree_handle",
        "person_connections",
        ["tree", "person_handle"],
    )


def downgrade():
    op.drop_index("idx_person_connections_tree_handle", table_name="person_connections")
    op.drop_table("person_connections")
