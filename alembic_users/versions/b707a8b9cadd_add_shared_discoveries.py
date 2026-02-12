"""add shared_discoveries table for chat sharing feed

Revision ID: b707a8b9cadd
Revises: a607a8b9cadc
Create Date: 2026-02-12 22:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from gramps_webapi.auth.sql_guid import GUID


# revision identifiers, used by Alembic.
revision = "b707a8b9cadd"
down_revision = "a607a8b9cadc"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "shared_discoveries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("tree", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("person_ids", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "idx_shared_discoveries_tree_created",
        "shared_discoveries",
        ["tree", "created_at"],
    )


def downgrade():
    op.drop_index("idx_shared_discoveries_tree_created", table_name="shared_discoveries")
    op.drop_table("shared_discoveries")
