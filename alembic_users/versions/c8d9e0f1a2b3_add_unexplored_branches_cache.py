"""add unexplored_branches_cache table

Revision ID: c8d9e0f1a2b3
Revises: b707a8b9cadd
Create Date: 2026-02-20 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c8d9e0f1a2b3"
down_revision = "b707a8b9cadd"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "unexplored_branches_cache",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tree", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("home_person_id", sa.String(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("dead_ends_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_unexplored_tree_user",
        "unexplored_branches_cache",
        ["tree", "user_id"],
    )


def downgrade():
    op.drop_index(
        "idx_unexplored_tree_user", table_name="unexplored_branches_cache"
    )
    op.drop_table("unexplored_branches_cache")
