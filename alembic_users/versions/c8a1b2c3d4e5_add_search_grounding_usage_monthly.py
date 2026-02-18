"""add search_grounding_usage_monthly table

Revision ID: c8a1b2c3d4e5
Revises: b707a8b9cadd
Create Date: 2026-02-18 14:30:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c8a1b2c3d4e5"
down_revision = "b707a8b9cadd"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "search_grounding_usage_monthly",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column(
            "grounded_prompts_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "web_search_queries_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "period_start",
            name="uq_search_grounding_usage_monthly_period_start",
        ),
    )
    op.create_index(
        "idx_search_grounding_usage_monthly_period_start",
        "search_grounding_usage_monthly",
        ["period_start"],
    )


def downgrade():
    op.drop_index(
        "idx_search_grounding_usage_monthly_period_start",
        table_name="search_grounding_usage_monthly",
    )
    op.drop_table("search_grounding_usage_monthly")
