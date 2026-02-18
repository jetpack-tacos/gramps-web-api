"""add grounding alert timestamp columns

Revision ID: d9f0a1b2c3d4
Revises: c8a1b2c3d4e5
Create Date: 2026-02-18 16:10:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d9f0a1b2c3d4"
down_revision = "c8a1b2c3d4e5"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "search_grounding_usage_monthly",
        sa.Column("soft_cap_alert_sent_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "search_grounding_usage_monthly",
        sa.Column("hard_cap_alert_sent_at", sa.DateTime(), nullable=True),
    )


def downgrade():
    op.drop_column("search_grounding_usage_monthly", "hard_cap_alert_sent_at")
    op.drop_column("search_grounding_usage_monthly", "soft_cap_alert_sent_at")
