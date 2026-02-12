"""add this_day_cache table for daily family event digest

Revision ID: c3d4e5f607a8
Revises: b2c3d4e5f607
Create Date: 2026-02-11 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3d4e5f607a8'
down_revision = 'b2c3d4e5f607'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'this_day_cache',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('tree', sa.String(), nullable=False),
        sa.Column('month_day', sa.String(5), nullable=False),  # e.g., "02-14"
        sa.Column('content', sa.Text(), nullable=False),  # JSON with events and narrative
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('idx_this_day_tree_month_day', 'this_day_cache', ['tree', 'month_day'])


def downgrade():
    op.drop_index('idx_this_day_tree_month_day', table_name='this_day_cache')
    op.drop_table('this_day_cache')
