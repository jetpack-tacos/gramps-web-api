"""add branch_ids column to users table for branch computation

Revision ID: e5f607a8b9ca
Revises: d4e5f607a8b9
Create Date: 2026-02-12 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e5f607a8b9ca'
down_revision = 'd4e5f607a8b9'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('users', sa.Column('branch_ids', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('users', 'branch_ids')
