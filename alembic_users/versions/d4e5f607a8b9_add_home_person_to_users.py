"""add home_person column to users table

Revision ID: d4e5f607a8b9
Revises: c3d4e5f607a8
Create Date: 2026-02-12 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd4e5f607a8b9'
down_revision = 'c3d4e5f607a8'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('users', sa.Column('home_person', sa.String(), nullable=True))


def downgrade():
    op.drop_column('users', 'home_person')
