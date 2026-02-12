"""add cluster_id to users and nuggets tables for user clustering

Revision ID: f607a8b9cadb
Revises: e5f607a8b9ca
Create Date: 2026-02-12 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f607a8b9cadb'
down_revision = 'e5f607a8b9ca'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('users', sa.Column('cluster_id', sa.String(), nullable=True))
    op.add_column('nuggets', sa.Column('cluster_id', sa.String(), nullable=True))
    op.create_index('idx_nuggets_cluster_id', 'nuggets', ['cluster_id'])


def downgrade():
    op.drop_index('idx_nuggets_cluster_id', table_name='nuggets')
    op.drop_column('nuggets', 'cluster_id')
    op.drop_column('users', 'cluster_id')
