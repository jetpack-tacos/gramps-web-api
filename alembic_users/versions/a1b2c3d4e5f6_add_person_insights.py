"""add person_insights table for AI-generated person page insights

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-02-09 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from gramps_webapi.auth.sql_guid import GUID


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'f1a2b3c4d5e6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'person_insights',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('tree', sa.String(), nullable=False),
        sa.Column('person_handle', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('generated_by', GUID(), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['generated_by'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('tree', 'person_handle', name='uq_person_insight_tree_handle'),
    )
    op.create_index('idx_person_insights_tree_handle', 'person_insights', ['tree', 'person_handle'])


def downgrade():
    op.drop_index('idx_person_insights_tree_handle', table_name='person_insights')
    op.drop_table('person_insights')
