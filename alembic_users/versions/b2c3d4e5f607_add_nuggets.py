"""add nuggets table for interesting fact nuggets on home page

Revision ID: b2c3d4e5f607
Revises: a1b2c3d4e5f6
Create Date: 2026-02-10 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from gramps_webapi.auth.sql_guid import GUID


# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f607'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'nuggets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('tree', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('nugget_type', sa.String(), nullable=False),  # 'person', 'event', 'family', etc.
        sa.Column('target_handle', sa.String(), nullable=True),  # Handle of person/event/etc
        sa.Column('target_gramps_id', sa.String(), nullable=True),  # Gramps ID for easy linking
        sa.Column('generated_by', GUID(), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('display_count', sa.Integer(), nullable=False, server_default='0'),  # Track how many times shown
        sa.Column('click_count', sa.Integer(), nullable=False, server_default='0'),  # Track engagement
        sa.ForeignKeyConstraint(['generated_by'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index('idx_nuggets_tree_created', 'nuggets', ['tree', 'created_at'])
    op.create_index('idx_nuggets_tree_display_count', 'nuggets', ['tree', 'display_count'])


def downgrade():
    op.drop_index('idx_nuggets_tree_display_count', table_name='nuggets')
    op.drop_index('idx_nuggets_tree_created', table_name='nuggets')
    op.drop_table('nuggets')
