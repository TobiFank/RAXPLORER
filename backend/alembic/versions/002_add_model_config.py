# alembic/versions/002_add_model_config.py
"""add model config table

Revision ID: 002_add_model_config
Create Date: 2024-03-08 11:00:00.000000
"""
import sqlalchemy as sa
from alembic import op

revision = '002_add_model_config'
down_revision = '001_initial_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'model_configs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('api_key', sa.String(), nullable=True),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=False, server_default='0.7'),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'))
    )


def downgrade() -> None:
    op.drop_table('model_configs')
