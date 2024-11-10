# alembic/versions/003_add_system_message.py
"""add system message to model config

Revision ID: 003_add_system_message
Revises: 002_add_model_config
Create Date: 2024-03-10 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '003_add_system_message'
down_revision = '002_add_model_config'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('model_configs', sa.Column('system_message', sa.String(), nullable=True))

def downgrade() -> None:
    op.drop_column('model_configs', 'system_message')