# alembic/versions/001_initial_tables.py
"""initial tables

Revision ID: 001_initial_tables
Create Date: 2024-03-08 10:00:00.000000
"""
import sqlalchemy as sa
from alembic import op

revision = '001_initial_tables'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create chats table
    op.create_table(
        'chats',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False)
    )

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('chat_id', sa.String(), sa.ForeignKey('chats.id'), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('model_provider', sa.String()),
        sa.Column('model_name', sa.String()),
        sa.Column('temperature', sa.Float())
    )

    # Create files table
    op.create_table(
        'files',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('pages', sa.Integer()),
        sa.Column('uploaded_at', sa.DateTime(), nullable=False),
        sa.Column('vectorized', sa.Boolean(), default=False)
    )

    # Create file_chunks table
    op.create_table(
        'file_chunks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('file_id', sa.String(), sa.ForeignKey('files.id'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('vector_id', sa.String())
    )
