# alembic/versions/003_add_ollama_model.py
"""add ollama model field

Revision ID: 003_add_ollama_model
Create Date: 2024-03-08 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# Handle SQLite limitations for altering columns
def _alter_column_sqlite(
        table_name: str,
        column_name: str,
        nullable: bool = True,
        type_: sa.types.TypeEngine = None,
        default: any = None,
        server_default: str = None,
):
    """Helper function for SQLite column alterations"""
    with op.batch_alter_table(table_name) as batch_op:
        if type_:
            batch_op.alter_column(
                column_name,
                type_=type_,
                existing_nullable=nullable,
                existing_server_default=server_default,
            )

revision = '003_add_ollama_model'
down_revision = '002_add_model_config'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # For SQLite, we need to use batch operations
    with op.batch_alter_table('model_configs') as batch_op:
        batch_op.add_column(sa.Column('ollamaModel', sa.String(), nullable=True))

    # Migrate existing Ollama configs to use ollamaModel
    op.execute(
        """
        UPDATE model_configs 
        SET ollamaModel = model 
        WHERE provider = 'ollama' AND model IS NOT NULL
        """
    )

def downgrade() -> None:
    # For SQLite, we need to use batch operations
    with op.batch_alter_table('model_configs') as batch_op:
        batch_op.drop_column('ollamaModel')