"""Phase 2: chat_turns table for persisting conversation history.

Revision ID: 0002_chat_turns
Revises: 0001_initial
Create Date: 2026-10-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002_chat_turns"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_turns",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("session_id", sa.Uuid(), sa.ForeignKey("care_sessions.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("session_id", "sequence", name="uq_chat_turns_session_seq"),
    )
    op.create_index("ix_chat_turns_session_created", "chat_turns", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_chat_turns_session_created")
    op.drop_table("chat_turns")
