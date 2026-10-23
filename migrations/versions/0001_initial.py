"""Phase 1 initial schema: actors, sessions, consent, runs, events, model_runs.

Revision ID: 0001_initial
Revises: -
Create Date: 2026-10-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "demo_actors",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("handle", sa.String(40), unique=True, nullable=False),
        sa.Column("display_name", sa.String(80), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "care_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("actor_id", sa.Uuid(), sa.ForeignKey("demo_actors.id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("modality", sa.String(20), nullable=False, server_default="text"),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("workflow_version", sa.String(40), nullable=False),
        sa.Column("idempotency_key", sa.String(80), unique=True, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "consent_records",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("actor_id", sa.Uuid(), sa.ForeignKey("demo_actors.id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("session_id", sa.Uuid(), sa.ForeignKey("care_sessions.id", ondelete="RESTRICT"), nullable=True, index=True),
        sa.Column("purpose", sa.String(60), nullable=False),
        sa.Column("consent_version", sa.String(40), nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
    )

    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("session_id", sa.Uuid(), sa.ForeignKey("care_sessions.id", ondelete="RESTRICT"), nullable=True, index=True),
        sa.Column("workflow_name", sa.String(80), nullable=False),
        sa.Column("workflow_version", sa.String(40), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False, index=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(60), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("output_ref", sa.String(200), nullable=True),
    )

    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

    op.create_table(
        "workflow_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("aggregate_type", sa.String(40), nullable=False),
        sa.Column("aggregate_id", sa.String(64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("schema_version", sa.String(10), nullable=False, server_default="1"),
        sa.Column("actor", sa.String(80), nullable=False),
        sa.Column("session_id", sa.Uuid(), sa.ForeignKey("care_sessions.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("workflow_runs.id", ondelete="RESTRICT"), nullable=True, index=True),
        sa.Column("trace_id", sa.String(64), nullable=False, index=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("payload", json_type, nullable=False, server_default="{}"),
        sa.UniqueConstraint("aggregate_type", "aggregate_id", "sequence", name="uq_workflow_events_aggregate_seq"),
    )
    op.create_index("ix_workflow_events_session_occurred", "workflow_events", ["session_id", "occurred_at"])

    op.create_table(
        "model_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("workflow_runs.id", ondelete="RESTRICT"), nullable=True, index=True),
        sa.Column("provider", sa.String(40), nullable=False),
        sa.Column("model_id", sa.String(160), nullable=False),
        sa.Column("capability", sa.String(40), nullable=False),
        sa.Column("prompt_version", sa.String(80), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(60), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("model_runs")
    op.drop_index("ix_workflow_events_session_occurred")
    op.drop_table("workflow_events")
    op.drop_table("workflow_runs")
    op.drop_table("consent_records")
    op.drop_table("care_sessions")
    op.drop_table("demo_actors")
