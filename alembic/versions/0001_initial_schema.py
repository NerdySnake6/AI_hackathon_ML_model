"""Create initial query labeling schema.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create catalog, query, prediction, and review tables."""
    op.create_table(
        "search_queries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "catalog_titles",
        sa.Column("id", sa.String(length=100), primary_key=True),
        sa.Column("canonical_title", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=50), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("popularity", sa.Float(), nullable=False),
        sa.Column("external_source", sa.String(length=100), nullable=True),
        sa.Column("external_id", sa.String(length=100), nullable=True),
    )
    op.create_table(
        "title_aliases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title_id", sa.String(length=100), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("normalized_alias", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["title_id"], ["catalog_titles.id"]),
    )
    op.create_index(
        "ix_title_aliases_normalized_alias",
        "title_aliases",
        ["normalized_alias"],
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("is_prof_video", sa.Boolean(), nullable=False),
        sa.Column("domain_label", sa.String(length=50), nullable=False),
        sa.Column("content_type", sa.String(length=50), nullable=True),
        sa.Column("title_id", sa.String(length=100), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=50), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["query_id"], ["search_queries.id"]),
        sa.ForeignKeyConstraint(["title_id"], ["catalog_titles.id"]),
    )
    op.create_table(
        "prediction_candidates",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("title_id", sa.String(length=100), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("matched_alias", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
        sa.ForeignKeyConstraint(["title_id"], ["catalog_titles.id"]),
    )
    op.create_table(
        "review_queue",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
    )


def downgrade() -> None:
    """Drop query labeling tables."""
    op.drop_table("review_queue")
    op.drop_table("prediction_candidates")
    op.drop_table("predictions")
    op.drop_index("ix_title_aliases_normalized_alias", table_name="title_aliases")
    op.drop_table("title_aliases")
    op.drop_table("catalog_titles")
    op.drop_table("search_queries")

