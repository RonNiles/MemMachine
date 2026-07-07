"""Add covering index for the profile-fetch/consolidation hot path.

Ingestion fetches the whole profile per message with

    WHERE set_id = ? AND semantic_category_id = ?
    ORDER BY created_at, id

served (for the filter only) by ``idx_feature_set_id_semantic_category``.
Postgres then had to Sort every matching row by (created_at, id); once a
profile outgrows ``work_mem`` that sort spills to disk (external merge),
which is the superlinear cost observed during large ingestion runs.

This replaces that two-column index with a four-column one whose trailing
``created_at, id`` supply the ORDER BY, so the planner returns rows already
sorted (Index Scan, no Sort node). The new index is a strict
superset-prefix of the old one, so dropping the old one loses no read path
and removes its per-write maintenance cost.

Revision ID: e4b1c2d3f5a6
Revises: c7a2f8e31b90
Create Date: 2026-07-07 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e4b1c2d3f5a6"
down_revision: str | Sequence[str] | None = "c7a2f8e31b90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "idx_feature_set_semantic_category_created_id",
        "feature",
        ["set_id", "semantic_category_id", "created_at", "id"],
        if_not_exists=True,
    )
    # Redundant now: the new index covers (set_id, semantic_category_id) as a
    # leading prefix.
    op.drop_index(
        "idx_feature_set_id_semantic_category",
        table_name="feature",
        if_exists=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.create_index(
        "idx_feature_set_id_semantic_category",
        "feature",
        ["set_id", "semantic_category_id"],
        if_not_exists=True,
    )
    op.drop_index(
        "idx_feature_set_semantic_category_created_id",
        table_name="feature",
        if_exists=True,
    )
