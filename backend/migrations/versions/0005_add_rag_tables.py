"""add passages, generation runs, citations, feedback and daily usage tables

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('document_passages',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('index_version_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('language', sa.String(length=2), nullable=False),
    sa.Column('char_start', sa.Integer(), nullable=True),
    sa.Column('char_end', sa.Integer(), nullable=True),
    sa.Column('estimated_tokens', sa.Integer(), nullable=False),
    sa.Column('content_sha256', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("language IN ('en', 'es', 'fr')", name='ck_document_passages_language'),
    sa.CheckConstraint('ordinal >= 0', name='ck_document_passages_ordinal'),
    sa.CheckConstraint('char_length(content_sha256) = 64', name='ck_document_passages_sha256_len'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.ForeignKeyConstraint(['index_version_id'], ['index_versions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('index_version_id', 'document_id', 'ordinal', name='uq_document_passages_version_doc_ord')
    )
    op.create_index('ix_document_passages_version_doc', 'document_passages', ['index_version_id', 'document_id'], unique=False)
    op.create_index('ix_document_passages_content_sha256', 'document_passages', ['content_sha256'], unique=False)

    op.create_table('generation_runs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('question_sha256', sa.String(length=64), nullable=False),
    sa.Column('question_text', sa.String(length=500), nullable=True),
    sa.Column('index_version_id', sa.UUID(), nullable=True),
    sa.Column('passage_artifact_version', sa.String(length=64), nullable=True),
    sa.Column('retrieval_mode', sa.String(length=16), nullable=True),
    sa.Column('languages', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('provider', sa.String(length=64), nullable=True),
    sa.Column('model', sa.String(length=128), nullable=True),
    sa.Column('prompt_version', sa.String(length=32), nullable=False),
    sa.Column('context_builder_version', sa.String(length=32), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('abstained', sa.Boolean(), nullable=False),
    sa.Column('fallback_used', sa.Boolean(), nullable=False),
    sa.Column('abstention_reason', sa.String(length=48), nullable=True),
    sa.Column('fallback_reason', sa.String(length=48), nullable=True),
    sa.Column('candidate_count', sa.Integer(), nullable=False),
    sa.Column('passage_count', sa.Integer(), nullable=False),
    sa.Column('claim_count', sa.Integer(), nullable=False),
    sa.Column('citation_count', sa.Integer(), nullable=False),
    sa.Column('retrieval_ms', sa.Numeric(precision=10, scale=3), nullable=True),
    sa.Column('provider_ms', sa.Numeric(precision=10, scale=3), nullable=True),
    sa.Column('validation_ms', sa.Numeric(precision=10, scale=3), nullable=True),
    sa.Column('total_ms', sa.Numeric(precision=10, scale=3), nullable=True),
    sa.Column('input_tokens', sa.Integer(), nullable=True),
    sa.Column('output_tokens', sa.Integer(), nullable=True),
    sa.Column('total_tokens', sa.Integer(), nullable=True),
    sa.Column('estimated_cost', sa.Numeric(precision=12, scale=6), nullable=True),
    sa.Column('cache_status', sa.String(length=8), nullable=False),
    sa.Column('error_code', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('answered', 'abstained', 'fallback')", name='ck_generation_runs_status'),
    sa.CheckConstraint("cache_status IN ('hit', 'miss', 'bypass')", name='ck_generation_runs_cache_status'),
    sa.CheckConstraint("retrieval_mode IS NULL OR retrieval_mode IN ('dense', 'lexical', 'hybrid')", name='ck_generation_runs_mode'),
    sa.CheckConstraint("abstained = (status = 'abstained') AND fallback_used = (status = 'fallback')", name='ck_generation_runs_flags_match_status'),
    sa.ForeignKeyConstraint(['index_version_id'], ['index_versions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_generation_runs_created_at', 'generation_runs', ['created_at'], unique=False)
    op.create_index('ix_generation_runs_status', 'generation_runs', ['status'], unique=False)
    op.create_index('ix_generation_runs_model_prompt', 'generation_runs', ['model', 'prompt_version'], unique=False)

    op.create_table('generation_citations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('generation_run_id', sa.UUID(), nullable=False),
    sa.Column('passage_id', sa.UUID(), nullable=False),
    sa.Column('label', sa.String(length=8), nullable=False),
    sa.Column('claim_ordinal', sa.Integer(), nullable=False),
    sa.Column('retrieval_rank', sa.Integer(), nullable=False),
    sa.Column('retrieval_score', sa.Numeric(precision=12, scale=8), nullable=False),
    sa.CheckConstraint('claim_ordinal >= 0', name='ck_generation_citations_claim_ordinal'),
    sa.CheckConstraint('retrieval_rank >= 1', name='ck_generation_citations_rank'),
    sa.ForeignKeyConstraint(['generation_run_id'], ['generation_runs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['passage_id'], ['document_passages.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('generation_run_id', 'claim_ordinal', 'label', name='uq_generation_citations_claim_label')
    )
    op.create_index('ix_generation_citations_run', 'generation_citations', ['generation_run_id'], unique=False)

    op.create_table('generation_feedback',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('generation_run_id', sa.UUID(), nullable=False),
    sa.Column('rating', sa.String(length=16), nullable=False),
    sa.Column('reason_code', sa.String(length=32), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("rating IN ('helpful', 'not_helpful')", name='ck_generation_feedback_rating'),
    sa.CheckConstraint("reason_code IS NULL OR reason_code IN ('inaccurate', 'missing_citation', 'unclear', 'other')", name='ck_generation_feedback_reason'),
    sa.ForeignKeyConstraint(['generation_run_id'], ['generation_runs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('generation_run_id', name='uq_generation_feedback_run')
    )

    op.create_table('llm_usage_days',
    sa.Column('usage_date', sa.Date(), nullable=False),
    sa.Column('reserved_tokens', sa.BigInteger(), nullable=False),
    sa.Column('used_tokens', sa.BigInteger(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('reserved_tokens >= 0 AND used_tokens >= 0', name='ck_llm_usage_days_nonneg'),
    sa.PrimaryKeyConstraint('usage_date')
    )


def downgrade() -> None:
    op.drop_table('llm_usage_days')
    op.drop_table('generation_feedback')
    op.drop_index('ix_generation_citations_run', table_name='generation_citations')
    op.drop_table('generation_citations')
    op.drop_index('ix_generation_runs_model_prompt', table_name='generation_runs')
    op.drop_index('ix_generation_runs_status', table_name='generation_runs')
    op.drop_index('ix_generation_runs_created_at', table_name='generation_runs')
    op.drop_table('generation_runs')
    op.drop_index('ix_document_passages_content_sha256', table_name='document_passages')
    op.drop_index('ix_document_passages_version_doc', table_name='document_passages')
    op.drop_table('document_passages')
