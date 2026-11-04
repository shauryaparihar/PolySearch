"""add expiring token reservations

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa


revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('llm_reservations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('usage_date', sa.Date(), nullable=False),
    sa.Column('tokens', sa.BigInteger(), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('tokens > 0', name='ck_llm_reservations_positive'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_llm_reservations_day_expiry', 'llm_reservations', ['usage_date', 'expires_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_llm_reservations_day_expiry', table_name='llm_reservations')
    op.drop_table('llm_reservations')
