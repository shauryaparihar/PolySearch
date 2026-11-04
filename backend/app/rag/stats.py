"""Aggregate answer-mode numbers for the admin report: counts, latencies, tokens, budget.

Only counts and numbers: no question, passage or answer text exists in these tables by default.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.budget import snapshot


def _counts(session: Session, sql: str, since) -> dict[str, int]:
    return {str(k): int(n) for k, n in session.execute(text(sql), {"since": since})}


def build_answer_report(session: Session, days: int, settings, factory) -> dict:
    since = datetime.now(UTC) - timedelta(days=days)
    total, hits, median, p95, in_tok, out_tok, cost = session.execute(
        text(
            """
            select count(*),
                   count(*) filter (where cache_status = 'hit'),
                   percentile_cont(0.5) within group (order by total_ms),
                   percentile_cont(0.95) within group (order by total_ms),
                   coalesce(sum(input_tokens), 0), coalesce(sum(output_tokens), 0),
                   sum(estimated_cost)
            from generation_runs where created_at >= :since
            """
        ),
        {"since": since},
    ).one()
    feedback = _counts(
        session,
        "select f.rating, count(*) from generation_feedback f "
        "join generation_runs r on r.id = f.generation_run_id "
        "where r.created_at >= :since group by 1 order by 1",
        since,
    )
    budget = snapshot(factory)
    budget["daily_budget_tokens"] = settings.rag_daily_token_budget
    return {
        "window_days": days,
        "total": int(total),
        "by_status": _counts(
            session,
            "select status, count(*) from generation_runs where created_at >= :since "
            "group by 1 order by 1",
            since,
        ),
        "fallback_reasons": _counts(
            session,
            "select fallback_reason, count(*) from generation_runs "
            "where created_at >= :since and fallback_reason is not null group by 1 order by 1",
            since,
        ),
        "abstention_reasons": _counts(
            session,
            "select abstention_reason, count(*) from generation_runs "
            "where created_at >= :since and abstention_reason is not null group by 1 order by 1",
            since,
        ),
        "cache_hit_rate": round(int(hits) / int(total), 4) if total else None,
        "latency_ms": {
            "median": round(float(median), 3) if median is not None else None,
            "p95": round(float(p95), 3) if p95 is not None else None,
        },
        "tokens": {"input": int(in_tok), "output": int(out_tok), "total": int(in_tok + out_tok)},
        "estimated_cost": float(cost) if cost is not None else None,
        "estimated_cost_note": "from configured prices; null when none are configured",
        "feedback": {
            "helpful": feedback.get("helpful", 0),
            "not_helpful": feedback.get("not_helpful", 0),
        },
        "budget_today": budget,
    }
