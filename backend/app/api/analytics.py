"""Operational and product analytics computed directly from PostgreSQL rows.

CTR definition (also in the docs and interview notes):

    CTR = distinct clicked results / result impressions

* An *impression* is a stored result that was actually returned to a client
  (``search_results.shown_at`` is set): the first page at search time, later pages when fetched.
* A *clicked result* counts once per (search, document), however many click events exist.

CTR is behavioural evidence, not proof of relevance: position bias (top results get clicked
more), accidental clicks, and queries that are answered by the snippet all distort it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.search.trie import looks_sensitive

SINCE = "r.created_at >= :since"


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _counts(session: Session, sql: str, **params) -> dict:
    return {str(k): int(v) for k, v in session.execute(text(sql), params).all()}


def build_report(
    session: Session,
    days: int,
    rate_limited: dict[str, int],
    store_query_text: bool,
    min_text_count: int,
    now: datetime | None = None,
) -> dict:
    since = (now or datetime.now(UTC)) - timedelta(days=days)
    p = {"since": since}

    total, ok, err, zero, hits, median, p95 = session.execute(
        text(
            """
            select count(*),
                   count(*) filter (where r.status = 'ok'),
                   count(*) filter (where r.status = 'error'),
                   count(*) filter (where r.status = 'ok' and r.result_pool_size = 0),
                   count(*) filter (where r.cache_hit),
                   percentile_cont(0.5) within group (order by r.total_ms),
                   percentile_cont(0.95) within group (order by r.total_ms)
            from search_runs r where r.created_at >= :since
            """
        ),
        p,
    ).one()

    per_day = [
        {"day": d, "count": int(n)}
        for d, n in session.execute(
            text(
                """
                select to_char(date_trunc('day', r.created_at at time zone 'UTC'), 'YYYY-MM-DD') d,
                       count(*) from search_runs r where r.created_at >= :since
                group by 1 order by 1
                """
            ),
            p,
        )
    ]

    distribution = {
        "query_language": _counts(
            session, "select r.query_language, count(*) from search_runs r "
            "where r.created_at >= :since group by 1 order by 1", **p),
        "mode": _counts(
            session, "select r.requested_mode, count(*) from search_runs r "
            "where r.created_at >= :since group by 1 order by 1", **p),
        "target_language": _counts(
            session, "select t, count(*) from search_runs r, "
            "jsonb_array_elements_text(r.target_languages) t "
            "where r.created_at >= :since group by 1 order by 1", **p),
        "source": _counts(
            session, "select t, count(*) from search_runs r, "
            "jsonb_array_elements_text(r.sources) t "
            "where r.created_at >= :since group by 1 order by 1", **p),
        "index_version": _counts(
            session, "select v.version, count(*) from search_runs r "
            "join index_versions v on v.id = r.index_version_id "
            "where r.created_at >= :since group by 1 order by 1", **p),
    }  # fmt: skip

    by_rank = [
        {"rank": int(rank), "impressions": int(imp), "clicked_results": int(clk),
         "ctr": _ratio(int(clk), int(imp))}
        for rank, imp, clk in session.execute(
            text(
                """
                select sr.rank, count(*),
                       count(*) filter (where exists (
                           select 1 from click_events c
                           where c.search_run_id = sr.search_run_id
                             and c.document_id = sr.document_id))
                from search_results sr join search_runs r on r.id = sr.search_run_id
                where r.created_at >= :since and sr.shown_at is not null
                group by sr.rank order by sr.rank
                """
            ),
            p,
        )
    ]  # fmt: skip
    impressions = sum(row["impressions"] for row in by_rank)
    clicked = sum(row["clicked_results"] for row in by_rank)
    click_events = session.execute(
        text(
            "select count(*) from click_events c join search_runs r on r.id = c.search_run_id "
            "where r.created_at >= :since"
        ),
        p,
    ).scalar_one()

    feedback = _counts(
        session,
        "select f.label, count(*) from relevance_feedback f join search_runs r "
        "on r.id = f.search_run_id where r.created_at >= :since group by 1 order by 1",
        **p,
    )

    top_hashes = [
        {"query_sha256_prefix": h[:12], "count": int(n)}
        for h, n in session.execute(
            text(
                "select r.query_sha256, count(*) c from search_runs r where r.created_at >= :since "
                "group by 1 order by c desc, 1 limit 10"
            ),
            p,
        )
    ]
    top_text = None
    if store_query_text:  # only when explicitly configured; never one-off or sensitive-looking
        rows = session.execute(
            text(
                "select r.analytics_query_text, count(*) c from search_runs r "
                "where r.created_at >= :since and r.analytics_query_text is not null "
                "group by 1 having count(*) >= :min order by c desc, 1 limit 10"
            ),
            {**p, "min": min_text_count},
        )
        top_text = [{"query": q, "count": int(n)} for q, n in rows if not looks_sensitive(q)]

    return {
        "window_days": days,
        "searches": {
            "total": int(total),
            "ok": int(ok),
            "error": int(err),
            "per_day": per_day,
            "cache_hit_rate": _ratio(int(hits), int(total)),
            "zero_result_rate": _ratio(int(zero), int(ok)),
        },
        "latency_ms": {
            "median": round(float(median), 3) if median is not None else None,
            "p95": round(float(p95), 3) if p95 is not None else None,
        },
        "rate_limited_since_process_start": dict(rate_limited),
        "distribution": distribution,
        "engagement": {
            "impressions": impressions,
            "clicked_results": clicked,
            "ctr": _ratio(clicked, impressions),
            "ctr_by_rank": by_rank,
            "click_events": int(click_events),
            "ctr_definition": "distinct clicked results / result impressions",
        },
        "feedback": {
            "relevant": feedback.get("relevant", 0),
            "not_relevant": feedback.get("not_relevant", 0),
        },
        "top_queries": {"by_hash": top_hashes, "text": top_text},
    }
