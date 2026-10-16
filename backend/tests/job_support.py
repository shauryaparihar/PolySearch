from __future__ import annotations

import time

from api_support import test_settings  # noqa: F401  (re-exported for convenience)
from app.db.models import ContentSource
from app.ingestion import service
from app.ingestion.sources import NormalizedRecord
from app.ingestion.text import content_sha256
from sqlalchemy import select

KEY = "job-test-admin-key-123456"
HEAD = {"x-admin-key": KEY}


def add_document(factory, title="Nuevo Tema", language="en", external_id=None) -> str:
    """Insert one more active document so the corpus (and so the index) changes."""
    external_id = external_id or title.replace(" ", "_")
    summary = f"{title} es un tema nuevo de la enciclopedia. Texto para {title}."
    with factory() as session:
        source_id = session.scalar(select(ContentSource.id))
        service.upsert_document(
            session,
            source_id,
            NormalizedRecord(
                external_id=external_id,
                language=language,
                title=title,
                summary=summary,
                source_url=f"https://{language}.wikipedia.org/wiki/{external_id}",
                source_revision="7",
                source_updated_at=None,
                content_sha256=content_sha256(title, summary),
            ),
        )
        session.commit()
    return f"{language}:{external_id}"


def start_index_job(client) -> str:
    r = client.post("/api/v1/admin/index-jobs", json={}, headers=HEAD)
    assert r.status_code == 202, r.text
    assert r.headers["location"].endswith(r.json()["job_id"])
    return r.json()["job_id"]


def get_job(client, kind: str, job_id: str) -> dict:
    r = client.get(f"/api/v1/admin/{kind}-jobs/{job_id}", headers=HEAD)
    assert r.status_code == 200, r.text
    return r.json()


def wait_for(client, kind: str, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get_job(client, kind, job_id)
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def version_dirs(root) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("idx-"))
