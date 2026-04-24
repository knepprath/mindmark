"""Enrichment pipeline: fetch page content, extract text, embed, persist.

Summarization approach
----------------------
mindmark is 100% local with no cloud dependencies.  Rather than running a
generative LLM, we use *extractive* summarization: the first
``SUMMARY_CHARS`` characters of the extracted page text become the
"summary".  This text is then embedded with the same BGE/MiniLM ONNX
model already used for bookmark metadata, producing a summary vector that
captures the page's semantic content.

At search time (Phase 4) the summary embedding will be blended with the
bookmark metadata embedding to improve result relevance.
"""
from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .fetcher import extract_text, fetch_page, text_content_hash

if TYPE_CHECKING:
    from .index import Index

# Characters of extracted text used as the embedding corpus for each page.
# 500 chars fits a dense paragraph and keeps embedding latency negligible.
SUMMARY_CHARS = 500


@dataclass
class EnrichResult:
    url: str
    status: str          # 'complete' | 'failed' | 'skipped'
    error: str | None = None
    http_status: int | None = None


def _enrich_one(url: str, idx: "Index", timeout: float) -> EnrichResult:
    """Fetch, extract, embed, and persist enrichment for a single URL.

    Returns an :class:`EnrichResult` describing the outcome.  All
    exceptions are caught internally so a single failure never aborts a
    batch.
    """
    try:
        result = fetch_page(url, timeout=timeout)
    except Exception as exc:  # pragma: no cover — defensive
        idx.fail_enrichment(url, error=str(exc), fetched_at=int(time.time()))
        return EnrichResult(url=url, status="failed", error=str(exc))

    if not result.ok:
        idx.fail_enrichment(
            url,
            error=result.error or "unknown fetch error",
            http_status=result.http_status,
            fetched_at=result.fetched_at,
        )
        return EnrichResult(
            url=url, status="failed",
            error=result.error, http_status=result.http_status,
        )

    raw_text = extract_text(result.html or "", max_chars=SUMMARY_CHARS)

    if not raw_text.strip():
        idx.fail_enrichment(
            url,
            error="no extractable text",
            http_status=result.http_status,
            fetched_at=result.fetched_at,
        )
        return EnrichResult(
            url=url, status="failed",
            error="no extractable text", http_status=result.http_status,
        )

    content_hash = text_content_hash(raw_text)

    # Skip re-embedding if content hasn't changed since last enrichment
    try:
        cur = idx.con.cursor()
        cur.execute(
            "SELECT page_content_hash, status FROM bookmark_enrichment WHERE url=?",
            (url,),
        )
        row = cur.fetchone()
        if row and row[1] == "complete" and row[0] == content_hash:
            return EnrichResult(url=url, status="skipped")
    except Exception:
        pass  # if DB read fails, proceed to re-embed

    # Embed the extractive summary
    try:
        vec = idx.embedder.embed_one(raw_text)
    except Exception as exc:
        idx.fail_enrichment(
            url,
            error=f"embedding error: {exc}",
            http_status=result.http_status,
            fetched_at=result.fetched_at,
        )
        return EnrichResult(url=url, status="failed", error=f"embedding error: {exc}")

    summarized_at = int(time.time())
    idx.save_enrichment(
        url=url,
        summary_text=raw_text,
        summary_embedding=vec,
        model_name=idx.model_name,
        content_hash=content_hash,
        http_status=result.http_status,
        fetched_at=result.fetched_at,
        summarized_at=summarized_at,
    )

    return EnrichResult(url=url, status="complete", http_status=result.http_status)


@dataclass
class BatchEnrichResult:
    complete: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.complete + self.failed + self.skipped

    def __str__(self) -> str:
        return (
            f"complete={self.complete} failed={self.failed} skipped={self.skipped}"
        )


def enrich_pending(
    idx: "Index",
    limit: int | None = None,
    workers: int = 8,
    timeout: float = 10.0,
    refresh_failed: bool = False,
) -> BatchEnrichResult:
    """Process pending enrichment rows from the index.

    Parameters
    ----------
    idx:
        Open :class:`~mindmark.index.Index` instance.
    limit:
        Maximum number of pending URLs to process.  ``None`` means all.
    workers:
        Number of parallel fetch threads.
    timeout:
        Per-request fetch timeout in seconds.
    refresh_failed:
        If ``True``, reset all ``failed`` rows to ``pending`` before
        processing so they are retried.
    """
    if refresh_failed:
        idx.reset_failed_enrichment()

    urls = idx.pending_enrichment_urls(limit=limit)
    batch = BatchEnrichResult()

    if not urls:
        return batch

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_enrich_one, url, idx, timeout): url for url in urls}
        for fut in concurrent.futures.as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:  # pragma: no cover — defensive
                batch.failed += 1
                continue
            if r.status == "complete":
                batch.complete += 1
            elif r.status == "skipped":
                batch.skipped += 1
            else:
                batch.failed += 1

    return batch
