"""Tests for the enrichment pipeline (enricher.py)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mindmark.enricher import (
    SUMMARY_CHARS,
    BatchEnrichResult,
    EnrichResult,
    _enrich_one,
    enrich_pending,
)
from mindmark.fetcher import FetchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_fetch_result(
    url: str = "https://example.com",
    html: str = "<p>Hello world content here.</p>",
    http_status: int = 200,
    error: str | None = None,
    content_type: str = "text/html",
) -> FetchResult:
    return FetchResult(
        url=url,
        html=html,
        http_status=http_status,
        content_type=content_type,
        error=error,
        fetched_at=int(time.time()),
    )


def _make_mock_idx(pending: list[str] | None = None) -> MagicMock:
    """Return a minimal mock Index."""
    idx = MagicMock()
    idx.model_name = "test-model"
    # Default pending queue
    idx.pending_enrichment_urls.return_value = pending or []
    # embedder.embed_one returns a small float32 vector
    idx.embedder.embed_one.side_effect = lambda text: np.ones(4, dtype=np.float32)
    # con.cursor() for the skip-check path
    cur = MagicMock()
    cur.fetchone.return_value = None  # no existing row by default
    idx.con.cursor.return_value = cur
    return idx


# ---------------------------------------------------------------------------
# _enrich_one — success path
# ---------------------------------------------------------------------------

class TestEnrichOne:
    def test_complete_status_on_success(self):
        idx = _make_mock_idx()

        with patch("mindmark.enricher.fetch_page", return_value=_fake_fetch_result()):
            result = _enrich_one("https://example.com", idx, timeout=5.0)

        assert result.status == "complete"
        assert result.url == "https://example.com"
        assert result.error is None
        idx.save_enrichment.assert_called_once()

    def test_save_enrichment_receives_correct_url(self):
        idx = _make_mock_idx()

        with patch("mindmark.enricher.fetch_page", return_value=_fake_fetch_result(url="https://a.com")):
            _enrich_one("https://a.com", idx, timeout=5.0)

        call_kwargs = idx.save_enrichment.call_args.kwargs
        assert call_kwargs["url"] == "https://a.com"
        assert call_kwargs["model_name"] == "test-model"

    def test_http_status_propagated_on_success(self):
        idx = _make_mock_idx()
        fetch = _fake_fetch_result(http_status=200)

        with patch("mindmark.enricher.fetch_page", return_value=fetch):
            result = _enrich_one("https://example.com", idx, timeout=5.0)

        assert result.http_status == 200

    # ---------------------------------------------------------------------------
    # _enrich_one — failure paths
    # ---------------------------------------------------------------------------

    def test_failed_status_on_http_error(self):
        idx = _make_mock_idx()
        bad = _fake_fetch_result(html=None, http_status=404, error="Not Found")
        # Simulate a non-ok result: FetchResult.ok is False when error is set
        bad = FetchResult(
            url="https://example.com",
            html=None,
            http_status=404,
            content_type="text/html",
            error="Not Found",
            fetched_at=int(time.time()),
        )

        with patch("mindmark.enricher.fetch_page", return_value=bad):
            result = _enrich_one("https://example.com", idx, timeout=5.0)

        assert result.status == "failed"
        assert result.http_status == 404
        idx.fail_enrichment.assert_called_once()

    def test_failed_status_on_empty_text(self):
        idx = _make_mock_idx()
        # Page with no visible text (only tags)
        empty_html = "<html><head><title>x</title></head><body></body></html>"

        with patch("mindmark.enricher.fetch_page", return_value=_fake_fetch_result(html=empty_html)):
            with patch("mindmark.enricher.extract_text", return_value="   "):
                result = _enrich_one("https://example.com", idx, timeout=5.0)

        assert result.status == "failed"
        assert "no extractable text" in (result.error or "")
        idx.fail_enrichment.assert_called_once()

    def test_failed_status_on_embedding_error(self):
        idx = _make_mock_idx()
        idx.embedder.embed_one.side_effect = RuntimeError("ONNX runtime error")

        with patch("mindmark.enricher.fetch_page", return_value=_fake_fetch_result()):
            result = _enrich_one("https://example.com", idx, timeout=5.0)

        assert result.status == "failed"
        assert "embedding error" in (result.error or "")

    # ---------------------------------------------------------------------------
    # _enrich_one — skip path
    # ---------------------------------------------------------------------------

    def test_skipped_when_content_hash_unchanged(self):
        idx = _make_mock_idx()
        html = "<p>Stable content</p>"

        # First call to extract_text returns a known string; we pre-compute its hash
        from mindmark.fetcher import extract_text, text_content_hash
        text = extract_text(html, max_chars=SUMMARY_CHARS)
        content_hash = text_content_hash(text)

        # Simulate existing DB row with matching hash and status=complete
        cur = MagicMock()
        cur.fetchone.return_value = (content_hash, "complete")
        idx.con.cursor.return_value = cur

        with patch("mindmark.enricher.fetch_page", return_value=_fake_fetch_result(html=html)):
            result = _enrich_one("https://example.com", idx, timeout=5.0)

        assert result.status == "skipped"
        idx.save_enrichment.assert_not_called()

    def test_not_skipped_when_hash_differs(self):
        idx = _make_mock_idx()

        # Row exists but hash is different → should re-embed
        cur = MagicMock()
        cur.fetchone.return_value = ("oldhash", "complete")
        idx.con.cursor.return_value = cur

        with patch("mindmark.enricher.fetch_page", return_value=_fake_fetch_result()):
            result = _enrich_one("https://example.com", idx, timeout=5.0)

        assert result.status == "complete"
        idx.save_enrichment.assert_called_once()


# ---------------------------------------------------------------------------
# enrich_pending — batch orchestration
# ---------------------------------------------------------------------------

class TestEnrichPending:
    def test_empty_pending_returns_zero_batch(self):
        idx = _make_mock_idx(pending=[])
        result = enrich_pending(idx, limit=None, workers=2, timeout=5.0)

        assert result.total == 0
        assert result.complete == 0
        assert result.failed == 0
        assert result.skipped == 0

    def test_batch_counts_mixed_results(self):
        urls = [
            "https://ok.com",
            "https://fail.com",
            "https://skip.com",
        ]
        idx = _make_mock_idx(pending=urls)

        def side_enrich(url, index, timeout):
            if "ok" in url:
                return EnrichResult(url=url, status="complete")
            if "fail" in url:
                return EnrichResult(url=url, status="failed", error="err")
            return EnrichResult(url=url, status="skipped")

        with patch("mindmark.enricher._enrich_one", side_effect=side_enrich):
            result = enrich_pending(idx, workers=2, timeout=5.0)

        assert result.complete == 1
        assert result.failed == 1
        assert result.skipped == 1
        assert result.total == 3

    def test_limit_passed_to_pending_urls(self):
        idx = _make_mock_idx(pending=[])
        enrich_pending(idx, limit=10, workers=2, timeout=5.0)
        idx.pending_enrichment_urls.assert_called_once_with(limit=10)

    def test_refresh_failed_calls_reset(self):
        idx = _make_mock_idx(pending=[])
        idx.reset_failed_enrichment.return_value = 3

        enrich_pending(idx, refresh_failed=True, workers=2, timeout=5.0)

        idx.reset_failed_enrichment.assert_called_once()

    def test_refresh_failed_false_does_not_reset(self):
        idx = _make_mock_idx(pending=[])
        enrich_pending(idx, refresh_failed=False, workers=2, timeout=5.0)
        idx.reset_failed_enrichment.assert_not_called()

    def test_str_representation(self):
        r = BatchEnrichResult(complete=3, failed=1, skipped=2)
        s = str(r)
        assert "complete=3" in s
        assert "failed=1" in s
        assert "skipped=2" in s

    def test_total_property(self):
        r = BatchEnrichResult(complete=2, failed=1, skipped=4)
        assert r.total == 7
