"""Tests for incremental sync logic in Index."""
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from mindmark.parser import Bookmark
from mindmark.index import Index, SyncResult, _content_hash


def _make_bookmark(url: str, title: str = "T", folder: str = "") -> Bookmark:
    return Bookmark(title=title, url=url, folder_path=folder, add_date=0, icon=None)


def _make_index(db_path: Path) -> Index:
    """Create an Index with a mock embedder to avoid loading the real model."""
    idx = Index(db_path=db_path)
    mock_embedder = MagicMock()
    dim = 4
    def fake_embed(texts):
        vecs = np.random.RandomState(42).randn(len(texts), dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
    mock_embedder.embed.side_effect = fake_embed
    mock_embedder.embed_one.side_effect = lambda t: fake_embed([t])[0]
    idx.embedder = mock_embedder
    return idx


@pytest.fixture
def idx(tmp_path):
    """Yield an Index with a mock embedder; close DB on teardown."""
    index = _make_index(tmp_path / "test.db")
    yield index
    index.close()


def test_sync_adds_new_bookmarks(idx):
    bms = [
        _make_bookmark("https://a.com", "A"),
        _make_bookmark("https://b.com", "B"),
    ]
    result = idx.sync(bms, source="chrome:Default")
    assert result.added == 2
    assert result.updated == 0
    assert result.removed == 0
    assert result.unchanged == 0
    assert not idx.is_empty()


def test_sync_unchanged_skips_embedding(idx):
    bms = [_make_bookmark("https://a.com", "A")]
    idx.sync(bms, source="test")

    # Reset call count
    idx.embedder.embed.reset_mock()

    # Sync again with same data
    result = idx.sync(bms, source="test")
    assert result.added == 0
    assert result.unchanged == 1
    # embed should NOT be called for unchanged bookmarks
    idx.embedder.embed.assert_not_called()


def test_sync_updates_changed_bookmarks(idx):
    bms = [_make_bookmark("https://a.com", "A", "Folder1")]
    idx.sync(bms, source="test")

    # Change the title
    bms2 = [_make_bookmark("https://a.com", "A Updated", "Folder1")]
    result = idx.sync(bms2, source="test")
    assert result.updated == 1
    assert result.added == 0
    assert result.unchanged == 0

    # Verify the title was updated in the DB
    cur = idx.con.cursor()
    cur.execute("SELECT title FROM bookmarks WHERE url = ?", ("https://a.com",))
    assert cur.fetchone()[0] == "A Updated"


def test_sync_removes_deleted_bookmarks(idx):
    bms = [
        _make_bookmark("https://a.com", "A"),
        _make_bookmark("https://b.com", "B"),
    ]
    idx.sync(bms, source="test")

    # Remove one bookmark
    bms2 = [_make_bookmark("https://a.com", "A")]
    result = idx.sync(bms2, source="test")
    assert result.removed == 1
    assert result.unchanged == 1

    # Verify b.com is gone
    cur = idx.con.cursor()
    cur.execute("SELECT COUNT(*) FROM bookmarks WHERE url = ?", ("https://b.com",))
    assert cur.fetchone()[0] == 0


def test_multi_source_no_cross_deletion(idx):
    """Syncing source A should not delete bookmarks from source B."""
    # Source A adds url X
    bms_a = [_make_bookmark("https://shared.com", "Shared")]
    idx.sync(bms_a, source="chrome:Default")

    # Source B also adds url X
    bms_b = [_make_bookmark("https://shared.com", "Shared")]
    idx.sync(bms_b, source="firefox:default")

    # Source A removes url X
    result = idx.sync([], source="chrome:Default")
    assert result.removed == 1  # removed from source A

    # But the bookmark should still exist (source B still references it)
    cur = idx.con.cursor()
    cur.execute("SELECT COUNT(*) FROM bookmarks WHERE url = ?", ("https://shared.com",))
    assert cur.fetchone()[0] == 1

    # Now remove from source B too
    result = idx.sync([], source="firefox:default")
    cur.execute("SELECT COUNT(*) FROM bookmarks WHERE url = ?", ("https://shared.com",))
    assert cur.fetchone()[0] == 0  # now truly gone


def test_sync_result_str():
    r = SyncResult(added=3, updated=1, removed=2, unchanged=10)
    s = str(r)
    assert "3 new" in s
    assert "1 updated" in s
    assert "2 removed" in s


def test_content_hash_deterministic():
    b = _make_bookmark("https://a.com", "A", "Work")
    h1 = _content_hash(b)
    h2 = _content_hash(b)
    assert h1 == h2
    assert len(h1) == 16  # truncated sha256


def test_content_hash_changes_on_title_change():
    b1 = _make_bookmark("https://a.com", "A", "Work")
    b2 = _make_bookmark("https://a.com", "B", "Work")
    assert _content_hash(b1) != _content_hash(b2)


def test_schema_migration_on_old_db(tmp_path):
    """Ensure opening a v1 database migrates cleanly."""
    db_path = tmp_path / "old.db"
    # Create a v1 database (no content_hash, no bookmark_sources)
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            folder_path TEXT NOT NULL,
            domain TEXT NOT NULL,
            add_date INTEGER NOT NULL,
            icon TEXT,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL
        );
    """)
    con.close()

    # Opening with Index should trigger migration
    idx = Index(db_path=db_path)
    try:
        cur = idx.con.cursor()
        cols = {r[1] for r in cur.execute("PRAGMA table_info(bookmarks)")}
        assert "content_hash" in cols

        tables = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "bookmark_sources" in tables
        assert "bookmark_enrichment" in tables

        cur.execute("SELECT value FROM meta WHERE key='schema_version'")
        assert cur.fetchone()[0] == "3"
    finally:
        idx.close()


def test_schema_v2_db_migrates_to_v3(tmp_path):
    """Ensure opening a v2 database adds enrichment table and bumps version."""
    db_path = tmp_path / "v2.db"
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            folder_path TEXT NOT NULL,
            domain TEXT NOT NULL,
            add_date INTEGER NOT NULL,
            icon TEXT,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            content_hash TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE bookmark_sources (
            url TEXT NOT NULL,
            source TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (url, source)
        );
        INSERT INTO meta(key, value) VALUES ('schema_version', '2');
    """)
    con.close()

    idx = Index(db_path=db_path)
    try:
        cur = idx.con.cursor()
        tables = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "bookmark_enrichment" in tables
        cur.execute("SELECT value FROM meta WHERE key='schema_version'")
        assert cur.fetchone()[0] == "3"
    finally:
        idx.close()


# ---- rebuild() tests ----

def test_rebuild_populates_content_hash(idx):
    """rebuild() must set content_hash so sync() can do incremental diffs."""
    bms = [_make_bookmark("https://a.com", "A", "Work")]
    idx.rebuild(bms)

    cur = idx.con.cursor()
    cur.execute("SELECT content_hash FROM bookmarks WHERE url = ?", ("https://a.com",))
    h = cur.fetchone()[0]
    assert h and len(h) == 16  # non-empty, truncated sha256


def test_rebuild_populates_bookmark_sources(idx):
    """rebuild() must populate bookmark_sources with source='html'."""
    bms = [
        _make_bookmark("https://a.com", "A"),
        _make_bookmark("https://b.com", "B"),
    ]
    idx.rebuild(bms)

    cur = idx.con.cursor()
    cur.execute("SELECT url, source FROM bookmark_sources ORDER BY url")
    rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0] == ("https://a.com", "html")
    assert rows[1] == ("https://b.com", "html")


def test_rebuild_marks_enrichment_pending(idx):
    bms = [
        _make_bookmark("https://a.com", "A"),
        _make_bookmark("https://b.com", "B"),
    ]
    idx.rebuild(bms)

    cur = idx.con.cursor()
    cur.execute("SELECT url, status FROM bookmark_enrichment ORDER BY url")
    rows = cur.fetchall()
    assert rows == [
        ("https://a.com", "pending"),
        ("https://b.com", "pending"),
    ]


def test_rebuild_clears_previous_data(idx):
    """rebuild() should clear old bookmarks and sources before inserting."""
    idx.rebuild([_make_bookmark("https://old.com", "Old")])
    idx.rebuild([_make_bookmark("https://new.com", "New")])

    cur = idx.con.cursor()
    cur.execute("SELECT COUNT(*) FROM bookmarks")
    assert cur.fetchone()[0] == 1
    cur.execute("SELECT url FROM bookmarks")
    assert cur.fetchone()[0] == "https://new.com"
    cur.execute("SELECT COUNT(*) FROM bookmark_sources")
    assert cur.fetchone()[0] == 1
    cur.execute("SELECT COUNT(*) FROM bookmark_enrichment")
    assert cur.fetchone()[0] == 1


def test_rebuild_empty_list(idx):
    result = idx.rebuild([])
    assert result["indexed"] == 0
    assert idx.is_empty()


def test_rebuild_then_sync_detects_unchanged(idx):
    """rebuild() followed by sync() with the same data should show all unchanged."""
    bms = [_make_bookmark("https://a.com", "A", "Work")]
    idx.rebuild(bms)

    idx.embedder.embed.reset_mock()
    result = idx.sync(bms, source="html")
    assert result.unchanged == 1
    assert result.added == 0
    idx.embedder.embed.assert_not_called()


# ---- stats() tests ----

def test_stats_on_populated_index(idx):
    bms = [
        _make_bookmark("https://github.com/a", "Repo A", "Work"),
        _make_bookmark("https://github.com/b", "Repo B", "Work"),
        _make_bookmark("https://docs.python.org", "Python Docs", "Ref"),
    ]
    idx.rebuild(bms)

    s = idx.stats()
    assert s["total"] == 3
    assert s["model"] is not None
    assert str(idx.db_path) in s["db_path"]
    # github.com should be top domain with count 2
    domains = dict(s["top_domains"])
    assert domains.get("github.com") == 2


def test_stats_on_empty_index(idx):
    s = idx.stats()
    assert s["total"] == 0


# ---- search() tests ----

def test_search_returns_results(idx):
    bms = [
        _make_bookmark("https://a.com", "Alpha"),
        _make_bookmark("https://b.com", "Beta"),
    ]
    idx.rebuild(bms)

    results = idx.search("anything", k=10)
    assert len(results) == 2
    assert all("score" in r for r in results)
    assert all("url" in r for r in results)


def test_search_empty_index(idx):
    results = idx.search("test")
    assert results == []


def test_search_domain_filter(idx):
    bms = [
        _make_bookmark("https://github.com/x", "GitHub"),
        _make_bookmark("https://docs.python.org", "Docs"),
    ]
    idx.rebuild(bms)

    results = idx.search("test", domain="github.com")
    assert all("github.com" in r["domain"] for r in results)


def test_search_folder_filter(idx):
    bms = [
        _make_bookmark("https://a.com", "A", "Work/Internal"),
        _make_bookmark("https://b.com", "B", "Personal"),
    ]
    idx.rebuild(bms)

    results = idx.search("test", folder="work")
    assert all("work" in r["folder_path"].lower() for r in results)


def test_search_k_limit(idx):
    bms = [_make_bookmark(f"https://{i}.com", f"Site {i}") for i in range(20)]
    idx.rebuild(bms)

    results = idx.search("test", k=5)
    assert len(results) == 5


def test_sync_adds_enrichment_pending_for_changed_urls(idx):
    bms = [
        _make_bookmark("https://a.com", "A"),
        _make_bookmark("https://b.com", "B"),
    ]
    idx.sync(bms, source="test")

    cur = idx.con.cursor()
    cur.execute("SELECT url, status FROM bookmark_enrichment ORDER BY url")
    rows = cur.fetchall()
    assert rows == [
        ("https://a.com", "pending"),
        ("https://b.com", "pending"),
    ]


def test_sync_removes_enrichment_when_bookmark_orphaned(idx):
    bms = [_make_bookmark("https://a.com", "A")]
    idx.sync(bms, source="chrome:Default")
    idx.sync([], source="chrome:Default")

    cur = idx.con.cursor()
    cur.execute("SELECT COUNT(*) FROM bookmark_enrichment WHERE url = ?", ("https://a.com",))
    assert cur.fetchone()[0] == 0


def test_remove_urls_also_clears_enrichment(idx):
    bms = [_make_bookmark("https://a.com", "A")]
    idx.rebuild(bms)

    removed = idx.remove_urls(["https://a.com"])
    assert removed == 1

    cur = idx.con.cursor()
    cur.execute("SELECT COUNT(*) FROM bookmark_enrichment WHERE url = ?", ("https://a.com",))
    assert cur.fetchone()[0] == 0


# ---- _remove_source() tests ----

def test_remove_source_cleans_orphans(idx):
    bms = [_make_bookmark("https://a.com", "A")]
    idx.sync(bms, source="chrome:Default")

    removed = idx._remove_source("chrome:Default")
    assert len(removed) == 1
    assert idx.is_empty()


def test_remove_source_preserves_other_sources(idx):
    idx.sync([_make_bookmark("https://a.com", "A")], source="chrome:Default")
    idx.sync([_make_bookmark("https://a.com", "A")], source="firefox:default")

    idx._remove_source("chrome:Default")
    assert not idx.is_empty()  # firefox still references it


# ---- Phase 4: Search Fusion with Summary Blending ----

class TestSearchFusion:
    def test_search_with_excerpt(self, idx):
        """include_excerpt flag adds relevant_excerpt to results."""
        bms = [_make_bookmark("https://a.com", "A")]
        idx.sync(bms, source="test")

        # Manually add a complete enrichment
        summary_text = "This is a test summary of the page content."
        test_vec = np.ones(4, dtype=np.float32) / 2.0
        idx.save_enrichment(
            url="https://a.com",
            summary_text=summary_text,
            summary_embedding=test_vec,
            model_name="test-model",
            content_hash="testhash",
            http_status=200,
            fetched_at=123,
            summarized_at=124,
        )

        results = idx.search("a", include_excerpt=True)
        assert len(results) >= 1
        assert "relevant_excerpt" in results[0]

    def test_search_without_excerpt_flag_omits_it(self, idx):
        """include_excerpt=False does not add relevant_excerpt."""
        bms = [_make_bookmark("https://a.com", "A")]
        idx.sync(bms, source="test")

        summary_text = "This is a test summary."
        test_vec = np.ones(4, dtype=np.float32) / 2.0
        idx.save_enrichment(
            url="https://a.com",
            summary_text=summary_text,
            summary_embedding=test_vec,
            model_name="test-model",
            content_hash="testhash",
            http_status=200,
            fetched_at=123,
            summarized_at=124,
        )

        results = idx.search("a", include_excerpt=False)
        assert len(results) >= 1
        assert "relevant_excerpt" not in results[0]

    def test_search_with_excerpt_generates_sentence(self, idx):
        """Excerpt extraction finds most relevant sentence."""
        bms = [_make_bookmark("https://a.com", "A")]
        idx.sync(bms, source="test")

        # Long summary
        summary_text = "x" * 200
        test_vec = np.ones(4, dtype=np.float32) / 2.0
        idx.save_enrichment(
            url="https://a.com",
            summary_text=summary_text,
            summary_embedding=test_vec,
            model_name="test-model",
            content_hash="testhash",
            http_status=200,
            fetched_at=123,
            summarized_at=124,
        )

        results = idx.search("a", include_excerpt=True)
        assert len(results) >= 1
        # Should have relevant_excerpt (from sentence extraction)
        assert "relevant_excerpt" in results[0]

    def test_blended_score_computation(self, idx):
        """Verify blended score = 0.65*base + 0.35*summary."""
        bms = [_make_bookmark("https://example.com", "Example")]
        idx.sync(bms, source="test")

        # Add a summary embedding
        summary_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        summary_vec /= np.linalg.norm(summary_vec)
        idx.save_enrichment(
            url="https://example.com",
            summary_text="test summary",
            summary_embedding=summary_vec,
            model_name="test-model",
            content_hash="testhash",
            http_status=200,
            fetched_at=123,
            summarized_at=124,
        )

        # Search and get score
        results = idx.search("example", k=1)
        assert len(results) == 1
        blended = results[0]["score"]
        assert -1 <= blended <= 1

    def test_search_respects_domain_filter_with_summaries(self, idx):
        """Domain filter works alongside summary blending."""
        bms = [
            _make_bookmark("https://a.com", "A"),
            _make_bookmark("https://b.com", "B"),
        ]
        idx.sync(bms, source="test")

        # Add summaries to both
        test_vec = np.ones(4, dtype=np.float32) / 2.0
        for url in ["https://a.com", "https://b.com"]:
            idx.save_enrichment(
                url=url,
                summary_text="summary",
                summary_embedding=test_vec,
                model_name="test-model",
                content_hash="hash",
                http_status=200,
                fetched_at=123,
                summarized_at=124,
            )

        # Filter by domain
        results = idx.search("", domain="b.com")
        assert all("b.com" in r["domain"] for r in results)

    def test_search_respects_folder_filter_with_summaries(self, idx):
        """Folder filter works alongside summary blending."""
        bms = [
            _make_bookmark("https://a.com", "A", folder="Folder1"),
            _make_bookmark("https://b.com", "B", folder="Folder2"),
        ]
        idx.sync(bms, source="test")

        # Add summaries
        test_vec = np.ones(4, dtype=np.float32) / 2.0
        for url in ["https://a.com", "https://b.com"]:
            idx.save_enrichment(
                url=url,
                summary_text="summary",
                summary_embedding=test_vec,
                model_name="test-model",
                content_hash="hash",
                http_status=200,
                fetched_at=123,
                summarized_at=124,
            )

        # Filter by folder
        results = idx.search("", folder="Folder1")
        assert len(results) >= 1
        assert "Folder1" in results[0]["folder_path"]

    def test_failed_enrichment_not_used_in_blend(self, idx):
        """Failed enrichment rows are skipped (only 'complete' used)."""
        bms = [_make_bookmark("https://a.com", "A")]
        idx.sync(bms, source="test")

        # Mark as failed
        idx.fail_enrichment("https://a.com", error="test error")

        # Search should still work, using only base embedding
        results = idx.search("a")
        assert len(results) >= 1
        # No excerpt should be included
        assert "relevant_excerpt" not in results[0]

    def test_search_with_relevant_excerpt(self, idx):
        """include_excerpt with relevant_excerpt shows most relevant sentence."""
        bms = [_make_bookmark("https://a.com", "A")]
        idx.sync(bms, source="test")

        summary_text = "First sentence here. Second sentence with details. Third sentence."
        test_vec = np.ones(4, dtype=np.float32) / 2.0
        idx.save_enrichment(
            url="https://a.com",
            summary_text=summary_text,
            summary_embedding=test_vec,
            model_name="test-model",
            content_hash="testhash",
            http_status=200,
            fetched_at=123,
            summarized_at=124,
        )

        results = idx.search("a", include_excerpt=True)
        assert len(results) >= 1
        # Should have relevant_excerpt with one of the sentences
        assert "relevant_excerpt" in results[0]
        excerpt = results[0]["relevant_excerpt"]
        assert excerpt in [
            "First sentence here.",
            "Second sentence with details.",
            "Third sentence.",
        ]


# ---- Phase 5: UX Improvements (Excerpt Extraction) ----

from mindmark.index import _find_relevant_excerpt


class TestRelevantExcerpt:
    def test_empty_summary_returns_empty(self):
        embedder_mock = MagicMock()
        query_vec = np.ones(4, dtype=np.float32)
        result = _find_relevant_excerpt(query_vec, "", embedder_mock)
        assert result == ""

    def test_whitespace_only_returns_empty(self):
        embedder_mock = MagicMock()
        query_vec = np.ones(4, dtype=np.float32)
        result = _find_relevant_excerpt(query_vec, "   \n\t  ", embedder_mock)
        assert result == ""

    def test_single_sentence_returned_as_is(self):
        embedder_mock = MagicMock()
        query_vec = np.ones(4, dtype=np.float32)
        text = "This is a single sentence."
        result = _find_relevant_excerpt(query_vec, text, embedder_mock)
        assert result == text

    def test_long_single_sentence_truncated(self):
        embedder_mock = MagicMock()
        query_vec = np.ones(4, dtype=np.float32)
        text = "x" * 300 + "."
        result = _find_relevant_excerpt(query_vec, text, embedder_mock)
        assert len(result) <= 200

    def test_multiple_sentences_picks_best_match(self):
        """With multiple sentences, should return the best-matching one."""
        embedder_mock = MagicMock()
        query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def fake_embed(texts):
            dim = 4
            vecs = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                seed = sum(ord(c) for c in text) % 100
                rng = np.random.RandomState(seed)
                v = rng.randn(dim).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                vecs[i] = v
            return vecs

        embedder_mock.embed.side_effect = fake_embed

        text = "First sentence here. Second sentence there. Third sentence too."
        result = _find_relevant_excerpt(query_vec, text, embedder_mock)

        # Should be one of the sentences
        assert result in [
            "First sentence here.",
            "Second sentence there.",
            "Third sentence too.",
        ]

    def test_no_sentence_markers_returns_first_150(self):
        """If no sentence markers and text is too short, return as-is (up to 200)."""
        embedder_mock = MagicMock()
        query_vec = np.ones(4, dtype=np.float32)
        text = "x" * 80  # No punctuation, shorter than 200
        result = _find_relevant_excerpt(query_vec, text, embedder_mock)
        # Unseparated text is treated as single sentence, returned up to 200 chars
        assert result == text
        assert len(result) == 80

    def test_long_unseparated_text_truncated_to_200(self):
        """Long text with no sentence markers gets treated as one sentence."""
        embedder_mock = MagicMock()
        query_vec = np.ones(4, dtype=np.float32)
        text = "x" * 300  # No punctuation, longer than 200
        result = _find_relevant_excerpt(query_vec, text, embedder_mock)
        # Unseparated text is single "sentence", truncated to 200 chars
        assert result == "x" * 200
        assert len(result) == 200

    def test_short_fragments_ignored(self):
        """Sentence fragments < 3 chars should be skipped."""
        embedder_mock = MagicMock()
        query_vec = np.ones(4, dtype=np.float32)

        def fake_embed(texts):
            dim = 4
            vecs = np.zeros((len(texts), dim), dtype=np.float32)
            for i in range(len(texts)):
                vecs[i] = np.ones(dim)
            return vecs

        embedder_mock.embed.side_effect = fake_embed

        text = "a. b. This is a real sentence."
        result = _find_relevant_excerpt(query_vec, text, embedder_mock)
        # Should return the real sentence, not the fragments
        assert "This is a real sentence" in result

    def test_embedding_error_fallback(self):
        """If embedding fails, fallback to first sentence."""

        class FailingEmbedder:
            def embed(self, texts):
                raise RuntimeError("Embedding failed")

        query_vec = np.ones(4, dtype=np.float32)
        text = "First sentence. Second sentence."
        result = _find_relevant_excerpt(query_vec, text, FailingEmbedder())
        # Should return first sentence
        assert result == "First sentence."

    def test_long_excerpt_truncated_with_ellipsis(self):
        """Excerpts longer than 200 chars should be truncated with ..."""
        embedder_mock = MagicMock()
        query_vec = np.ones(4, dtype=np.float32)

        def fake_embed(texts):
            dim = 4
            vecs = np.zeros((len(texts), dim), dtype=np.float32)
            # Make first sentence most similar
            vecs[0] = query_vec
            return vecs

        embedder_mock.embed.side_effect = fake_embed

        long_sentence = "A" * 250 + "."
        text = long_sentence + " Short."
        result = _find_relevant_excerpt(query_vec, text, embedder_mock)
        # Should end with ...
        assert result.endswith("...")
        assert len(result) <= 200
