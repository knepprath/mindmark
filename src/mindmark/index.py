"""Embedding + SQLite-backed vector index for bookmarks."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .parser import Bookmark

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_SCHEMA_VERSION = 3


def default_db_path() -> Path:
    env = os.environ.get("MINDMARK_HOME")
    if env:
        base = Path(env)
    elif os.name == "nt":
        # On Windows, use %LOCALAPPDATA%\mindmark (dotfolders are unusual)
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) / "mindmark" if local else Path.home() / ".mindmark"
    else:
        base = Path.home() / ".mindmark"
    base.mkdir(parents=True, exist_ok=True)
    return base / "index.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bookmarks (
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
CREATE TABLE IF NOT EXISTS bookmark_sources (
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (url, source)
);
CREATE TABLE IF NOT EXISTS bookmark_enrichment (
    url TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    page_content_hash TEXT NOT NULL DEFAULT '',
    summary_text TEXT,
    summary_embedding BLOB,
    summary_dim INTEGER,
    summary_model TEXT,
    llm_model TEXT,
    fetched_at INTEGER,
    summarized_at INTEGER,
    error TEXT,
    http_status INTEGER
);
CREATE INDEX IF NOT EXISTS idx_bookmarks_domain ON bookmarks(domain);
CREATE INDEX IF NOT EXISTS idx_bookmarks_folder ON bookmarks(folder_path);
CREATE INDEX IF NOT EXISTS idx_enrichment_status
ON bookmark_enrichment(status);
CREATE INDEX IF NOT EXISTS idx_enrichment_summarized_at
ON bookmark_enrichment(summarized_at);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.executescript(_SCHEMA)
    _migrate(con)
    return con


def _migrate(con: sqlite3.Connection) -> None:
    """Run schema migrations for existing databases."""
    cur = con.cursor()
    cur.execute("SELECT value FROM meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    version = int(row[0]) if row else 1

    if version < 2:
        # Add content_hash column if missing (pre-v2 databases)
        cols = {r[1] for r in cur.execute("PRAGMA table_info(bookmarks)")}
        if "content_hash" not in cols:
            cur.execute(
                "ALTER TABLE bookmarks ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
            )
        # Create bookmark_sources table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bookmark_sources (
                url TEXT NOT NULL,
                source TEXT NOT NULL,
                content_hash TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (url, source)
            )
        """)
        cur.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            ("2",),
        )
        con.commit()

    if version < 3:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bookmark_enrichment (
                url TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                page_content_hash TEXT NOT NULL DEFAULT '',
                summary_text TEXT,
                summary_embedding BLOB,
                summary_dim INTEGER,
                summary_model TEXT,
                llm_model TEXT,
                fetched_at INTEGER,
                summarized_at INTEGER,
                error TEXT,
                http_status INTEGER
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_enrichment_status "
            "ON bookmark_enrichment(status)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_enrichment_summarized_at "
            "ON bookmark_enrichment(summarized_at)"
        )
        cur.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            ("3",),
        )
        con.commit()


def _vec_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def _blob_to_vec(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _find_relevant_excerpt(
    query_embedding: np.ndarray,
    summary_text: str,
    embedder: "object",
) -> str:
    """Find the most relevant sentence(s) in summary text for the query.

    Splits summary into sentences, embeds each, finds the most similar to
    the query embedding, and returns that sentence with surrounding context
    (max 200 chars total).

    Parameters
    ----------
    query_embedding : np.ndarray
        Normalized embedding of the search query.
    summary_text : str
        Full extracted summary text (up to 500 chars).
    embedder : object
        BGE/MiniLM embedder with embed_one() method.

    Returns
    -------
    str
        Most relevant sentence with surrounding context, or first 150 chars
        if sentence detection fails.
    """
    if not summary_text or not summary_text.strip():
        return ""

    # Split on sentence boundaries (., !, ?)
    sentences = []
    current = []
    for char in summary_text:
        current.append(char)
        if char in ".!?":
            s = "".join(current).strip()
            if s and len(s) > 3:  # Skip very short fragments
                sentences.append(s)
            current = []
    if current:
        s = "".join(current).strip()
        if s and len(s) > 3:
            sentences.append(s)

    if not sentences:
        # Fallback: return first 150 chars if no sentences found
        return summary_text[:150]

    # If only one sentence, return it (up to 200 chars)
    if len(sentences) == 1:
        return sentences[0][:200]

    # Embed each sentence and find most similar to query
    try:
        sentence_embeddings = embedder.embed(sentences)
        similarities = sentence_embeddings @ query_embedding
        best_idx = int(np.argmax(similarities))
    except Exception:
        # Fallback if embedding fails
        return sentences[0][:200]

    # Return best sentence + surrounding context (up to 200 chars)
    best_sentence = sentences[best_idx]
    if len(best_sentence) <= 200:
        return best_sentence
    # Truncate with ellipsis
    return best_sentence[:197] + "..."



@dataclass
class Embedder:
    model_name: str = DEFAULT_MODEL
    _model: object | None = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = np.asarray(list(model.embed(texts)), dtype=np.float32)
        return _l2_normalize(vecs)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def _content_hash(b: Bookmark) -> str:
    """Hash the fields that affect embedding text."""
    payload = f"{b.url}\0{b.title}\0{b.folder_path}\0{b.domain}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class SyncResult:
    """Result of an incremental sync operation."""
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    source: str = ""

    @property
    def total_changed(self) -> int:
        return self.added + self.updated + self.removed

    def __str__(self) -> str:
        return (
            f"{self.added} new, {self.updated} updated, "
            f"{self.removed} removed, {self.unchanged} unchanged"
        )


class Index:
    def __init__(self, db_path: Path | None = None, model_name: str = DEFAULT_MODEL):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.model_name = model_name
        self.con = _connect(self.db_path)
        self.embedder = Embedder(model_name=model_name)

    def close(self) -> None:
        """Close the underlying database connection."""
        self.con.close()

    def _queue_enrichment_pending(self, cur: sqlite3.Cursor, urls: set[str]) -> None:
        """Mark URLs for content-enrichment work."""
        if not urls:
            return
        cur.executemany(
            """
            INSERT INTO bookmark_enrichment (url, status)
            VALUES (?, 'pending')
            ON CONFLICT(url) DO UPDATE SET
                status='pending',
                error=NULL
            """,
            [(u,) for u in sorted(urls)],
        )

    def is_empty(self) -> bool:
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM bookmarks")
        return cur.fetchone()[0] == 0

    def _model_changed(self) -> bool:
        """Check if the stored model differs from the current one."""
        cur = self.con.cursor()
        cur.execute("SELECT value FROM meta WHERE key = 'model'")
        row = cur.fetchone()
        if row is None:
            return False  # no model stored yet
        return row[0] != self.model_name

    def sync(
        self,
        bookmarks: list[Bookmark],
        source: str = "html",
        batch_size: int = 64,
    ) -> SyncResult:
        """Incrementally sync bookmarks from a source.

        Only embeds new/changed bookmarks.  Bookmarks removed from *this*
        source are deleted from the index only if no other source
        references them.
        """
        result = SyncResult(source=source)

        if not bookmarks:
            # Delete all bookmarks from this source
            removed_urls = self._remove_source(source)
            result.removed = len(removed_urls)
            return result

        # If the embedding model changed, force full re-embed
        force_reembed = self._model_changed()

        # 1. Hash incoming bookmarks
        new_map: dict[str, tuple[Bookmark, str]] = {}
        for b in bookmarks:
            h = _content_hash(b)
            if b.url not in new_map:  # dedup by URL
                new_map[b.url] = (b, h)

        # 2. Load existing hashes for this source
        cur = self.con.cursor()
        cur.execute(
            "SELECT url, content_hash FROM bookmark_sources WHERE source = ?",
            (source,),
        )
        existing: dict[str, str] = {r[0]: r[1] for r in cur.fetchall()}

        # 3. Compute diff
        new_urls = set(new_map.keys())
        old_urls = set(existing.keys())

        to_add_urls = new_urls - old_urls
        to_delete_urls = old_urls - new_urls
        common_urls = new_urls & old_urls

        to_update_urls: set[str] = set()
        for url in common_urls:
            _, new_hash = new_map[url]
            if force_reembed or new_hash != existing[url]:
                to_update_urls.add(url)

        result.unchanged = len(common_urls) - len(to_update_urls)
        result.added = len(to_add_urls)
        result.updated = len(to_update_urls)

        # 4. Embed only what changed
        to_embed_urls = to_add_urls | to_update_urls
        embed_list = [new_map[u] for u in to_embed_urls]

        embedded: dict[str, tuple[Bookmark, str, bytes, int]] = {}
        for start in range(0, len(embed_list), batch_size):
            chunk = embed_list[start:start + batch_size]
            texts = [b.embedding_text() for b, _h in chunk]
            vecs = self.embedder.embed(texts)
            for (b, h), v in zip(chunk, vecs):
                embedded[b.url] = (b, h, _vec_to_blob(v), int(v.shape[0]))

        # 5. Apply all DB changes in a single transaction
        cur = self.con.cursor()
        try:
            # Update model metadata
            cur.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('model', ?)",
                (self.model_name,),
            )

            # Insert/update bookmarks and sources
            for url in to_add_urls:
                b, h, vec_blob, dim = embedded[url]
                # Check if URL exists from another source
                cur.execute("SELECT 1 FROM bookmarks WHERE url = ?", (url,))
                if cur.fetchone():
                    # URL already indexed by another source — update metadata
                    cur.execute(
                        "UPDATE bookmarks SET title=?, folder_path=?, domain=?, "
                        "add_date=?, icon=?, embedding=?, dim=?, content_hash=? "
                        "WHERE url=?",
                        (b.title, b.folder_path, b.domain, b.add_date,
                         b.icon, vec_blob, dim, h, url),
                    )
                else:
                    cur.execute(
                        "INSERT INTO bookmarks "
                        "(url, title, folder_path, domain, add_date, icon, "
                        "embedding, dim, content_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                        (url, b.title, b.folder_path, b.domain, b.add_date,
                         b.icon, vec_blob, dim, h),
                    )
                cur.execute(
                    "INSERT OR REPLACE INTO bookmark_sources (url, source, content_hash) "
                    "VALUES (?, ?, ?)",
                    (url, source, h),
                )

            for url in to_update_urls:
                b, h, vec_blob, dim = embedded[url]
                cur.execute(
                    "UPDATE bookmarks SET title=?, folder_path=?, domain=?, "
                    "add_date=?, icon=?, embedding=?, dim=?, content_hash=? "
                    "WHERE url=?",
                    (b.title, b.folder_path, b.domain, b.add_date,
                     b.icon, vec_blob, dim, h, url),
                )
                cur.execute(
                    "UPDATE bookmark_sources SET content_hash=? "
                    "WHERE url=? AND source=?",
                    (h, url, source),
                )

            self._queue_enrichment_pending(cur, to_embed_urls)

            # Delete bookmarks removed from this source
            for url in to_delete_urls:
                cur.execute(
                    "DELETE FROM bookmark_sources WHERE url=? AND source=?",
                    (url, source),
                )
                # Only delete from bookmarks if no other source references it
                cur.execute(
                    "SELECT COUNT(*) FROM bookmark_sources WHERE url=?", (url,)
                )
                if cur.fetchone()[0] == 0:
                    cur.execute("DELETE FROM bookmarks WHERE url=?", (url,))
                    cur.execute("DELETE FROM bookmark_enrichment WHERE url=?", (url,))

            result.removed = len(to_delete_urls)
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

        return result

    def _remove_source(self, source: str) -> list[str]:
        """Remove all bookmarks from a source, cleaning up orphans."""
        cur = self.con.cursor()
        cur.execute(
            "SELECT url FROM bookmark_sources WHERE source = ?", (source,)
        )
        urls = [r[0] for r in cur.fetchall()]
        try:
            for url in urls:
                cur.execute(
                    "DELETE FROM bookmark_sources WHERE url=? AND source=?",
                    (url, source),
                )
                cur.execute(
                    "SELECT COUNT(*) FROM bookmark_sources WHERE url=?", (url,)
                )
                if cur.fetchone()[0] == 0:
                    cur.execute("DELETE FROM bookmarks WHERE url=?", (url,))
                    cur.execute("DELETE FROM bookmark_enrichment WHERE url=?", (url,))
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        return urls

    def rebuild(self, bookmarks: list[Bookmark], batch_size: int = 64) -> dict:
        cur = self.con.cursor()
        cur.execute("DELETE FROM bookmarks")
        cur.execute("DELETE FROM bookmark_sources")
        cur.execute("DELETE FROM bookmark_enrichment")
        cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('model', ?)", (self.model_name,))
        self.con.commit()

        total = len(bookmarks)
        if total == 0:
            return {"indexed": 0, "model": self.model_name, "dim": 0}

        rows = []
        source_rows = []
        for start in range(0, total, batch_size):
            chunk = bookmarks[start:start + batch_size]
            texts = [b.embedding_text() for b in chunk]
            vecs = self.embedder.embed(texts)
            for b, v in zip(chunk, vecs):
                h = _content_hash(b)
                rows.append((
                    b.url, b.title, b.folder_path, b.domain,
                    b.add_date, b.icon, _vec_to_blob(v), int(v.shape[0]), h,
                ))
                source_rows.append((b.url, "html", h))

        cur.executemany(
            "INSERT OR REPLACE INTO bookmarks "
            "(url, title, folder_path, domain, add_date, icon, embedding, dim, content_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        cur.executemany(
            "INSERT OR REPLACE INTO bookmark_sources (url, source, content_hash) "
            "VALUES (?,?,?)",
            source_rows,
        )
        self._queue_enrichment_pending(cur, {r[0] for r in rows})
        self.con.commit()
        return {"indexed": total, "model": self.model_name, "dim": rows[0][-2]}

    def stats(self) -> dict:
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM bookmarks")
        total = cur.fetchone()[0]
        cur.execute("SELECT value FROM meta WHERE key='model'")
        m = cur.fetchone()
        cur.execute(
            "SELECT domain, COUNT(*) c FROM bookmarks GROUP BY domain ORDER BY c DESC LIMIT 10"
        )
        top_domains = cur.fetchall()
        cur.execute(
            "SELECT folder_path, COUNT(*) c FROM bookmarks "
            "WHERE folder_path <> '' GROUP BY folder_path ORDER BY c DESC LIMIT 10"
        )
        top_folders = cur.fetchall()
        return {
            "db_path": str(self.db_path),
            "total": total,
            "model": m[0] if m else None,
            "top_domains": top_domains,
            "top_folders": top_folders,
        }

    def all_bookmarks(self) -> list[dict]:
        """Return all indexed bookmarks needed for validation/reporting."""
        cur = self.con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute(
            "SELECT url, title, folder_path, domain FROM bookmarks ORDER BY url"
        )
        rows = cur.fetchall()
        return [
            {
                "url": r["url"],
                "title": r["title"],
                "folder_path": r["folder_path"],
                "domain": r["domain"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Enrichment helpers
    # ------------------------------------------------------------------

    def pending_enrichment_urls(self, limit: int | None = None) -> list[str]:
        """Return URLs whose enrichment status is 'pending'."""
        cur = self.con.cursor()
        if limit is not None and limit > 0:
            cur.execute(
                "SELECT url FROM bookmark_enrichment WHERE status='pending' LIMIT ?",
                (limit,),
            )
        else:
            cur.execute("SELECT url FROM bookmark_enrichment WHERE status='pending'")
        return [r[0] for r in cur.fetchall()]

    def reset_failed_enrichment(self) -> int:
        """Mark all 'failed' enrichment rows back to 'pending' for retry."""
        cur = self.con.cursor()
        try:
            cur.execute(
                "UPDATE bookmark_enrichment SET status='pending', error=NULL "
                "WHERE status='failed'"
            )
            count = cur.rowcount
            self.con.commit()
            return count
        except Exception:
            self.con.rollback()
            raise

    def save_enrichment(
        self,
        url: str,
        summary_text: str,
        summary_embedding: "np.ndarray",
        model_name: str,
        content_hash: str,
        http_status: int | None,
        summarized_at: int,
        fetched_at: int,
    ) -> None:
        """Persist a successful enrichment result."""
        vec_blob = _vec_to_blob(summary_embedding)
        dim = int(summary_embedding.shape[0])
        cur = self.con.cursor()
        try:
            cur.execute(
                """
                UPDATE bookmark_enrichment SET
                    status='complete',
                    page_content_hash=?,
                    summary_text=?,
                    summary_embedding=?,
                    summary_dim=?,
                    summary_model=?,
                    http_status=?,
                    fetched_at=?,
                    summarized_at=?,
                    error=NULL
                WHERE url=?
                """,
                (
                    content_hash, summary_text, vec_blob, dim,
                    model_name, http_status, fetched_at, summarized_at,
                    url,
                ),
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def fail_enrichment(
        self,
        url: str,
        error: str,
        http_status: int | None = None,
        fetched_at: int | None = None,
    ) -> None:
        """Record a fetch/extraction failure for a URL."""
        cur = self.con.cursor()
        try:
            cur.execute(
                """
                UPDATE bookmark_enrichment SET
                    status='failed',
                    error=?,
                    http_status=?,
                    fetched_at=?
                WHERE url=?
                """,
                (error, http_status, fetched_at, url),
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def enrichment_stats(self) -> dict:
        """Return a summary of enrichment table status counts."""
        cur = self.con.cursor()
        cur.execute(
            "SELECT status, COUNT(*) FROM bookmark_enrichment GROUP BY status"
        )
        return dict(cur.fetchall())

    def remove_urls(self, urls: list[str]) -> int:
        """Delete bookmarks and source mappings for the given URLs."""
        if not urls:
            return 0

        unique_urls = sorted(set(urls))
        cur = self.con.cursor()
        placeholders = ",".join("?" for _ in unique_urls)
        try:
            cur.execute(
                f"DELETE FROM bookmark_sources WHERE url IN ({placeholders})",
                unique_urls,
            )
            cur.execute(
                f"DELETE FROM bookmarks WHERE url IN ({placeholders})",
                unique_urls,
            )
            removed = cur.rowcount
            cur.execute(
                f"DELETE FROM bookmark_enrichment WHERE url IN ({placeholders})",
                unique_urls,
            )
            self.con.commit()
            return removed
        except Exception:
            self.con.rollback()
            raise

    def _load_matrix(self):
        cur = self.con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute(
            "SELECT id, url, title, folder_path, domain, add_date, icon, embedding, dim "
            "FROM bookmarks"
        )
        rows = cur.fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []
        dim = rows[0]["dim"]
        mat = np.empty((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = _blob_to_vec(r["embedding"], dim)
        return mat, rows

    def _load_enrichments(self) -> dict:
        """Load all complete enrichments as url -> (embedding, text, dim)."""
        enrichments = {}
        cur = self.con.cursor()
        cur.execute(
            "SELECT url, summary_embedding, summary_text, summary_dim "
            "FROM bookmark_enrichment WHERE status='complete' AND summary_embedding IS NOT NULL"
        )
        for url, embedding_blob, summary_text, dim in cur.fetchall():
            if embedding_blob and dim:
                vec = _blob_to_vec(embedding_blob, dim)
                enrichments[url] = (vec, summary_text, dim)
        return enrichments

    def search(self, query: str, k: int = 10, domain: str | None = None,
               folder: str | None = None, include_excerpt: bool = False) -> list[dict]:
        """Search bookmarks by query, with optional summary blending.

        Blends base bookmark embedding similarity with summary embedding
        similarity (when available) using weights: 0.65 * base + 0.35 * summary.

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Maximum number of results.
        domain : str, optional
            Filter by domain substring.
        folder : str, optional
            Filter by folder substring.
        include_excerpt : bool
            If True, include relevant excerpt from summary_text in results.

        Returns
        -------
        list[dict]
            List of results with keys: url, title, folder_path, domain,
            score (blended if summary available). If include_excerpt=True,
            also includes 'relevant_excerpt' key.
        """
        mat, rows = self._load_matrix()
        if len(rows) == 0:
            return []

        q = self.embedder.embed_one(query)
        base_sims = mat @ q

        # Load complete enrichments for summary blending
        enrichments = self._load_enrichments()

        # Compute blended scores for each bookmark
        blended_sims = np.array(base_sims, copy=True)
        for idx, r in enumerate(rows):
            url = r["url"]
            if url in enrichments:
                summary_embedding, summary_text, summary_dim = enrichments[url]
                summary_sim = float(np.dot(summary_embedding, q))
                # Blend: 0.65 * base + 0.35 * summary
                blended_sims[idx] = 0.65 * base_sims[idx] + 0.35 * summary_sim

        order = np.argsort(-blended_sims)
        results: list[dict] = []
        for idx in order:
            r = rows[int(idx)]
            if domain and domain.lower() not in r["domain"]:
                continue
            if folder and folder.lower() not in r["folder_path"].lower():
                continue
            result = {
                "score": float(blended_sims[int(idx)]),
                "title": r["title"],
                "url": r["url"],
                "folder_path": r["folder_path"],
                "domain": r["domain"],
            }
            if include_excerpt and r["url"] in enrichments:
                summary_text = enrichments[r["url"]][1]
                # Find and include the most relevant excerpt for the query
                if summary_text:
                    result["relevant_excerpt"] = _find_relevant_excerpt(q, summary_text, self.embedder)
            results.append(result)
            if len(results) >= k:
                break
        return results
