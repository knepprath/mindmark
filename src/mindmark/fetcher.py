"""Fetch and extract text content from bookmark URLs.

Provides two public entry points:

* ``fetch_page(url, ...)`` — Downloads a URL and returns a :class:`FetchResult`.
  Uses stdlib only (no extra dependencies).

* ``extract_text(html)`` — Strips boilerplate (scripts, nav, footer, etc.) from
  raw HTML and returns normalised plain text suitable for summarisation.

* ``text_content_hash(text)`` — Short deterministic hash for change detection.
"""
from __future__ import annotations

import hashlib
import html as _html_lib
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# Tags whose subtree should be completely ignored (no text collected)
_SKIP_TAGS: frozenset[str] = frozenset({
    "script", "style", "noscript", "template",
    "nav", "header", "footer", "aside",
    "form", "button", "select", "textarea",
    "menu", "dialog", "figure", "figcaption",
    "iframe", "object", "embed", "svg", "canvas",
    "head",
})

# Tags that act as block separators (emit a space boundary around their text)
_BLOCK_TAGS: frozenset[str] = frozenset({
    "p", "div", "section", "article", "main",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "dt", "dd", "blockquote", "pre", "code",
    "td", "th", "tr", "caption",
    "br", "hr",
})

_MAX_FETCH_BYTES = 512 * 1024   # 512 KB hard cap
_MAX_TEXT_CHARS  = 4_000        # characters returned by extract_text
_DEFAULT_TIMEOUT = 10.0         # seconds per request
_USER_AGENT      = "mindmark/0.x (+bookmark-enrichment)"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    url: str
    html: str | None
    http_status: int | None
    content_type: str | None
    error: str | None
    fetched_at: int = field(default_factory=lambda: int(time.time()))

    @property
    def ok(self) -> bool:
        return self.html is not None and self.error is None


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def _is_http_url(url: str) -> bool:
    p = urlparse(url)
    return p.scheme.lower() in {"http", "https"} and bool(p.netloc)


def _read_capped(resp, max_bytes: int) -> bytes:
    """Read up to *max_bytes* from an HTTP response body."""
    chunks: list[bytes] = []
    remaining = max_bytes
    while remaining > 0:
        chunk = resp.read(min(8192, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode(raw: bytes, content_type: str | None) -> str:
    """Best-effort decode to str, honouring charset from Content-Type."""
    charset = "utf-8"
    if content_type:
        for part in content_type.split(";"):
            part = part.strip()
            if part.lower().startswith("charset="):
                charset = part[len("charset="):].strip().strip('"')
                break
    return raw.decode(charset, errors="replace")


def fetch_page(
    url: str,
    timeout: float = _DEFAULT_TIMEOUT,
    max_bytes: int = _MAX_FETCH_BYTES,
) -> FetchResult:
    """Fetch *url* and return a :class:`FetchResult`.

    * Only ``http`` / ``https`` URLs are fetched; others are skipped.
    * Response body is capped at *max_bytes* to avoid downloading huge pages.
    * Non-HTML content types (e.g. PDF, image) are skipped immediately.
    * On any network or HTTP error, returns a result with ``error`` set.
    """
    if not _is_http_url(url):
        return FetchResult(
            url=url, html=None, http_status=None,
            content_type=None, error="skipped: non-http URL",
        )

    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html,*/*;q=0.8"}

    def _do_get() -> FetchResult:
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 0) or 0)
                ct = resp.headers.get("Content-Type", "")
                # Skip non-HTML content types early (before reading body)
                if ct and not _looks_like_html(ct):
                    return FetchResult(
                        url=url, html=None, http_status=status,
                        content_type=ct, error=f"skipped: non-HTML content-type ({ct})",
                    )
                raw = _read_capped(resp, max_bytes)
                html = _decode(raw, ct or None)
                return FetchResult(
                    url=url, html=html, http_status=status,
                    content_type=ct or None, error=None,
                )
        except HTTPError as e:
            return FetchResult(
                url=url, html=None, http_status=int(e.code),
                content_type=None, error=f"HTTP {e.code}: {e.reason or 'error'}",
            )
        except URLError as e:
            reason = str(e.reason) if e.reason else str(e)
            return FetchResult(
                url=url, html=None, http_status=None,
                content_type=None, error=f"connection error: {reason}",
            )
        except TimeoutError:
            return FetchResult(
                url=url, html=None, http_status=None,
                content_type=None, error="timeout",
            )
        except Exception as e:  # pragma: no cover — defensive
            return FetchResult(
                url=url, html=None, http_status=None,
                content_type=None, error=str(e),
            )

    return _do_get()


def _looks_like_html(content_type: str) -> bool:
    ct_lower = content_type.lower()
    return "text/html" in ct_lower or "application/xhtml" in ct_lower


# ---------------------------------------------------------------------------
# Text extractor
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """SAX-style HTML parser that collects visible text, skipping boilerplate."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth: int = 0  # > 0 while inside a skip tag subtree
        self._last_was_block = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and not self._skip_depth:
            # Insert a separator between block elements
            if self._parts and not self._last_was_block:
                self._parts.append(" ")
            self._last_was_block = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS and not self._skip_depth:
            if self._parts and not self._last_was_block:
                self._parts.append(" ")
            self._last_was_block = True

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data
        if text.strip():
            self._parts.append(text)
            self._last_was_block = False

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse all whitespace sequences to a single space
        return re.sub(r"\s+", " ", raw).strip()


def extract_text(html: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """Extract human-readable text from *html*, stripping boilerplate.

    Returns at most *max_chars* characters.  Always returns a str (empty
    string on empty/unparse-able input).
    """
    if not html or not html.strip():
        return ""
    try:
        parser = _TextExtractor()
        parser.feed(html)
        text = parser.get_text()
        return text[:max_chars]
    except Exception:  # pragma: no cover — defensive
        return ""


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def text_content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest of *text* for change detection."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
