"""Deterministic tests for fetcher.py — no network access."""
from __future__ import annotations

import io
import time
from http.client import HTTPMessage
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from mindmark.fetcher import (
    FetchResult,
    _looks_like_html,
    extract_text,
    fetch_page,
    text_content_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8"):
    """Build a minimal mock urlopen response."""
    headers = HTTPMessage()
    headers["Content-Type"] = content_type
    resp = MagicMock()
    resp.status = status
    resp.headers = headers
    # Simulate streaming read
    _stream = io.BytesIO(body)
    resp.read.side_effect = lambda n: _stream.read(n)
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# fetch_page — non-HTTP skipping
# ---------------------------------------------------------------------------

def test_fetch_non_http_url_skipped():
    result = fetch_page("file:///etc/hosts")
    assert not result.ok
    assert "non-http" in result.error


def test_fetch_ftp_url_skipped():
    result = fetch_page("ftp://example.com/file.txt")
    assert not result.ok


def test_fetch_empty_string_skipped():
    result = fetch_page("")
    assert not result.ok


# ---------------------------------------------------------------------------
# fetch_page — successful fetch
# ---------------------------------------------------------------------------

def test_fetch_returns_html_on_200(monkeypatch):
    body = b"<html><body><p>Hello world</p></body></html>"
    resp = _make_response(body)

    with patch("mindmark.fetcher.urlopen", return_value=resp):
        result = fetch_page("https://example.com/", timeout=5)

    assert result.ok
    assert result.http_status == 200
    assert "Hello world" in result.html
    assert result.error is None


def test_fetch_sets_fetched_at(monkeypatch):
    before = int(time.time())
    body = b"<html><body>ok</body></html>"
    resp = _make_response(body)

    with patch("mindmark.fetcher.urlopen", return_value=resp):
        result = fetch_page("https://example.com/")

    assert result.fetched_at >= before


def test_fetch_decodes_charset_from_content_type():
    body = "héllo".encode("latin-1")
    resp = _make_response(body, content_type="text/html; charset=iso-8859-1")

    with patch("mindmark.fetcher.urlopen", return_value=resp):
        result = fetch_page("https://example.com/")

    assert result.ok
    assert "héllo" in result.html


# ---------------------------------------------------------------------------
# fetch_page — HTTP errors
# ---------------------------------------------------------------------------

def test_fetch_404_returns_error():
    with patch("mindmark.fetcher.urlopen", side_effect=HTTPError(
        url="https://example.com/", code=404, msg="Not Found", hdrs=None, fp=None
    )):
        result = fetch_page("https://example.com/missing")

    assert not result.ok
    assert result.http_status == 404
    assert "404" in result.error


def test_fetch_500_returns_error():
    with patch("mindmark.fetcher.urlopen", side_effect=HTTPError(
        url="https://example.com/", code=500, msg="Internal Server Error", hdrs=None, fp=None
    )):
        result = fetch_page("https://example.com/boom")

    assert not result.ok
    assert result.http_status == 500


# ---------------------------------------------------------------------------
# fetch_page — connection/timeout errors
# ---------------------------------------------------------------------------

def test_fetch_url_error_returns_error():
    with patch("mindmark.fetcher.urlopen", side_effect=URLError("Name or service not known")):
        result = fetch_page("https://nonexistent.invalid/")

    assert not result.ok
    assert result.http_status is None
    assert "connection error" in result.error


def test_fetch_timeout_returns_error():
    with patch("mindmark.fetcher.urlopen", side_effect=TimeoutError()):
        result = fetch_page("https://slow.example.com/")

    assert not result.ok
    assert "timeout" in result.error


# ---------------------------------------------------------------------------
# fetch_page — non-HTML content types
# ---------------------------------------------------------------------------

def test_fetch_pdf_skipped():
    resp = _make_response(b"%PDF-1.4", content_type="application/pdf")

    with patch("mindmark.fetcher.urlopen", return_value=resp):
        result = fetch_page("https://example.com/doc.pdf")

    assert not result.ok
    assert "non-HTML" in result.error


def test_fetch_image_skipped():
    resp = _make_response(b"\x89PNG\r\n", content_type="image/png")

    with patch("mindmark.fetcher.urlopen", return_value=resp):
        result = fetch_page("https://example.com/img.png")

    assert not result.ok


def test_fetch_json_skipped():
    resp = _make_response(b'{"key": "val"}', content_type="application/json")

    with patch("mindmark.fetcher.urlopen", return_value=resp):
        result = fetch_page("https://example.com/api")

    assert not result.ok


# ---------------------------------------------------------------------------
# fetch_page — size cap
# ---------------------------------------------------------------------------

def test_fetch_caps_body_size():
    # 600 KB body — only first 512 KB should be kept
    body = b"A" * (600 * 1024)
    resp = _make_response(body)

    with patch("mindmark.fetcher.urlopen", return_value=resp):
        result = fetch_page("https://example.com/big", max_bytes=512 * 1024)

    assert result.ok
    assert len(result.html.encode()) <= 512 * 1024 + 100  # small tolerance for encoding


# ---------------------------------------------------------------------------
# _looks_like_html helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ct,expected", [
    ("text/html; charset=utf-8", True),
    ("text/html", True),
    ("application/xhtml+xml", True),
    ("application/pdf", False),
    ("image/png", False),
    ("application/json", False),
    ("text/plain", False),
])
def test_looks_like_html(ct, expected):
    assert _looks_like_html(ct) is expected


# ---------------------------------------------------------------------------
# extract_text — basic extraction
# ---------------------------------------------------------------------------

def test_extract_basic_paragraph():
    html = "<html><body><p>Hello world</p></body></html>"
    assert "Hello world" in extract_text(html)


def test_extract_heading_and_paragraph():
    html = "<html><body><h1>Title</h1><p>Body text here.</p></body></html>"
    text = extract_text(html)
    assert "Title" in text
    assert "Body text here." in text


def test_extract_multiple_paragraphs_joined():
    html = "<p>First.</p><p>Second.</p><p>Third.</p>"
    text = extract_text(html)
    assert "First." in text
    assert "Second." in text
    assert "Third." in text


# ---------------------------------------------------------------------------
# extract_text — boilerplate stripping
# ---------------------------------------------------------------------------

def test_extract_strips_script():
    html = "<p>Good</p><script>alert('evil');</script><p>Also good</p>"
    text = extract_text(html)
    assert "Good" in text
    assert "alert" not in text


def test_extract_strips_style():
    html = "<p>Content</p><style>body { color: red; }</style>"
    text = extract_text(html)
    assert "Content" in text
    assert "color" not in text


def test_extract_strips_nav():
    html = "<nav><a href='/'>Home</a><a href='/about'>About</a></nav><main><p>Article text</p></main>"
    text = extract_text(html)
    assert "Article text" in text
    assert "Home" not in text
    assert "About" not in text


def test_extract_strips_header():
    html = "<header><h1>Site Logo</h1></header><article><p>Main content</p></article>"
    text = extract_text(html)
    assert "Main content" in text
    assert "Site Logo" not in text


def test_extract_strips_footer():
    html = "<p>Body</p><footer><p>Copyright 2026</p></footer>"
    text = extract_text(html)
    assert "Body" in text
    assert "Copyright" not in text


def test_extract_strips_nested_skip_tags():
    html = "<nav><div><ul><li><a>Nested nav link</a></li></ul></div></nav><p>Real content</p>"
    text = extract_text(html)
    assert "Real content" in text
    assert "Nested nav link" not in text


def test_extract_strips_head_section():
    html = "<html><head><title>Page Title</title><meta name='description' content='desc'></head><body><p>Body content</p></body></html>"
    text = extract_text(html)
    assert "Body content" in text
    assert "Page Title" not in text


# ---------------------------------------------------------------------------
# extract_text — whitespace normalisation
# ---------------------------------------------------------------------------

def test_extract_normalises_whitespace():
    html = "<p>  Too    many   spaces  </p>"
    text = extract_text(html)
    assert "  " not in text
    assert "Too many spaces" in text


def test_extract_collapses_newlines():
    html = "<p>Line\n\n\none</p>"
    text = extract_text(html)
    assert "\n\n" not in text


# ---------------------------------------------------------------------------
# extract_text — edge cases
# ---------------------------------------------------------------------------

def test_extract_empty_string():
    assert extract_text("") == ""


def test_extract_whitespace_only():
    assert extract_text("   \n\t  ") == ""


def test_extract_no_visible_text():
    html = "<html><head></head><body><nav><a>x</a></nav></body></html>"
    # All text is inside nav, so result should be empty or very short
    text = extract_text(html)
    assert "x" not in text


def test_extract_html_entities_decoded():
    html = "<p>Caf&eacute; &amp; Boulangerie &lt;Paris&gt;</p>"
    text = extract_text(html)
    assert "Café" in text
    assert "&eacute;" not in text
    assert "&amp;" not in text


def test_extract_truncates_to_max_chars():
    html = "<p>" + "word " * 5000 + "</p>"
    text = extract_text(html, max_chars=100)
    assert len(text) <= 100


def test_extract_real_world_structure():
    html = """
    <html>
    <head><title>My Blog</title></head>
    <body>
      <header><nav><a href="/">Home</a></nav></header>
      <main>
        <article>
          <h1>Understanding Python Async</h1>
          <p>Async programming in Python uses the <code>asyncio</code> module.</p>
          <p>It allows concurrent IO-bound tasks without threads.</p>
        </article>
      </main>
      <aside><p>Related: other posts</p></aside>
      <footer><p>© 2026 My Blog</p></footer>
    </body>
    </html>
    """
    text = extract_text(html)
    assert "Understanding Python Async" in text
    assert "asyncio" in text
    assert "concurrent IO-bound" in text
    # boilerplate
    assert "Home" not in text
    assert "Related" not in text
    assert "© 2026" not in text


# ---------------------------------------------------------------------------
# text_content_hash
# ---------------------------------------------------------------------------

def test_hash_is_deterministic():
    assert text_content_hash("hello") == text_content_hash("hello")


def test_hash_changes_on_different_input():
    assert text_content_hash("hello") != text_content_hash("world")


def test_hash_is_16_chars():
    h = text_content_hash("some text")
    assert len(h) == 16


def test_hash_empty_string():
    h = text_content_hash("")
    assert len(h) == 16
