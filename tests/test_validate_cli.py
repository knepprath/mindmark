"""Tests for URL validation CLI flow."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from mindmark import cli
from mindmark.index import Index
from mindmark.parser import Bookmark


def _make_bookmark(url: str, title: str = "T", folder: str = "") -> Bookmark:
    return Bookmark(title=title, url=url, folder_path=folder, add_date=0, icon=None)


def _build_index(db_path: Path, bookmarks: list[Bookmark]) -> None:
    idx = Index(db_path=db_path)
    mock_embedder = MagicMock()

    def fake_embed(texts: list[str]) -> np.ndarray:
        vecs = np.ones((len(texts), 4), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    mock_embedder.embed.side_effect = fake_embed
    mock_embedder.embed_one.side_effect = lambda t: fake_embed([t])[0]
    idx.embedder = mock_embedder
    idx.rebuild(bookmarks)
    idx.close()


def test_validate_all_healthy_no_prompt(tmp_path, monkeypatch):
    db = tmp_path / "validate_ok.db"
    _build_index(
        db,
        [
            _make_bookmark("https://a.example.com", "A"),
            _make_bookmark("https://b.example.com", "B"),
        ],
    )

    monkeypatch.setattr(cli, "_check_url_status", lambda url, timeout: (url, 200, None))

    def fail_input(_prompt: str) -> str:
        raise AssertionError("input() should not be called when all URLs are healthy")

    monkeypatch.setattr("builtins.input", fail_input)

    args = Namespace(db=db, timeout=0.5, workers=2, yes=False)
    rc = cli._cmd_validate(args)
    assert rc == 0


def test_validate_stale_prompt_yes_trims(tmp_path, monkeypatch):
    db = tmp_path / "validate_trim.db"
    stale_url = "https://stale.example.com"
    keep_url = "https://keep.example.com"
    _build_index(
        db,
        [
            _make_bookmark(stale_url, "Stale"),
            _make_bookmark(keep_url, "Keep"),
        ],
    )

    def fake_check(url: str, timeout: float):
        if url == stale_url:
            return (url, 404, "Not Found")
        return (url, 200, None)

    monkeypatch.setattr(cli, "_check_url_status", fake_check)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    args = Namespace(db=db, timeout=0.5, workers=2, yes=False)
    rc = cli._cmd_validate(args)
    assert rc == 0

    idx = Index(db_path=db)
    try:
        urls = [b["url"] for b in idx.all_bookmarks()]
        assert keep_url in urls
        assert stale_url not in urls
    finally:
        idx.close()


def test_main_validate_dispatch(monkeypatch, tmp_path):
    db = tmp_path / "dispatch.db"
    called = {"ok": False}

    def fake_validate(args):
        called["ok"] = True
        assert args.db == str(db)
        assert args.yes is True
        return 0

    monkeypatch.setattr(cli, "_cmd_validate", fake_validate)
    rc = cli.main(["--validate", "--yes", "--db", str(db)])
    assert rc == 0
    assert called["ok"] is True


def test_main_validate_rejects_subcommand(tmp_path):
    db = tmp_path / "reject.db"
    with pytest.raises(SystemExit):
        cli.main(["--validate", "--db", str(db), "stats"])
