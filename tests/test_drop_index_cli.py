"""Tests for dropping the local index via CLI flag."""
from __future__ import annotations

from pathlib import Path

import pytest

from mindmark import cli


def test_main_drop_index_deletes_file(tmp_path):
    db = tmp_path / "drop.db"
    db.write_text("placeholder", encoding="utf-8")

    rc = cli.main(["--db", str(db), "drop-index", "--yes"])
    assert rc == 0
    assert not db.exists()


def test_main_drop_index_cancelled_by_prompt(tmp_path, monkeypatch):
    db = tmp_path / "drop_cancel.db"
    db.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = cli.main(["--db", str(db), "drop-index"])
    assert rc == 0
    assert db.exists()


def test_main_drop_index_rejects_subcommand(tmp_path):
    db = tmp_path / "drop_reject.db"
    with pytest.raises(SystemExit):
        cli.main(["drop-index", "stats", "--db", str(db)])


def test_main_drop_index_rejects_validate_combo(tmp_path):
    db = tmp_path / "drop_validate_combo.db"
    with pytest.raises(SystemExit):
        cli.main(["drop-index", "--validate", "--db", str(db)])


def test_main_drop_index_permission_fallback(tmp_path, monkeypatch):
    db = tmp_path / "locked.db"
    db.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(Path, "unlink", lambda _self: (_ for _ in ()).throw(PermissionError("in use")))
    monkeypatch.setattr(cli, "_clear_index_contents", lambda _path: True)

    rc = cli.main(["--db", str(db), "drop-index", "--yes"])
    assert rc == 0
