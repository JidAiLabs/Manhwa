"""
tests/test_cli.py

Tests for studio.cli.

Mocks the source adapter via sources.base.REGISTRY; mocks adapter.download to
write a sentinel image file; monkeypatches the DB path to a temp file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import studio.cli as cli_mod
from studio.catalog.db import connect
from studio.catalog import repo
from studio.catalog.models import Chapter
from studio.sources.base import ChapterRef, SeriesMeta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXED_NOW = "2026-06-09T00:00:00+00:00"


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Redirect the CLI's DB to a temp file; return the temp path."""
    db_file = tmp_path / "studio.db"
    monkeypatch.setattr(cli_mod, "_db_path", lambda: db_file)
    return db_file


@pytest.fixture()
def mock_adapter(monkeypatch):
    """Return a MagicMock adapter registered under id 'mock'."""
    from studio.sources import base

    adapter = MagicMock()
    adapter.id = "mock"

    # Default series_meta return value
    adapter.series_meta.return_value = SeriesMeta(
        source="mock",
        series_url="https://mock.test/series/foo",
        title="Test Series",
        slug="test-series",
    )

    # Default list_chapters: 3 chapters
    adapter.list_chapters.return_value = [
        ChapterRef(number=1.0, label="Chapter 1", url="https://mock.test/c1"),
        ChapterRef(number=2.0, label="Chapter 2", url="https://mock.test/c2"),
        ChapterRef(number=3.0, label="Chapter 3", url="https://mock.test/c3"),
    ]

    # download writes a sentinel file
    def _download(chapter_ref, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "001.jpg").write_bytes(b"fake-image")
        return [dest_dir / "001.jpg"]

    adapter.download.side_effect = _download

    # Register under "mock"
    original = base.REGISTRY.copy()
    base.REGISTRY["mock"] = adapter
    yield adapter
    # Restore registry to avoid test pollution
    base.REGISTRY.clear()
    base.REGISTRY.update(original)


# ---------------------------------------------------------------------------
# parse_chapter_selector
# ---------------------------------------------------------------------------

class TestParseChapterSelector:
    def _make_chapters(self, numbers: list[float]) -> list[Chapter]:
        return [
            Chapter(
                id=i + 1,
                series_id=1,
                number=n,
                label=f"Ch {n}",
                url=f"https://x.test/c{i}",
                status="discovered",
            )
            for i, n in enumerate(numbers)
        ]

    def test_single(self):
        chapters = self._make_chapters([1.0, 2.0, 3.0])
        result = cli_mod.parse_chapter_selector("2", chapters)
        assert [c.number for c in result] == [2.0]

    def test_range(self):
        chapters = self._make_chapters([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cli_mod.parse_chapter_selector("1-3", chapters)
        assert [c.number for c in result] == [1.0, 2.0, 3.0]

    def test_new_returns_only_discovered(self):
        chapters = self._make_chapters([1.0, 2.0, 3.0])
        chapters[0] = Chapter(
            id=1, series_id=1, number=1.0, label="Ch 1",
            url="https://x.test/c0", status="downloaded",
        )
        result = cli_mod.parse_chapter_selector("new", chapters)
        assert [c.number for c in result] == [2.0, 3.0]

    def test_invalid_spec_raises(self):
        chapters = self._make_chapters([1.0])
        with pytest.raises(ValueError, match="Invalid"):
            cli_mod.parse_chapter_selector("all", chapters)

    def test_range_inclusive(self):
        chapters = self._make_chapters([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cli_mod.parse_chapter_selector("2-4", chapters)
        assert [c.number for c in result] == [2.0, 3.0, 4.0]

    def test_new_empty_when_all_downloaded(self):
        chapters = self._make_chapters([1.0, 2.0])
        for ch in chapters:
            object.__setattr__(ch, "status", "downloaded") if hasattr(ch, "__setattr__") else None
        # dataclass is mutable
        chapters[0].status = "downloaded"
        chapters[1].status = "downloaded"
        result = cli_mod.parse_chapter_selector("new", chapters)
        assert result == []


# ---------------------------------------------------------------------------
# add-series
# ---------------------------------------------------------------------------

class TestAddSeries:
    def test_inserts_series_and_chapters(self, tmp_db, mock_adapter, capsys):
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])

        con = connect(tmp_db)
        series_list = repo.list_series(con)
        assert len(series_list) == 1
        s = series_list[0]
        assert s.source == "mock"
        assert s.slug == "test-series"
        assert s.title == "Test Series"

        chapters = repo.list_chapters(con, s.id)
        assert len(chapters) == 3
        assert [c.number for c in chapters] == [1.0, 2.0, 3.0]
        assert all(c.status == "discovered" for c in chapters)

        out = capsys.readouterr().out
        assert "series_id=" in out
        assert "chapters=3" in out

    def test_idempotent_add(self, tmp_db, mock_adapter):
        """Running add-series twice should not duplicate."""
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])

        con = connect(tmp_db)
        assert len(repo.list_series(con)) == 1
        series_list = repo.list_series(con)
        assert len(repo.list_chapters(con, series_list[0].id)) == 3


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

class TestFetch:
    def test_fetch_sets_downloaded_status(self, tmp_db, mock_adapter, tmp_path, monkeypatch, capsys):
        # Seed DB
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        con = connect(tmp_db)
        sid = repo.list_series(con)[0].id

        # Redirect ongoing/ to tmp_path so we don't write into the real repo
        monkeypatch.setattr(studio_config, "REPO_ROOT", tmp_path)

        cli_mod.main(["fetch", str(sid), "--chapters", "1"])

        chapters = repo.list_chapters(con, sid)
        ch1 = next(c for c in chapters if c.number == 1.0)
        assert ch1.status == "downloaded"
        assert ch1.ep_dir is not None
        # sentinel image written by stub download
        assert (Path(ch1.ep_dir) / "001.jpg").exists()

    def test_fetch_skips_already_downloaded_without_force(self, tmp_db, mock_adapter, tmp_path, monkeypatch, capsys):
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        con = connect(tmp_db)
        sid = repo.list_series(con)[0].id

        monkeypatch.setattr(studio_config, "REPO_ROOT", tmp_path)

        # First fetch
        cli_mod.main(["fetch", str(sid), "--chapters", "1"])
        first_call_count = mock_adapter.download.call_count

        # Second fetch without --force → skipped
        cli_mod.main(["fetch", str(sid), "--chapters", "1"])
        assert mock_adapter.download.call_count == first_call_count

    def test_fetch_force_redownloads(self, tmp_db, mock_adapter, tmp_path, monkeypatch):
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        con = connect(tmp_db)
        sid = repo.list_series(con)[0].id

        monkeypatch.setattr(studio_config, "REPO_ROOT", tmp_path)

        cli_mod.main(["fetch", str(sid), "--chapters", "1"])
        first_count = mock_adapter.download.call_count

        cli_mod.main(["fetch", str(sid), "--chapters", "1", "--force"])
        assert mock_adapter.download.call_count == first_count + 1

    def test_fetch_range(self, tmp_db, mock_adapter, tmp_path, monkeypatch):
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        con = connect(tmp_db)
        sid = repo.list_series(con)[0].id

        monkeypatch.setattr(studio_config, "REPO_ROOT", tmp_path)

        cli_mod.main(["fetch", str(sid), "--chapters", "1-3"])
        assert mock_adapter.download.call_count == 3

        chapters = repo.list_chapters(con, sid)
        assert all(c.status == "downloaded" for c in chapters)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_prints_without_error(self, tmp_db, mock_adapter, tmp_path, monkeypatch, capsys):
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        con = connect(tmp_db)
        sid = repo.list_series(con)[0].id

        monkeypatch.setattr(studio_config, "REPO_ROOT", tmp_path)
        cli_mod.main(["fetch", str(sid), "--chapters", "1"])

        cli_mod.main(["status", str(sid)])
        out = capsys.readouterr().out
        assert "downloaded" in out
        assert "Chapter 1" in out

    def test_status_no_series_id_lists_all(self, tmp_db, mock_adapter, capsys):
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        cli_mod.main(["status"])
        out = capsys.readouterr().out
        assert "Test Series" in out


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

class TestList:
    def test_list_no_args_shows_series(self, tmp_db, mock_adapter, capsys):
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        cli_mod.main(["list"])
        out = capsys.readouterr().out
        assert "Test Series" in out

    def test_list_series_shows_chapters(self, tmp_db, mock_adapter, capsys):
        cli_mod.main(["add-series", "mock", "https://mock.test/series/foo"])
        con = connect(tmp_db)
        sid = repo.list_series(con)[0].id
        cli_mod.main(["list", "--series", str(sid)])
        out = capsys.readouterr().out
        assert "Chapter 1" in out
        assert "Chapter 2" in out
        assert "Chapter 3" in out


# ---------------------------------------------------------------------------
# --help smoke test
# ---------------------------------------------------------------------------

class TestHelp:
    def test_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            cli_mod.main(["--help"])
        assert exc.value.code == 0

    def test_add_series_help(self):
        with pytest.raises(SystemExit) as exc:
            cli_mod.main(["add-series", "--help"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Import studio_config for monkeypatching REPO_ROOT
# ---------------------------------------------------------------------------

from studio import config as studio_config
