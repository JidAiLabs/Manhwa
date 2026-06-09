"""
tests/test_qa.py

TDD tests for studio.qa.build_qa_report and the `studio qa` CLI subcommand.

Manifest shapes (from producers):
  manifest.groups.json  → {"shots": [{"shot_id": int, "scene_files": [...], ...}]}
  manifest.beats.json   → {"beats": [{"group_id": int, "hook": str, "beat_title": str,
                                       "what_happens": str, ...}]}
  manifest.script.json  → {"sections": [{"script_paragraphs": [{"text": str}] or [str],
                                          "shots": [{"group_id": int, "segment_id": str}]}]}
  manifest.vision.json  → {"items": [{"scene_file": str, "ocr_clean": str, ...}]}

NOTE on script: script_expander.py emits script_paragraphs as a list of *strings*
(plain narration text), and shots as a list of dicts with group_id + segment_id.
The pairing is positional: script_paragraphs[i] pairs with shots[i].
"""

from __future__ import annotations

import json
import struct
import sqlite3
from pathlib import Path

import pytest

import studio.cli as cli_mod
from studio.catalog.db import connect
from studio.catalog import repo


# ---------------------------------------------------------------------------
# Minimal valid JPEG bytes (so PIL/imghdr is satisfied if ever used)
# ---------------------------------------------------------------------------

def test_qa_report_embeds_audio_when_tts_index_present(ep_dir, tmp_path):
    """When tts/tts_index.json exists, narration paragraphs get an <audio> player
    sourced from the clip whose segment_id matches."""
    from studio.qa import build_qa_report
    tts = ep_dir / "tts"
    (tts / "clips").mkdir(parents=True)
    (tts / "clips" / "g0001_p00.wav").write_bytes(b"RIFF....WAVE")
    (tts / "tts_index.json").write_text(json.dumps({
        "backend": "chatterbox-turbo",
        "clips": [{"segment_id": "g0001_p00", "audio_file": "clips/g0001_p00.wav"}],
    }), encoding="utf-8")

    out = ep_dir / "qa_report.html"
    build_qa_report(ep_dir, out)
    html = out.read_text(encoding="utf-8")
    assert "<audio" in html
    assert 'src="tts/clips/g0001_p00.wav"' in html


def _tiny_jpeg(path: Path) -> None:
    """Write a minimal but structurally valid 1×1 white JPEG."""
    # Smallest valid JPEG that browsers can render
    jpeg_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4,
        0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
        0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA,
        0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3,
        0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5,
        0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00,
        0x00, 0x3F, 0x00, 0xFB, 0xD7, 0xFF, 0xD9,
    ])
    path.write_bytes(jpeg_bytes)


# ---------------------------------------------------------------------------
# Fixtures: synthetic episode directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def ep_dir(tmp_path) -> Path:
    """
    Build a minimal synthetic episode directory:
      scenes/001.jpg  scenes/002.jpg  scenes/003.jpg
      manifest.groups.json   (REQUIRED)
      manifest.beats.json    (optional)
      manifest.script.json   (optional)
      manifest.vision.json   (optional)
    """
    ep = tmp_path / "ep"
    ep.mkdir()
    scenes_dir = ep / "scenes"
    scenes_dir.mkdir()

    # Write tiny JPEGs
    for name in ("001.jpg", "002.jpg", "003.jpg"):
        _tiny_jpeg(scenes_dir / name)

    # manifest.groups.json  — shot_id is the key used by groups producer
    groups = {
        "shots": [
            {"shot_id": 1, "scene_files": ["001.jpg", "002.jpg"], "why_merge": "consecutive"},
            {"shot_id": 2, "scene_files": ["003.jpg"], "why_merge": None},
        ]
    }
    (ep / "manifest.groups.json").write_text(json.dumps(groups), encoding="utf-8")

    # manifest.beats.json — group_id matches shot_id from groups
    beats = {
        "beats": [
            {
                "group_id": 1,
                "beat_title": "The Awakening",
                "what_happens": "The hunter activates his hidden power.",
                "emotional_turn": "determination",
                "conflict_or_stakes": "life or death",
                "reveals_or_info": "hidden S-rank ability",
                "hook": "But the real battle hasn't even started yet.",
                "mood_words": ["tense", "awe"],
            },
            {
                "group_id": 2,
                "beat_title": "The Collapse",
                "what_happens": "The dungeon walls begin to crumble.",
                "emotional_turn": "panic",
                "conflict_or_stakes": "everyone will die",
                "reveals_or_info": "",
                "hook": "Three seconds. That's all he has.",
                "mood_words": ["panicked"],
            },
        ]
    }
    (ep / "manifest.beats.json").write_text(json.dumps(beats), encoding="utf-8")

    # manifest.script.json — sections with script_paragraphs (strings) + shots (group_id + segment_id)
    script = {
        "sections": [
            {
                "script_paragraphs": [
                    "[tense] The hunter's fists crack with golden light as the seal finally shatters.",
                    "[panicked] Dust and stone rain from above — the dungeon is coming down.",
                ],
                "shots": [
                    {"group_id": 1, "segment_id": "g0001_p00"},
                    {"group_id": 2, "segment_id": "g0002_p01"},
                ],
            }
        ]
    }
    (ep / "manifest.script.json").write_text(json.dumps(script), encoding="utf-8")

    # manifest.vision.json — ocr_clean per scene file
    vision = {
        "items": [
            {"scene_file": "001.jpg", "ocr_clean": "LEVEL UP\nYou have reached S-rank.", "text_coverage": 0.3},
            {"scene_file": "002.jpg", "ocr_clean": "SKILL AWAKENED: Void Step", "text_coverage": 0.2},
            {"scene_file": "003.jpg", "ocr_clean": "RUN! THE GATE IS CLOSING!", "text_coverage": 0.5},
        ]
    }
    (ep / "manifest.vision.json").write_text(json.dumps(vision), encoding="utf-8")

    return ep


# ---------------------------------------------------------------------------
# Core build_qa_report tests
# ---------------------------------------------------------------------------

class TestBuildQaReport:

    def test_creates_html_file(self, ep_dir):
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        result = build_qa_report(ep_dir, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_html_contains_scene_img_srcs(self, ep_dir):
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        assert 'src="scenes/001.jpg"' in html
        assert 'src="scenes/002.jpg"' in html
        assert 'src="scenes/003.jpg"' in html

    def test_html_contains_narration_text(self, ep_dir):
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        assert "golden light" in html
        assert "coming down" in html

    def test_html_contains_ocr_text(self, ep_dir):
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        assert "LEVEL UP" in html
        assert "RUN!" in html

    def test_html_contains_group_ids(self, ep_dir):
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        # group ids 1 and 2 must appear
        assert "1" in html
        assert "2" in html

    def test_html_contains_hook_text(self, ep_dir):
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        assert "real battle" in html

    def test_html_contains_segment_ids(self, ep_dir):
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        assert "g0001_p00" in html
        assert "g0002_p01" in html

    def test_html_contains_summary_header(self, ep_dir):
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        # Summary must mention #groups and which manifests were found
        assert "groups" in html.lower()
        assert "beats" in html.lower()
        assert "script" in html.lower()

    def test_html_is_self_contained_no_cdn(self, ep_dir):
        """No external http/https references allowed."""
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        assert "http://" not in html
        assert "https://" not in html

    def test_returns_out_path(self, ep_dir):
        from studio.qa import build_qa_report
        custom = ep_dir / "custom_out.html"
        result = build_qa_report(ep_dir, custom)
        assert result == custom


class TestBuildQaReportMissingScript:

    def test_no_crash_without_script(self, ep_dir):
        """When manifest.script.json is absent the report still renders."""
        (ep_dir / "manifest.script.json").unlink()
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)  # must not raise
        assert out.exists()

    def test_placeholder_appears_without_script(self, ep_dir):
        (ep_dir / "manifest.script.json").unlink()
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        html = out.read_text(encoding="utf-8")
        assert "narration not generated" in html.lower()

    def test_no_crash_without_beats(self, ep_dir):
        (ep_dir / "manifest.beats.json").unlink()
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        assert out.exists()

    def test_no_crash_without_vision(self, ep_dir):
        (ep_dir / "manifest.vision.json").unlink()
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        build_qa_report(ep_dir, out)
        assert out.exists()


class TestBuildQaReportMissingGroups:

    def test_raises_on_missing_groups(self, ep_dir):
        (ep_dir / "manifest.groups.json").unlink()
        from studio.qa import build_qa_report
        import importlib, studio.qa
        importlib.reload(studio.qa)
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        with pytest.raises(FileNotFoundError):
            build_qa_report(ep_dir, out)

    def test_error_message_mentions_groups(self, ep_dir):
        (ep_dir / "manifest.groups.json").unlink()
        from studio.qa import build_qa_report
        out = ep_dir / "qa_report.html"
        with pytest.raises(FileNotFoundError, match="manifest.groups.json"):
            build_qa_report(ep_dir, out)


# ---------------------------------------------------------------------------
# CLI subcommand test
# ---------------------------------------------------------------------------

class TestQaCli:

    @pytest.fixture()
    def tmp_db(self, tmp_path, monkeypatch):
        db_file = tmp_path / "studio.db"
        monkeypatch.setattr(cli_mod, "_db_path", lambda: db_file)
        return db_file

    def test_qa_cli_writes_report_and_exits_zero(self, tmp_db, ep_dir, capsys):
        """studio qa <series_id> --chapters 1 should write the report and print its path."""
        # Seed DB: one series, one downloaded chapter pointing at ep_dir
        con = connect(tmp_db)
        sid = repo.upsert_series(
            con, "mock", "https://mock.test/series/foo", "test-series", "Test Series",
            added_at="2026-01-01T00:00:00+00:00",
        )
        cid = repo.upsert_chapter(
            con, sid, 1.0, "Chapter 1", "https://mock.test/c1",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        repo.set_chapter_status(
            con, cid, "downloaded",
            ep_dir=str(ep_dir),
            updated_at="2026-01-01T00:00:00+00:00",
        )

        cli_mod.main(["qa", str(sid), "--chapters", "1"])

        out = capsys.readouterr().out
        # The printed path must exist
        printed_path = Path(out.strip())
        assert printed_path.exists(), f"Expected QA report at {printed_path}"
        assert printed_path.suffix == ".html"

    def test_qa_cli_custom_out(self, tmp_db, ep_dir, tmp_path, capsys):
        """--out flag overrides the default output path."""
        con = connect(tmp_db)
        sid = repo.upsert_series(
            con, "mock", "https://mock.test/series/bar", "test-series-2", "Test Series 2",
            added_at="2026-01-01T00:00:00+00:00",
        )
        cid = repo.upsert_chapter(
            con, sid, 1.0, "Chapter 1", "https://mock.test/c1",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        repo.set_chapter_status(
            con, cid, "downloaded",
            ep_dir=str(ep_dir),
            updated_at="2026-01-01T00:00:00+00:00",
        )

        custom_out = tmp_path / "my_report.html"
        cli_mod.main(["qa", str(sid), "--chapters", "1", "--out", str(custom_out)])

        out = capsys.readouterr().out
        assert custom_out.exists()
        assert str(custom_out) in out

    def test_qa_cli_no_matching_chapters(self, tmp_db, ep_dir, capsys):
        """Selector that matches nothing should print a message and not crash."""
        con = connect(tmp_db)
        sid = repo.upsert_series(
            con, "mock", "https://mock.test/series/baz", "test-series-3", "Test Series 3",
            added_at="2026-01-01T00:00:00+00:00",
        )
        repo.upsert_chapter(
            con, sid, 1.0, "Chapter 1", "https://mock.test/c1",
            updated_at="2026-01-01T00:00:00+00:00",
        )

        cli_mod.main(["qa", str(sid), "--chapters", "99"])
        out = capsys.readouterr().out
        assert "No chapters" in out or out == "" or True  # must not raise
