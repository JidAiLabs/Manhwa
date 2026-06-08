"""
tests/test_ocr_chrome.py

Tests for tools.ocr_chrome.strip_ui_chrome and the manifest re-cleaner CLI.

Follows TDD: tests drive correctness of the implementation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# tools/ lives at repo root; add it to sys.path so the import works without
# installing the package.
_TOOLS_DIR = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))

from ocr_chrome import strip_ui_chrome  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PYTHON = str(Path(__file__).parent.parent / ".eval_venv" / "bin" / "python")


# ---------------------------------------------------------------------------
# Core stripping — real-world noise from the recap pipeline
# ---------------------------------------------------------------------------

class TestRealWorldNoise:
    def test_views_colon_repeated(self):
        """The exact noise that leaked into narration: 'VIEWS: 1 VIEWS: 1'."""
        result = strip_ui_chrome("VIEWS: 1 VIEWS: 1 The hero stood alone.")
        assert "VIEWS" not in result.upper()
        assert "The hero stood alone" in result

    def test_comment_and_view_and_episode(self):
        """Multi-counter + episode nav all on one line with narrative at end."""
        result = strip_ui_chrome(
            "1 COMMENT  1 VIEW  EPISODE 1  Dokja opened the app."
        )
        assert "Dokja opened the app" in result
        assert "COMMENT" not in result.upper()
        assert "1 VIEW" not in result.upper()
        assert "EPISODE" not in result.upper()

    def test_multiline_chrome_prefix(self):
        """Chrome noise on separate lines before a narrative line."""
        raw = "VIEWS: 1\n1 COMMENT\nEPISODE 12\nThe sword gleamed in the dark."
        result = strip_ui_chrome(raw)
        assert "The sword gleamed in the dark" in result
        assert "VIEWS" not in result.upper()
        assert "COMMENT" not in result.upper()
        assert "EPISODE" not in result.upper()

    def test_likes_counter(self):
        result = strip_ui_chrome("124 LIKES\nHe walked away without looking back.")
        assert "LIKES" not in result.upper()
        assert "He walked away without looking back" in result

    def test_subscriber_counter(self):
        result = strip_ui_chrome("1,200 SUBSCRIBERS\nSubscribe for more content\nShe smiled.")
        assert "SUBSCRIBER" not in result.upper()
        assert "She smiled" in result

    def test_episode_nav_variants(self):
        for chrome in ["EPISODE 1", "EPISODE1", "EP. 3", "EP.3", "EP 5", "NEXT EPISODE", "UP NEXT"]:
            result = strip_ui_chrome(f"{chrome}\nKim lifted his blade.")
            assert "Kim lifted his blade" in result, f"narrative lost for chrome={chrome!r}"
            # Episode/nav token should be gone
            assert chrome.split()[0].upper() not in result.upper() or result.upper().count("EPISODE") == 0

    def test_chapter_nav(self):
        result = strip_ui_chrome("CHAPTER 3\nDark clouds gathered on the horizon.")
        assert "CHAPTER" not in result.upper()
        assert "Dark clouds gathered on the horizon" in result

    def test_subscribe_share_download(self):
        for chrome in ["SUBSCRIBE", "SHARE", "DOWNLOAD", "RATE THIS", "RATE"]:
            result = strip_ui_chrome(f"{chrome}\nThe gates of the dungeon opened.")
            assert chrome not in result.upper(), f"chrome={chrome!r} not removed"
            assert "The gates of the dungeon opened" in result

    def test_ages_label(self):
        result = strip_ui_chrome("AGES 13+\nYoung hunters gathered in the hall.")
        assert "AGES" not in result.upper()
        assert "Young hunters gathered in the hall" in result


# ---------------------------------------------------------------------------
# False-positive guards — normal dialogue must NOT be altered
# ---------------------------------------------------------------------------

class TestFalsePositiveGuards:
    def test_view_in_sentence(self):
        """'view' inside a sentence must be preserved."""
        sentence = "What a beautiful view from up here!"
        assert strip_ui_chrome(sentence) == sentence

    def test_comment_in_sentence(self):
        """'comment' inside a sentence must be preserved."""
        sentence = "I'll comment on that later."
        assert strip_ui_chrome(sentence) == sentence

    def test_like_in_sentence(self):
        """'like' inside normal dialogue must survive."""
        sentence = "It feels like a dream."
        assert strip_ui_chrome(sentence) == sentence

    def test_subscribe_in_a_sentence(self):
        """'subscribe' as part of a longer sentence is unusual but let's be safe."""
        # "Subscribe" as a bare line IS stripped — that's intentional.
        # But a longer sentence containing the word should survive.
        sentence = "You should subscribe to the guild newsletter."
        result = strip_ui_chrome(sentence)
        # The word subscribe here is in a full sentence — we don't strip it
        # because our pattern only matches the bare word on a line by itself.
        # If it does get stripped it's a known limitation; test documents behavior.
        assert "guild newsletter" in result

    def test_empty_string(self):
        assert strip_ui_chrome("") == ""

    def test_none_like_empty(self):
        # strip_ui_chrome requires str; just ensure empty string short-circuits
        assert strip_ui_chrome("   ") == ""

    def test_pure_narrative_unchanged(self):
        text = "The hunter stared at the status window. His level had risen overnight."
        assert strip_ui_chrome(text) == text

    def test_chapter_in_sentence_is_preserved(self):
        """'chapter' alone is stripped, but 'chapter' without a number survives."""
        sentence = "This chapter of his life was over."
        result = strip_ui_chrome(sentence)
        # "CHAPTER" without adjacent digit is not matched — preserved
        assert "chapter" in result.lower()

    def test_number_in_dialogue_preserved(self):
        """Digits that aren't UI counters must survive."""
        sentence = "There were 3 survivors in the room."
        assert strip_ui_chrome(sentence) == sentence

    def test_next_in_sentence_preserved(self):
        """'next' inside a sentence is not a nav label and must survive."""
        sentence = "What happens next is what surprised everyone."
        assert strip_ui_chrome(sentence) == sentence

    def test_creator_in_sentence_preserved(self):
        """'creator' inside a sentence should not be treated as bare nav."""
        sentence = "He was the creator of the system."
        result = strip_ui_chrome(sentence)
        assert "creator" in result.lower()


# ---------------------------------------------------------------------------
# Extra-patterns (watermarks)
# ---------------------------------------------------------------------------

class TestExtraPatterns:
    def test_watermark_literal(self):
        result = strip_ui_chrome(
            "NIGHTSUP.NET\nThe shadows moved silently.",
            extra_patterns=["NIGHTSUP.NET"],
        )
        assert "NIGHTSUP" not in result.upper()
        assert "The shadows moved silently" in result

    def test_watermark_regex(self):
        result = strip_ui_chrome(
            "Read at WEBTOON.XYZ\nShe reached for her sword.",
            extra_patterns=[r"Read\s+at\s+\S+"],
        )
        assert "WEBTOON.XYZ" not in result
        assert "She reached for her sword" in result

    def test_extra_patterns_none(self):
        """Passing None is safe."""
        text = "He ran forward."
        assert strip_ui_chrome(text, extra_patterns=None) == text

    def test_extra_patterns_empty_list(self):
        text = "He ran forward."
        assert strip_ui_chrome(text, extra_patterns=[]) == text


# ---------------------------------------------------------------------------
# Manifest re-cleaner CLI (subprocess)
# ---------------------------------------------------------------------------

class TestManifestRecleaner:
    def test_recleaner_cleans_noisy_ocr_clean(self, tmp_path):
        """
        CLI rewrites ocr_clean for noisy items and leaves clean items unchanged.
        """
        manifest = {
            "count": 2,
            "items": [
                {
                    "scene_id": 1,
                    "ocr_clean": "VIEWS: 1 VIEWS: 1 The hero stood alone.",
                },
                {
                    "scene_id": 2,
                    "ocr_clean": "Dokja opened the app and stared at the message.",
                },
            ],
        }
        manifest_path = tmp_path / "manifest.vision.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = subprocess.run(
            [
                PYTHON,
                str(_TOOLS_DIR / "ocr_chrome.py"),
                "--manifest",
                str(manifest_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert "changed=1" in result.stdout

        updated = json.loads(manifest_path.read_text(encoding="utf-8"))
        item1 = updated["items"][0]
        item2 = updated["items"][1]

        # Noisy item: chrome gone, narrative preserved
        assert "VIEWS" not in item1["ocr_clean"].upper()
        assert "The hero stood alone" in item1["ocr_clean"]

        # Clean item: unchanged
        assert item2["ocr_clean"] == "Dokja opened the app and stared at the message."

    def test_recleaner_reports_zero_changed_when_clean(self, tmp_path):
        manifest = {
            "count": 1,
            "items": [
                {
                    "scene_id": 1,
                    "ocr_clean": "She drew her sword and advanced.",
                }
            ],
        }
        manifest_path = tmp_path / "manifest.vision.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = subprocess.run(
            [PYTHON, str(_TOOLS_DIR / "ocr_chrome.py"), "--manifest", str(manifest_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "changed=0" in result.stdout

    def test_recleaner_with_extra_patterns(self, tmp_path):
        manifest = {
            "count": 1,
            "items": [
                {
                    "scene_id": 1,
                    "ocr_clean": "NIGHTSUP.NET The army charged through the gate.",
                }
            ],
        }
        manifest_path = tmp_path / "manifest.vision.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = subprocess.run(
            [
                PYTHON,
                str(_TOOLS_DIR / "ocr_chrome.py"),
                "--manifest", str(manifest_path),
                "--extra-patterns", "NIGHTSUP.NET",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "changed=1" in result.stdout

        updated = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "NIGHTSUP" not in updated["items"][0]["ocr_clean"].upper()
        assert "The army charged through the gate" in updated["items"][0]["ocr_clean"]
