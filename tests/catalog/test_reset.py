"""rewind_chapter — the ONE atomic chapter rewind (files + DB together).

The class this exists to kill: every rewind used to hand-pick its own subset
(dashboard buttons, ad-hoc SQL) and each left stale state behind — most
visibly the append-only stage_run rows that kept readiness counters claiming
voiced/rendered after the artifacts were deleted (2026-07-03).
"""
from __future__ import annotations

import json

import pytest

from studio.catalog import reset
from studio.catalog.db import connect


def _mk_chapter(tmp_path, status="rendered"):
    con = connect(tmp_path / "studio.db")
    ep = tmp_path / "ep"
    (ep / "tts" / "clips").mkdir(parents=True)
    (ep / "render").mkdir()
    (ep / "scenes").mkdir()
    (ep / "001.jpg").write_text("page")
    (ep / "scenes" / "p1.jpg").write_text("scene")
    (ep / "tts" / "tts_index.json").write_text("{}")
    (ep / "tts" / "clips" / "g0001_p00.wav").write_text("wav")
    (ep / "render" / "segment_both.mp4").write_text("mp4")
    for name in ("manifest.beats.json", "manifest.script.json",
                 "render.plan.json", "render.plan.clean.json",
                 "prep_qa.json", "prep_qa.html", "heal_corrections.json",
                 "manifest.groups.json", "manifest.vision.json",
                 "manifest.panels.understood.json", "manifest.cast.json",
                 "manifest.story.json", "manifest.series.json"):
        (ep / name).write_text(json.dumps({"name": name}))
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','s','T','now')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES "
                "(310, 1, 1, 'Chapter 1', 'u', ?, ?, 'now')",
                (status, str(ep)))
    for stage in ("chain:scripted", "prepped", "qa_scan", "planned",
                  "voiced", "render_segment"):
        con.execute("INSERT INTO stage_run (chapter_id, stage, ok) "
                    "VALUES (310, ?, 1)", (stage,))
    for gate in ("voice", "render"):
        con.execute("INSERT INTO approval (gate, chapter_id) VALUES (?, 310)",
                    (gate,))
    con.commit()
    return con, ep


def _stages(con):
    return sorted(r[0] for r in con.execute(
        "SELECT stage FROM stage_run WHERE chapter_id=310"))


def test_rewind_to_grouped_clears_everything_derived(tmp_path):
    con, ep = _mk_chapter(tmp_path)
    s = reset.rewind_chapter(con, 310, "grouped")
    assert s["status"] == "grouped"
    assert con.execute("SELECT status FROM chapter WHERE id=310"
                       ).fetchone()[0] == "grouped"
    # narration + voice artifacts gone; upstream kept
    for gone in ("manifest.beats.json", "manifest.script.json",
                 "render.plan.json", "render.plan.clean.json",
                 "prep_qa.json", "heal_corrections.json", "tts", "render"):
        assert not (ep / gone).exists(), gone
    for kept in ("001.jpg", "scenes/p1.jpg", "manifest.groups.json",
                 "manifest.vision.json", "manifest.panels.understood.json",
                 "manifest.cast.json", "manifest.story.json",
                 "manifest.series.json"):
        assert (ep / kept).exists(), kept
    # beats backed up before deletion
    assert s["backup"] and (ep / s["backup"]).exists()
    # THE class this tool exists for: stage_run history rewound with the files
    assert _stages(con) == []
    assert con.execute("SELECT COUNT(*) FROM approval WHERE chapter_id=310"
                       ).fetchone()[0] == 0


def test_rewind_to_scripted_keeps_narration_clears_voice(tmp_path):
    con, ep = _mk_chapter(tmp_path)
    s = reset.rewind_chapter(con, 310, "scripted",
                             clear_approvals=("render",))
    assert (ep / "manifest.beats.json").exists()      # narration survives
    assert (ep / "manifest.script.json").exists()
    assert not (ep / "tts").exists()
    assert not (ep / "render").exists()
    assert s["backup"] is None                        # nothing destroyed
    assert _stages(con) == ["chain:scripted", "planned", "prepped", "qa_scan"]
    gates = [r[0] for r in con.execute(
        "SELECT gate FROM approval WHERE chapter_id=310")]
    assert gates == ["voice"]                         # render gate dropped


def test_rewind_keep_base_preserves_marker(tmp_path):
    con, ep = _mk_chapter(tmp_path)
    (ep / ".narration_keepbase").write_text("")
    reset.rewind_chapter(con, 310, "grouped")
    assert not (ep / ".narration_keepbase").exists()  # default: re-roll wins
    (ep / ".narration_keepbase").write_text("")
    reset.rewind_chapter(con, 310, "grouped", keep_base=True)
    assert (ep / ".narration_keepbase").exists()


def test_rewind_to_downloaded_wipes_visuals_keeps_pages(tmp_path):
    con, ep = _mk_chapter(tmp_path)
    reset.rewind_chapter(con, 310, "downloaded")
    assert (ep / "001.jpg").exists()                  # sources survive
    assert (ep / "manifest.series.json").exists()     # niche survives
    for gone in ("scenes", "manifest.groups.json", "manifest.vision.json",
                 "manifest.panels.understood.json", "manifest.cast.json",
                 "manifest.story.json"):
        assert not (ep / gone).exists(), gone


def test_rewind_unknown_target_or_chapter_raises(tmp_path):
    con, _ = _mk_chapter(tmp_path)
    with pytest.raises(ValueError):
        reset.rewind_chapter(con, 310, "beated")
    with pytest.raises(ValueError):
        reset.rewind_chapter(con, 999, "grouped")


def test_rewind_survives_missing_ep_dir(tmp_path):
    con, ep = _mk_chapter(tmp_path)
    import shutil
    shutil.rmtree(ep)
    s = reset.rewind_chapter(con, 310, "grouped")     # no crash
    assert s["status"] == "grouped"
    assert _stages(con) == []
