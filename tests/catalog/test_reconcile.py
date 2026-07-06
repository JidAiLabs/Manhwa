"""reconcile_chapter — derive true status from disk + repair the unsynced DB
stores. Complements rewind_chapter: reconcile never deletes an artifact, it only
makes chapter.status / stage_run / approval agree with the files that exist.
"""
from __future__ import annotations

import json
import os

from studio.catalog import reconcile
from studio.catalog.db import connect


def _mk_chapter(tmp_path, status="rendered"):
    con = connect(tmp_path / "studio.db")
    ep = tmp_path / "ep"
    (ep / "tts" / "clips").mkdir(parents=True)
    (ep / "render").mkdir()
    (ep / "scenes").mkdir()
    (ep / "001.jpg").write_text("page")
    (ep / "tts" / "tts_index.json").write_text("{}")
    (ep / "render" / "segment_both.mp4").write_text("mp4")
    for name in ("manifest.stitch.json", "manifest.panels.expanded.json",
                 "manifest.scenes.json", "manifest.vision.json",
                 "manifest.groups.json", "manifest.beats.json",
                 "manifest.script.json", "render.plan.json",
                 "render.plan.clean.json", "prep_qa.json"):
        (ep / name).write_text(json.dumps({"name": name}))
    # the render is produced FROM the clean plan, so a fresh mp4 is the newest
    # file (else _render_is_stale would flag the fixture itself as stale)
    _pm = os.path.getmtime(ep / "render.plan.clean.json")
    os.utime(ep / "render" / "segment_both.mp4", (_pm + 100, _pm + 100))
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','s','T','now')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES "
                "(310, 1, 1, 'Chapter 1', 'u', ?, ?, 'now')", (status, str(ep)))
    for stage in ("chain:scripted", "prepped", "qa_scan", "planned",
                  "voiced", "render_segment"):
        con.execute("INSERT INTO stage_run (chapter_id, stage, ok) "
                    "VALUES (310, ?, 1)", (stage,))
    for gate in ("voice", "render"):
        con.execute("INSERT INTO approval (gate, chapter_id) VALUES (?, 310)",
                    (gate,))
    con.commit()
    return con, ep


def _ch(con):
    r = con.execute("SELECT id, status, ep_dir FROM chapter WHERE id=310").fetchone()
    return {"id": r[0], "status": r[1], "ep_dir": r[2]}


def _stages(con):
    return sorted(r[0] for r in con.execute(
        "SELECT stage FROM stage_run WHERE chapter_id=310"))


def _n_approvals(con, gate):
    return con.execute("SELECT COUNT(*) FROM approval WHERE gate=? AND "
                       "chapter_id=310", (gate,)).fetchone()[0]


# ---- the marker mapping must not drift from the pipeline stage table ---------

def test_status_markers_in_sync_with_pipeline():
    from studio.pipeline import _STAGE_TABLE
    expected = [("rendered", "render/segment_both.mp4")] + [
        (status, marker) for status, _fn, marker in reversed(_STAGE_TABLE)]
    assert reconcile._STATUS_MARKERS == expected


# ---- derive_status -----------------------------------------------------------

def test_derive_status_walks_down_as_artifacts_vanish(tmp_path):
    _con, ep = _mk_chapter(tmp_path)
    ep = str(ep)
    assert reconcile.derive_status(ep) == "rendered"
    (tmp_path / "ep" / "render" / "segment_both.mp4").unlink()
    assert reconcile.derive_status(ep) == "planned"
    (tmp_path / "ep" / "render.plan.json").unlink()
    assert reconcile.derive_status(ep) == "voiced"
    (tmp_path / "ep" / "tts" / "tts_index.json").unlink()
    assert reconcile.derive_status(ep) == "scripted"


def test_derive_status_downloaded_and_empty(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert reconcile.derive_status(str(d)) is None       # nothing on disk
    (d / "001.jpg").write_text("page")
    assert reconcile.derive_status(str(d)) == "downloaded"
    assert reconcile.derive_status(str(tmp_path / "nope")) is None  # missing dir


def test_derive_status_estimated_plan_is_scripted_not_planned(tmp_path):
    # the 'prepare' stage writes render.plan.json (estimated) BEFORE voicing; with no
    # voiced audio it must read as 'scripted', not 'planned' — else a
    # scripted-awaiting-review chapter is wrongly advanced by reconcile.
    _con, ep = _mk_chapter(tmp_path)
    (ep / "render" / "segment_both.mp4").unlink()   # not rendered
    (ep / "tts" / "tts_index.json").unlink()        # not voiced (plan is an estimate)
    assert reconcile.derive_status(str(ep)) == "scripted"


def test_zero_byte_marker_not_promoted(tmp_path):
    # a torn write leaves a 0-byte/garbage .json marker; existence alone must
    # not promote the status (the 0-byte-manifest status promotion bug)
    _con, ep = _mk_chapter(tmp_path)
    (ep / "render" / "segment_both.mp4").unlink()      # walk below 'rendered'
    (ep / "render.plan.json").write_bytes(b"")         # torn planned marker
    assert reconcile.derive_status(str(ep)) == "voiced"
    (ep / "tts" / "tts_index.json").write_text("{not json")   # torn voiced too
    assert reconcile.derive_status(str(ep)) == "scripted"
    # non-JSON markers keep exists-only semantics: a 0-byte mp4 still counts
    (ep / "render" / "segment_both.mp4").write_bytes(b"")
    assert reconcile.derive_status(str(ep)) == "rendered"


# ---- reconcile_chapter -------------------------------------------------------

def test_consistent_chapter_is_a_noop(tmp_path):
    con, _ep = _mk_chapter(tmp_path, status="rendered")
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["status_to"] is None
    assert out["stage_runs_pruned"] == 0
    assert out["render_approval_cleared"] == 0
    assert len(_stages(con)) == 6


def test_repairs_status_prunes_stage_run_and_clears_approval_when_video_gone(tmp_path):
    con, ep = _mk_chapter(tmp_path, status="rendered")
    (ep / "render" / "segment_both.mp4").unlink()          # video gone
    out = reconcile.reconcile_chapter(con, _ch(con))
    # status repaired down to what the files say
    assert out["status_from"] == "rendered" and out["status_to"] == "planned"
    assert con.execute("SELECT status FROM chapter WHERE id=310").fetchone()[0] == "planned"
    # the render_segment stage_run row (its artifact is gone) is pruned; others stay
    assert "render_segment" not in _stages(con)
    assert "voiced" in _stages(con)
    # the render approval that outlived its video is cleared; voice untouched
    assert _n_approvals(con, "render") == 0
    assert _n_approvals(con, "voice") == 1


def test_prunes_voiced_stage_run_when_tts_gone(tmp_path):
    con, ep = _mk_chapter(tmp_path, status="voiced")
    (ep / "render" / "segment_both.mp4").unlink()          # not rendered
    (ep / "render.plan.json").unlink()                     # not planned
    (ep / "tts" / "tts_index.json").unlink()               # voiced artifact gone
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["status_to"] == "scripted"                  # highest surviving marker
    assert "voiced" not in _stages(con)                    # the lie is pruned


def test_keeps_fresh_render_approval(tmp_path):
    con, _ep = _mk_chapter(tmp_path, status="rendered")    # mp4 present + newest
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["render_approval_cleared"] == 0
    assert _n_approvals(con, "render") == 1


def test_clears_render_approval_when_video_older_than_plan(tmp_path):
    con, ep = _mk_chapter(tmp_path, status="rendered")
    import os
    # make the clean plan newer than the rendered mp4 (a stale render)
    os.utime(ep / "render" / "segment_both.mp4", (1, 1))
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["render_approval_cleared"] == 1
    assert _n_approvals(con, "render") == 0


def test_skips_chapter_with_active_job(tmp_path):
    con, ep = _mk_chapter(tmp_path, status="rendered")
    (ep / "render" / "segment_both.mp4").unlink()          # drift present
    con.execute("INSERT INTO job (type, state, chapter_id) "
                "VALUES ('prepare','running',310)")
    con.commit()
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["status_to"] is None                        # worker owns it — skip
    assert con.execute("SELECT status FROM chapter WHERE id=310").fetchone()[0] == "rendered"
    assert len(_stages(con)) == 6


def test_skips_failed_status(tmp_path):
    con, ep = _mk_chapter(tmp_path, status="beated_failed")
    (ep / "render" / "segment_both.mp4").unlink()
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["status_to"] is None                        # preserve the failure
    assert con.execute("SELECT status FROM chapter WHERE id=310").fetchone()[0] == "beated_failed"


def test_reconcile_series_totals(tmp_path):
    con, ep = _mk_chapter(tmp_path, status="rendered")
    (ep / "render" / "segment_both.mp4").unlink()
    totals = reconcile.reconcile_series(con, 1)
    assert totals["chapters_repaired"] == 1
    assert totals["stage_runs_pruned"] == 1                # render_segment row


# ---- _valid memo cache --------------------------------------------------------

def test_valid_rewrite_with_new_size_is_not_masked_by_the_cache(tmp_path):
    # A corrupt marker is invalid; fixing its content changes its size, so the
    # (mtime, size) cache key naturally misses and the repaired file is
    # re-parsed instead of being stuck on the stale corrupt verdict.
    reconcile._VALID_CACHE.clear()
    ep = tmp_path / "ep"
    ep.mkdir()
    marker = ep / "manifest.beats.json"
    marker.write_text("{not json")
    assert reconcile._valid(str(ep), "manifest.beats.json") is False
    marker.write_text(json.dumps({"beats": []}))
    assert reconcile._valid(str(ep), "manifest.beats.json") is True


def test_valid_caches_unchanged_file_skips_reparse(tmp_path, monkeypatch):
    # An untouched file's second _valid() call must hit the (mtime, size)
    # memo instead of re-parsing — the whole point of the cache, since a
    # dashboard load re-validates the same markers across every chapter.
    reconcile._VALID_CACHE.clear()
    ep = tmp_path / "ep"
    ep.mkdir()
    marker = ep / "manifest.beats.json"
    marker.write_text(json.dumps({"beats": []}))

    calls = {"n": 0}
    real_load = json.load

    def counting_load(f):
        calls["n"] += 1
        return real_load(f)

    monkeypatch.setattr(reconcile.json, "load", counting_load)

    assert reconcile._valid(str(ep), "manifest.beats.json") is True
    assert reconcile._valid(str(ep), "manifest.beats.json") is True
    assert calls["n"] == 1                     # second call hit the cache, no re-parse
