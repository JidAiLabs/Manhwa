"""reconcile_chapter — derive true status from disk + repair the unsynced DB
stores. Complements rewind_chapter: reconcile never deletes an artifact, it only
makes chapter.status / stage_run / approval agree with the files that exist.
"""
from __future__ import annotations

import json
import os

import pytest

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


def test_legacy_null_sha_approval_is_backfilled_on_reconcile(tmp_path):
    """A pre-content_sha approval row (content_sha IS NULL) is grandfathered
    valid, but reconcile stamps it with TODAY's sha (one-time) so future
    drift becomes detectable instead of the row staying eternally NULL.
    _mk_chapter creates BOTH the 'render' and 'voice' approval rows with a
    NULL content_sha, so a single reconcile backfills both -> count is 2
    (Task 12 extended this backfill to the voice gate)."""
    from studio.dashboard import gates
    con, ep = _mk_chapter(tmp_path, status="rendered")   # both approval rows NULL
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["content_sha_backfilled"] == 2
    stored = con.execute("SELECT content_sha FROM approval WHERE gate='render' "
                         "AND chapter_id=310").fetchone()[0]
    assert stored is not None
    assert stored == gates.gate_sha("render", str(ep))
    voice_stored = con.execute("SELECT content_sha FROM approval WHERE gate='voice' "
                               "AND chapter_id=310").fetchone()[0]
    assert voice_stored is not None
    assert voice_stored == gates.gate_sha("voice", str(ep))


def test_clears_render_approval_when_sha_drifts(tmp_path):
    """Replaces the old mtime-based staleness check (video older than plan),
    which raced under concurrent writes. A healed script / regenerated plan
    changes render.plan.clean.json or tts/tts_index.json BYTES -> the stored
    content_sha no longer matches -> reconcile clears the stale approval."""
    from studio.dashboard import gates
    con, ep = _mk_chapter(tmp_path, status="rendered")
    current = gates.gate_sha("render", str(ep))
    con.execute("UPDATE approval SET content_sha=? WHERE gate='render' "
                "AND chapter_id=310", (current,))
    con.commit()
    (ep / "tts" / "tts_index.json").write_text('{"changed": true}')  # re-voiced
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["render_approval_cleared"] == 1
    assert _n_approvals(con, "render") == 0


def test_keeps_render_approval_when_video_merely_older_than_plan(tmp_path):
    """mtime alone is no longer a staleness signal (it raced under concurrent
    writes) — only a missing mp4 or a content_sha mismatch clears the
    approval now, so an mp4 that's simply OLDER than the plan (content
    unchanged) is NOT cleared."""
    con, ep = _mk_chapter(tmp_path, status="rendered")
    import os
    os.utime(ep / "render" / "segment_both.mp4", (1, 1))
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["render_approval_cleared"] == 0
    assert _n_approvals(con, "render") == 1


def test_clears_voice_approval_when_script_sha_drifts(tmp_path):
    """Mirrors test_clears_render_approval_when_sha_drifts, but for the
    PRE-voice gate: a healed/regenerated manifest.script.json changes its
    bytes -> the stored content_sha no longer matches -> reconcile clears
    the stale voice approval so the worker/UI stop treating it as approved."""
    from studio.dashboard import gates
    con, ep = _mk_chapter(tmp_path, status="rendered")
    current = gates.gate_sha("voice", str(ep))
    con.execute("UPDATE approval SET content_sha=? WHERE gate='voice' "
                "AND chapter_id=310", (current,))
    con.commit()
    (ep / "manifest.script.json").write_text('{"changed": true}')  # script healed
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["voice_approval_cleared"] == 1
    assert _n_approvals(con, "voice") == 0
    # render's approval subject (plan+tts bytes) is untouched by the script edit
    assert _n_approvals(con, "render") == 1


def test_keeps_fresh_voice_approval(tmp_path):
    con, ep = _mk_chapter(tmp_path, status="rendered")
    from studio.dashboard import gates
    current = gates.gate_sha("voice", str(ep))
    con.execute("UPDATE approval SET content_sha=? WHERE gate='voice' "
                "AND chapter_id=310", (current,))
    con.commit()
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["voice_approval_cleared"] == 0
    assert _n_approvals(con, "voice") == 1


def test_reconcile_never_enqueues_a_job(tmp_path):
    """CAUTION from the Task 12 brief: clearing a stale voice (or render)
    approval must NOT enqueue anything — reconcile only repairs state, it
    never drives the pipeline forward on its own."""
    con, ep = _mk_chapter(tmp_path, status="rendered")
    (ep / "manifest.script.json").write_text('{"changed": true}')   # drifts voice
    (ep / "render" / "segment_both.mp4").unlink()                   # drifts render
    reconcile.reconcile_chapter(con, _ch(con))
    n_jobs = con.execute("SELECT COUNT(*) FROM job WHERE chapter_id=310").fetchone()[0]
    assert n_jobs == 0


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


def test_reconcile_series_counts_voice_only_repairs(tmp_path):
    # a chapter whose ONLY drift is a stale voice approval (status, stage_run
    # and render approval all untouched) must still count as repaired at the
    # series level — mirrors the pre-existing render_approval_cleared check.
    from studio.dashboard import gates
    con, ep = _mk_chapter(tmp_path, status="rendered")
    current = gates.gate_sha("voice", str(ep))
    con.execute("UPDATE approval SET content_sha=? WHERE gate='voice' "
                "AND chapter_id=310", (current,))
    con.commit()
    (ep / "manifest.script.json").write_text('{"changed": true}')  # only voice drifts
    totals = reconcile.reconcile_series(con, 1)
    assert totals["chapters_repaired"] == 1


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


# ---- corrupt-marker matrix: 0-byte / truncated / valid, for BOTH roles a
# marker plays -- the status-marker role (derive_status) and the
# stage-artifact role (reconcile_chapter's stage_run pruning). _valid() backs
# both, but only deletion (not corruption) had exercised the pruning role
# before this. ----------------------------------------------------------------

def _write_marker(path, kind):
    if kind == "zero_byte":
        path.write_bytes(b"")
    elif kind == "truncated":
        path.write_text('{"trunc')           # non-empty, unparseable JSON
    else:
        path.write_text(json.dumps({"ok": True}))


@pytest.mark.parametrize("kind,counted", [
    ("zero_byte", False), ("truncated", False), ("valid", True)])
def test_status_marker_corruption_matrix(tmp_path, kind, counted):
    """derive_status's per-marker read (status-marker role): a 0-byte or
    truncated manifest.script.json must NOT promote status to 'scripted';
    a valid one must."""
    reconcile._VALID_CACHE.clear()
    ep = tmp_path / "ep"
    ep.mkdir()
    _write_marker(ep / "manifest.script.json", kind)
    assert (reconcile.derive_status(str(ep)) == "scripted") is counted


@pytest.mark.parametrize("kind,counted", [
    ("zero_byte", False), ("truncated", False), ("valid", True)])
def test_stage_artifact_corruption_matrix(tmp_path, kind, counted):
    """reconcile_chapter's stage_run pruning (stage-artifact role): a 0-byte
    or truncated prep_qa.json (the qa_scan stage-artifact) must prune the
    qa_scan row exactly as a DELETED file would; a valid one must keep it."""
    reconcile._VALID_CACHE.clear()
    con, ep = _mk_chapter(tmp_path)
    _write_marker(ep / "prep_qa.json", kind)
    reconcile.reconcile_chapter(con, _ch(con))
    assert ("qa_scan" in _stages(con)) is counted


def test_skips_chapter_with_cancelling_job(tmp_path):
    """'cancelling' is a LIVE state — the job's child is still running until the
    worker reaps it — so reconcile must not mutate the chapter mid-cancel."""
    con, ep = _mk_chapter(tmp_path, status="rendered")
    (ep / "render" / "segment_both.mp4").unlink()          # drift present
    con.execute("INSERT INTO job (type, state, chapter_id) "
                "VALUES ('prepare','cancelling',310)")
    con.commit()
    out = reconcile.reconcile_chapter(con, _ch(con))
    assert out["status_to"] is None                        # busy — skip
    assert con.execute("SELECT status FROM chapter WHERE id=310"
                       ).fetchone()[0] == "rendered"
