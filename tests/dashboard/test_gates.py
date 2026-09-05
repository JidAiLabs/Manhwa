"""Worker-side gates: QA must pass and the user must approve before render."""
from studio.catalog.db import connect
from studio.dashboard import gates


def _con(tmp_path):
    return connect(tmp_path / "s.db")


def _qa(con, chapter_id, ok):
    con.execute("INSERT INTO stage_run (chapter_id, stage, duration_sec, ok) "
                "VALUES (?,?,?,?)", (chapter_id, "qa_scan", 100, int(ok)))
    con.commit()


def test_render_blocked_without_qa(tmp_path):
    con = _con(tmp_path)
    allowed, why = gates.render_allowed(con, 1)
    assert not allowed and "QA" in why


def test_render_blocked_without_approval(tmp_path):
    con = _con(tmp_path)
    _qa(con, 1, ok=True)
    allowed, why = gates.render_allowed(con, 1)
    assert not allowed and "approval" in why


def test_render_allowed_with_qa_and_approval(tmp_path):
    con = _con(tmp_path)
    _qa(con, 1, ok=False)
    _qa(con, 1, ok=True)          # LATEST scan decides
    gates.approve(con, "render", chapter_id=1, note="looks good")
    assert gates.render_allowed(con, 1) == (True, "")


def test_latest_failed_qa_blocks(tmp_path):
    con = _con(tmp_path)
    _qa(con, 1, ok=True)
    _qa(con, 1, ok=False)         # regression after approval
    gates.approve(con, "render", chapter_id=1)
    allowed, why = gates.render_allowed(con, 1)
    assert not allowed and "QA" in why


def test_concat_gate(tmp_path):
    con = _con(tmp_path)
    allowed, why = gates.concat_allowed(con, 5)
    assert not allowed and "approval" in why
    gates.approve(con, "concat", bundle_id=5)
    assert gates.concat_allowed(con, 5) == (True, "")


def test_concat_blocked_when_teaser_planned(tmp_path):
    """A PLANNED-but-unreviewed teaser blocks the concat (don't ship a teaser
    nobody approved); 'approved' or 'declined' both unblock it. concat_allowed
    must stay None-safe when no bundle row exists (test_concat_gate above)."""
    con = _con(tmp_path)
    con.execute("INSERT INTO series (source, series_url, slug, title, added_at) "
                "VALUES ('x','u','s','T', datetime('now'))")
    sid = con.execute("SELECT id FROM series").fetchone()[0]
    con.execute("INSERT INTO bundle (series_id, kind) "
                "VALUES (?, 'manual')", (sid,))
    # the teaser is per MANHWA and opens the series' FIRST video
    con.execute("UPDATE series SET teaser_state='planned' WHERE id=?", (sid,))
    bid = con.execute("SELECT id FROM bundle").fetchone()[0]
    gates.approve(con, "concat", bundle_id=bid)
    assert gates.concat_allowed(con, bid)[0] is False        # 'planned' blocks
    con.execute("UPDATE series SET teaser_state='approved' WHERE id=?", (sid,))
    con.commit()
    assert gates.concat_allowed(con, bid)[0] is True
    con.execute("UPDATE bundle SET teaser_state='declined' WHERE id=?", (bid,))
    con.commit()
    assert gates.concat_allowed(con, bid)[0] is True


def test_voice_gate_requires_narration_approval(tmp_path):
    con = _con(tmp_path)
    allowed, why = gates.voice_allowed(con, 1)
    assert not allowed and "narration" in why
    gates.approve(con, "voice", chapter_id=1, note="read the lines, good")
    assert gates.voice_allowed(con, 1) == (True, "")


def test_voice_approval_invalidated_by_script_edit(tmp_path):
    """content_sha binds the approval to the SCRIPT BYTES: healing/rewriting
    the narration after approval must invalidate the old approval."""
    con = _con(tmp_path)
    ep = tmp_path / "ep"
    ep.mkdir()
    script = ep / "manifest.script.json"
    script.write_text('{"paragraphs": ["original line"]}')
    gates.approve(con, "voice", chapter_id=1,
                 content_sha=gates.gate_sha("voice", ep))
    assert gates.voice_allowed(con, 1, ep) == (True, "")
    script.write_text('{"paragraphs": ["healed line"]}')   # re-narrated
    allowed, why = gates.voice_allowed(con, 1, ep)
    assert not allowed and "narration" in why


def test_render_sha_roundtrip(tmp_path):
    """content_sha binds the render approval to plan+tts_index BYTES together:
    approve with the current sha -> allowed; regenerate either file -> not."""
    con = _con(tmp_path)
    _qa(con, 1, ok=True)
    ep = tmp_path / "ep"
    (ep / "tts").mkdir(parents=True)
    (ep / "render.plan.clean.json").write_text('{"cuts": []}')
    (ep / "tts" / "tts_index.json").write_text('{"clips": []}')
    gates.approve(con, "render", chapter_id=1,
                 content_sha=gates.gate_sha("render", ep))
    assert gates.render_allowed(con, 1, ep) == (True, "")
    (ep / "tts" / "tts_index.json").write_text('{"clips": ["new"]}')  # re-voiced
    allowed, why = gates.render_allowed(con, 1, ep)
    assert not allowed and "approval" in why


def test_legacy_null_sha_approval_allowed_even_with_real_content(tmp_path):
    """An approval predating content_sha (stored NULL) is grandfathered valid
    even when the gate's subject file exists and hashes to something real —
    NULL means 'trust it', not 'compare it'."""
    con = _con(tmp_path)
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "manifest.script.json").write_text('{"paragraphs": ["line"]}')
    gates.approve(con, "voice", chapter_id=1)          # no content_sha -> NULL
    assert gates.voice_allowed(con, 1, ep) == (True, "")


def test_approval_with_sha_invalid_when_subject_file_deleted(tmp_path):
    """A row that DOES carry a sha requires the content to still be there —
    a deleted script is not the content that was approved."""
    con = _con(tmp_path)
    ep = tmp_path / "ep"
    ep.mkdir()
    script = ep / "manifest.script.json"
    script.write_text('{"paragraphs": ["line"]}')
    gates.approve(con, "voice", chapter_id=1,
                 content_sha=gates.gate_sha("voice", ep))
    script.unlink()                                     # approved content gone
    allowed, why = gates.voice_allowed(con, 1, ep)
    assert not allowed and "narration" in why


def test_newest_approval_wins_over_older_mismatch(tmp_path):
    """Two rows, same gate: an older mismatching sha must not shadow a newer
    row that IS valid for current content — newest-by-id is authoritative."""
    con = _con(tmp_path)
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "manifest.script.json").write_text('{"paragraphs": ["line"]}')
    gates.approve(con, "voice", chapter_id=1, content_sha="stale-sha")
    gates.approve(con, "voice", chapter_id=1,
                 content_sha=gates.gate_sha("voice", ep))
    assert gates.voice_allowed(con, 1, ep) == (True, "")


def test_newest_approval_loses_to_newer_mismatch(tmp_path):
    """Inverse: a valid older row must not paper over a newer mismatching one."""
    con = _con(tmp_path)
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "manifest.script.json").write_text('{"paragraphs": ["line"]}')
    gates.approve(con, "voice", chapter_id=1,
                 content_sha=gates.gate_sha("voice", ep))
    gates.approve(con, "voice", chapter_id=1, content_sha="stale-sha")
    allowed, why = gates.voice_allowed(con, 1, ep)
    assert not allowed and "narration" in why


def test_thumbnail_approval_is_series_scoped(tmp_path):
    """One thumbnail per manhwa — approved at the SERIES level, not chapter or
    bundle. Other series stay unapproved, and same-id chapter/bundle approvals
    on a different gate must not leak in."""
    con = _con(tmp_path)
    assert gates.thumbnail_approved(con, 7) is False
    gates.approve(con, "thumbnail", series_id=7, note="this is the one")
    assert gates.thumbnail_approved(con, 7) is True
    assert gates.thumbnail_approved(con, 8) is False        # different series
    gates.approve(con, "render", chapter_id=7)              # same id, other gate
    gates.approve(con, "concat", bundle_id=7)
    assert gates.thumbnail_approved(con, 8) is False        # no cross-talk
