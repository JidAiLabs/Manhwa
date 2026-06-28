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
    con.execute("INSERT INTO bundle (series_id, kind, teaser_state) "
                "VALUES (?, 'manual', 'planned')", (sid,))
    bid = con.execute("SELECT id FROM bundle").fetchone()[0]
    gates.approve(con, "concat", bundle_id=bid)
    assert gates.concat_allowed(con, bid)[0] is False        # 'planned' blocks
    con.execute("UPDATE bundle SET teaser_state='approved' WHERE id=?", (bid,))
    con.commit()
    assert gates.concat_allowed(con, bid)[0] is True
    con.execute("UPDATE bundle SET teaser_state='declined' WHERE id=?", (bid,))
    con.commit()
    assert gates.concat_allowed(con, bid)[0] is True


def test_teaser_gate_requires_teaser_approval(tmp_path):
    con = _con(tmp_path)
    allowed, why = gates.teaser_allowed(con, 5)
    assert not allowed and "teaser" in why
    gates.approve(con, "teaser", bundle_id=5, note="hook is strong")
    assert gates.teaser_allowed(con, 5) == (True, "")


def test_voice_gate_requires_narration_approval(tmp_path):
    con = _con(tmp_path)
    allowed, why = gates.voice_allowed(con, 1)
    assert not allowed and "narration" in why
    gates.approve(con, "voice", chapter_id=1, note="read the lines, good")
    assert gates.voice_allowed(con, 1) == (True, "")


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
