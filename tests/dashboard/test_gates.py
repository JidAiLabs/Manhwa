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


def test_voice_gate_requires_narration_approval(tmp_path):
    con = _con(tmp_path)
    allowed, why = gates.voice_allowed(con, 1)
    assert not allowed and "narration" in why
    gates.approve(con, "voice", chapter_id=1, note="read the lines, good")
    assert gates.voice_allowed(con, 1) == (True, "")
