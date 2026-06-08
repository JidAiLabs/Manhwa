from studio.catalog.db import connect
from studio.catalog import repo

def test_upsert_series_idempotent(tmp_path):
    con = connect(tmp_path/"t.db")
    sid1 = repo.upsert_series(con,"asura","url","slug","Title",added_at="2026-01-01T00:00:00Z")
    sid2 = repo.upsert_series(con,"asura","url","slug","Title",added_at="2026-02-01T00:00:00Z")
    assert sid1 == sid2

def test_status_transition_and_resume(tmp_path):
    con = connect(tmp_path/"t.db")
    sid = repo.upsert_series(con,"a","u","s","t",added_at="t0")
    cid = repo.upsert_chapter(con, sid, 1.0,"Ch 1","curl",updated_at="t0")
    repo.set_chapter_status(con, cid, "downloaded", updated_at="t1")
    assert repo.get_chapter(con, cid).status == "downloaded"
    repo.set_chapter_status(con, cid, "stitched_failed", error="boom", updated_at="t2")
    ch = repo.get_chapter(con, cid)
    assert ch.status == "stitched_failed" and ch.error == "boom"

def test_next_actionable_skips_planned(tmp_path):
    con = connect(tmp_path/"t.db")
    sid = repo.upsert_series(con,"a","u","s","t",added_at="t0")
    c1 = repo.upsert_chapter(con, sid, 1.0,"c1","u1",updated_at="t0")
    repo.set_chapter_status(con, c1, "planned", updated_at="t1")
    c2 = repo.upsert_chapter(con, sid, 2.0,"c2","u2",updated_at="t0")
    assert repo.next_actionable(con, sid).id == c2
