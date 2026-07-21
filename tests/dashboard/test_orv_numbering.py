"""The one-off ORV renumbering: rows take the number they should have had."""
import pytest

from studio.catalog import fix_orv_numbering as fix
from studio.catalog.db import connect


def _orv(tmp_path, *, empty_third=True, live_job=False):
    con = connect(tmp_path / "s.db")
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'webtoon','u','orv','ORV','t')")
    rows = [
        (1, 0, "Episode 0 (Prologue)", ".../episode-0-prologue",
         "/x/ongoing/orv/Episode_0_Prologue"),
        (2, 1, "Episode 1", ".../episode-0-prologue", "/x/ongoing/orv/Episode_1"),
        (3, 2, "Episode 2", ".../episode-1",
         None if empty_third else "/x/ongoing/orv/Episode_2"),
    ]
    for cid, num, label, url, ep in rows:
        con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                    "status, ep_dir, updated_at) VALUES (?,1,?,?,?,'scripted',"
                    "?,'t')", (cid, num, label, url, ep))
    # real work attached to the two downloaded rows
    for cid in (1, 2):
        con.execute("INSERT INTO stage_run (chapter_id, stage, ok) "
                    "VALUES (?, 'grouped', 1)", (cid,))
    if live_job:
        con.execute("INSERT INTO job (type, chapter_id, state) "
                    "VALUES ('prepare', 1, 'running')")
    con.commit()
    return con


def test_plan_reorders_so_each_target_number_is_free(tmp_path):
    con = _orv(tmp_path)
    plan, problems = fix.build_plan(con, 1)
    assert problems == []
    assert [(cid, target) for cid, _c, target, _d in plan] == \
        [(3, None), (2, 2.0), (1, 1.0)]


def test_apply_pairs_each_row_with_its_own_content(tmp_path):
    con = _orv(tmp_path)
    plan, _ = fix.build_plan(con, 1)
    fix.apply_plan(con, plan)
    got = con.execute("SELECT number, ep_dir FROM chapter WHERE series_id=1 "
                      "ORDER BY number").fetchall()
    assert got == [(1.0, "/x/ongoing/orv/Episode_0_Prologue"),
                   (2.0, "/x/ongoing/orv/Episode_1")]


def test_history_stays_attached_to_its_chapter(tmp_path):
    """The whole point of renumbering ROWS instead of remapping chapter_id:
    every stage_run/approval/job stays bound to the work it describes."""
    con = _orv(tmp_path)
    before = dict(con.execute("SELECT chapter_id, COUNT(*) FROM stage_run "
                              "GROUP BY chapter_id").fetchall())
    fix.apply_plan(con, fix.build_plan(con, 1)[0])
    after = dict(con.execute("SELECT chapter_id, COUNT(*) FROM stage_run "
                             "GROUP BY chapter_id").fetchall())
    assert before == after == {1: 1, 2: 1}


def test_refuses_when_the_third_row_has_content(tmp_path):
    con = _orv(tmp_path, empty_third=False)
    plan, problems = fix.build_plan(con, 1)
    assert any("refusing to delete" in p for p in problems)


def test_refuses_when_the_third_row_has_history(tmp_path):
    con = _orv(tmp_path)
    con.execute("INSERT INTO stage_run (chapter_id, stage, ok) "
                "VALUES (3,'grouped',1)")
    con.commit()
    _plan, problems = fix.build_plan(con, 1)
    assert any("has history" in p for p in problems)


def test_refuses_while_a_job_is_live(tmp_path):
    con = _orv(tmp_path, live_job=True)
    _plan, problems = fix.build_plan(con, 1)
    assert any("live" in p for p in problems)


def test_is_not_rerunnable_once_applied(tmp_path):
    """Guards against double-application, which would corrupt the numbering."""
    con = _orv(tmp_path)
    fix.apply_plan(con, fix.build_plan(con, 1)[0])
    _plan, problems = fix.build_plan(con, 1)
    assert problems and "already been applied" in problems[0]
    assert con.execute("SELECT COUNT(*) FROM chapter").fetchone()[0] == 2


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    con = _orv(tmp_path)
    rc = fix.main(["--db", str(tmp_path / "s.db"), "--series", "1"])
    assert rc == 0 and "dry run" in capsys.readouterr().out
    assert con.execute("SELECT number FROM chapter WHERE id=1"
                       ).fetchone()[0] == 0


def test_cli_refuses_on_an_unexpected_shape(tmp_path, capsys):
    con = _orv(tmp_path)
    con.execute("UPDATE chapter SET number=9 WHERE id=3")
    con.commit()
    rc = fix.main(["--db", str(tmp_path / "s.db"), "--series", "1", "--apply"])
    assert rc == 2
    assert con.execute("SELECT COUNT(*) FROM chapter").fetchone()[0] == 3
