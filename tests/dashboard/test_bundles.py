"""Bundles = season/full/manual chapter runs -> concat plan with exactly one
intro (first segment) and one outro (last)."""
from studio.catalog.db import connect
from studio.dashboard import bundles


def _seed(con):
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','nano','Nano Machine','t')")
    for i, (num, season) in enumerate(
            [(1, 1), (2, 1), (3, 1), (4, 2), (5, 2)], start=1):
        con.execute(
            "INSERT INTO chapter (id, series_id, number, label, url, status,"
            " updated_at, season) VALUES (?,1,?,?,?,'planned','t',?)",
            (i, num, f"Chapter {num}", f"u{num}", season))
    con.commit()


def test_season_bundle_selects_ordered_chapters(tmp_path):
    con = connect(tmp_path / "s.db")
    _seed(con)
    bid = bundles.create_bundle(con, 1, "season", season_no=1,
                                title="Nano Machine — Season 1")
    rows = con.execute("SELECT chapter_id FROM bundle_chapter WHERE "
                       "bundle_id=? ORDER BY position", (bid,)).fetchall()
    assert [r[0] for r in rows] == [1, 2, 3]


def test_full_and_manual_bundles(tmp_path):
    con = connect(tmp_path / "s.db")
    _seed(con)
    full = bundles.create_bundle(con, 1, "full")
    assert con.execute("SELECT COUNT(*) FROM bundle_chapter WHERE bundle_id=?",
                       (full,)).fetchone()[0] == 5
    man = bundles.create_bundle(con, 1, "manual", chapter_range=(2, 4))
    rows = con.execute("SELECT chapter_id FROM bundle_chapter WHERE "
                       "bundle_id=? ORDER BY position", (man,)).fetchall()
    assert [r[0] for r in rows] == [2, 3, 4]


def test_branding_for_position():
    assert bundles.branding_for_position(0, 5) == "intro"
    assert bundles.branding_for_position(4, 5) == "outro"
    assert bundles.branding_for_position(2, 5) == "none"
    assert bundles.branding_for_position(0, 1) == "both"


def test_concat_cmd_and_listfile():
    argv, listfile = bundles.concat_cmd(
        ["/a/seg1.mp4", "/a/seg2.mp4"], "/out/bundle.mp4")
    assert argv[:4] == ["ffmpeg", "-y", "-f", "concat"]
    assert argv[-2:] == ["copy", "/out/bundle.mp4"]
    assert "file '/a/seg1.mp4'" in listfile and "file '/a/seg2.mp4'" in listfile


def test_projected_runtime_uses_plans_with_eta_fallback(tmp_path):
    con = connect(tmp_path / "s.db")
    _seed(con)
    bid = bundles.create_bundle(con, 1, "season", season_no=1)
    durs = {1: 600.0, 2: 540.0}          # ch3 has no plan yet -> ETA seed
    total = bundles.projected_runtime_sec(
        con, bid, plan_loader=lambda cid: durs.get(cid))
    assert total > 600 + 540             # + estimated ch3 + intro/outro


def test_wrap_with_branding_prepends_and_appends_when_present():
    segs = ["/a/ch1.mp4", "/a/ch2.mp4"]
    out = bundles.wrap_with_branding(
        segs, "/b/intro.mp4", "/b/outro.mp4",
        exists=lambda p: True)
    assert out == ["/b/intro.mp4", "/a/ch1.mp4", "/a/ch2.mp4", "/b/outro.mp4"]
    # missing branding files -> plain segments (graceful)
    assert bundles.wrap_with_branding(segs, "/b/i.mp4", "/b/o.mp4",
                                      exists=lambda p: False) == segs


def test_branding_intro_plan_shape():
    plan = bundles.branding_intro_plan("thumb.jpg", 800, 450, intro_dur=7.0)
    item = plan["timeline"][0]
    assert item["branding"] == "intro"
    assert item["cuts"][0]["file"] == "thumb.jpg"
    assert plan["scene_dims"]["thumb.jpg"]["w"] == 800
    assert plan["total_duration_sec"] == item["duration_sec"] > 7.0
