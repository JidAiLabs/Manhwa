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


def test_wrap_with_branding_appends_outro_only_no_intro():
    # channel decision 2026-06-15: NO intro — videos open on the story; outro kept
    segs = ["/a/ch1.mp4", "/a/ch2.mp4"]
    out = bundles.wrap_with_branding(
        segs, "/b/intro.mp4", "/b/outro.mp4",
        exists=lambda p: True)
    assert out == ["/a/ch1.mp4", "/a/ch2.mp4", "/b/outro.mp4"]   # intro ignored
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


def test_create_debut_bundle_once_is_get_or_create(tmp_path):
    from studio.catalog.db import connect
    from studio.dashboard import bundles
    con = connect(tmp_path / "s.db")
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'a','u','s','S','t')")
    for i in (1, 2, 3):
        con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                    "status, updated_at) VALUES (?,1,?,'C','u','beated','t')",
                    (i, i))
    con.commit()
    bid = bundles.create_debut_bundle_once(con, 1, [1, 2, 3], title="Debut")
    assert bid is not None
    assert bundles.bundle_chapters(con, bid) == [1, 2, 3]
    # second call: a bundle already exists -> None, no new bundle
    assert bundles.create_debut_bundle_once(con, 1, [1, 2, 3]) is None
    assert con.execute("SELECT COUNT(*) FROM bundle").fetchone()[0] == 1


def test_create_debut_bundle_once_empty_ids_is_noop(tmp_path):
    from studio.catalog.db import connect
    from studio.dashboard import bundles
    con = connect(tmp_path / "s.db")
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'a','u','s','S','t')")
    con.commit()
    assert bundles.create_debut_bundle_once(con, 1, []) is None
    assert con.execute("SELECT COUNT(*) FROM bundle").fetchone()[0] == 0


# ---- free range selection: already-produced episodes stay selectable -------

def _rendered(con, cid, number, label, ep_dir="/tmp/x"):
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at, season) VALUES "
                "(?,1,?,?,'u','rendered',?,'t',1)", (cid, number, label, ep_dir))


def _seed_series(con):
    con.execute("INSERT OR IGNORE INTO series (id, source, series_url, slug, "
                "title, added_at) VALUES (1,'asura','u','s','S','t')")


def test_rendered_chapters_keeps_already_bundled_and_flags_them(tmp_path):
    """The owner wants to pick any range, including episodes already in a
    video. unbundled_chapters HIDES those ("so a continuation batch can never
    re-select them"), which is why the from-dropdown started at Episode 12.
    rendered_chapters shows everything and labels what is already used, so the
    choice is visible instead of removed."""
    from studio.catalog.db import connect
    from studio.dashboard import bundles
    con = connect(tmp_path / "s.db")
    _seed_series(con)
    for cid, n in ((1, 1), (2, 2), (3, 3)):
        _rendered(con, cid, n, f"Episode {n}")
    con.commit()
    bundles.create_bundle(con, 1, "manual", chapter_ids=[1, 2], title="v1")

    assert [c["id"] for c in bundles.unbundled_chapters(con, 1)] == [3]

    allc = bundles.rendered_chapters(con, 1)
    assert [c["id"] for c in allc] == [1, 2, 3]
    assert [c["bundled"] for c in allc] == [True, True, False]


def test_ids_in_range_can_reuse_produced_episodes(tmp_path):
    from studio.catalog.db import connect
    from studio.dashboard import bundles
    con = connect(tmp_path / "s.db")
    _seed_series(con)
    for cid, n in ((1, 1), (2, 2), (3, 3)):
        _rendered(con, cid, n, f"Episode {n}")
    con.commit()
    bundles.create_bundle(con, 1, "manual", chapter_ids=[1, 2], title="v1")

    assert bundles.unbundled_ids_in_range(con, 1, num_from=1, num_to=3) == [3]
    assert bundles.ids_in_range(con, 1, num_from=1, num_to=3) == [1, 2, 3]


def test_ids_in_range_still_excludes_unrendered(tmp_path):
    """A video is a concat of FINISHED segments — an unrendered chapter has no
    .mp4 to concatenate, so free selection must not reach it."""
    from studio.catalog.db import connect
    from studio.dashboard import bundles
    con = connect(tmp_path / "s.db")
    _seed_series(con)
    _rendered(con, 1, 1, "Episode 1")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at, season) VALUES "
                "(2,1,2,'Episode 2','u','scripted','','t',1)")
    con.commit()
    assert bundles.ids_in_range(con, 1, num_from=1, num_to=2) == [1]


def test_a_chapter_may_belong_to_two_videos(tmp_path):
    """PRIMARY KEY (bundle_id, chapter_id) allows reuse across videos; only a
    duplicate INSIDE one bundle is prevented."""
    from studio.catalog.db import connect
    from studio.dashboard import bundles
    con = connect(tmp_path / "s.db")
    _seed_series(con)
    _rendered(con, 1, 1, "Episode 1")
    con.commit()
    b1 = bundles.create_bundle(con, 1, "manual", chapter_ids=[1], title="v1")
    b2 = bundles.create_bundle(con, 1, "manual", chapter_ids=[1], title="v2")
    assert b1 != b2
    assert bundles.bundle_chapters(con, b1) == [1]
    assert bundles.bundle_chapters(con, b2) == [1]
