"""AniList discovery: parse fixture, cache upsert keeps user status, offline-safe."""
import json
from pathlib import Path

from studio.catalog.db import connect
from studio.dashboard import discovery

FIX = json.loads((Path(__file__).parent / "fixtures"
                  / "anilist_trending.json").read_text())


def test_parse_trending_fixture():
    rows = discovery.parse_trending(FIX)
    assert rows[0]["anilist_id"] == 105398
    assert rows[0]["title"] == "Solo Leveling"          # english preferred
    assert rows[1]["title"] == "Mount Hua"              # romaji fallback
    assert rows[0]["trend_score"] == 94 and rows[0]["chapters"] == 201


def test_upsert_preserves_tracked_status(tmp_path):
    con = connect(tmp_path / "s.db")
    rows = discovery.parse_trending(FIX)
    discovery.upsert_discovery(con, rows)
    discovery.mark(con, 105398, "tracked")
    discovery.upsert_discovery(con, rows)               # refresh
    st = con.execute("SELECT status FROM discovery_title WHERE anilist_id=?",
                     (105398,)).fetchone()[0]
    assert st == "tracked"


def test_fetch_offline_keeps_cache(tmp_path):
    con = connect(tmp_path / "s.db")
    discovery.upsert_discovery(con, discovery.parse_trending(FIX))

    class Boom:
        def post(self, *a, **k):
            raise OSError("offline")

    n = discovery.fetch_trending(con, client=Boom())
    assert n == 0
    assert con.execute("SELECT COUNT(*) FROM discovery_title").fetchone()[0] == 2


def test_best_match_fuzzy_and_substring():
    cands = [("Nano Machine", "https://a/nano"),
             ("Nano List", "https://a/nanolist"),
             ("Magic Emperor", "https://a/magic")]
    hit = discovery.best_match("Nano Machine", cands)
    assert hit["url"] == "https://a/nano" and hit["score"] >= 0.9
    # substring tolerance: site appends season/extra words
    hit2 = discovery.best_match("Omniscient Reader",
                                [("Omniscient Reader's Viewpoint", "u")])
    assert hit2 and hit2["score"] >= 0.9
    assert discovery.best_match("Totally Different", cands) is None or \
        discovery.best_match("Totally Different", cands)["score"] < 0.7


def test_parse_ytdlp_and_opportunity():
    payload = {"entries": [
        {"view_count": 534000, "channel": "Magical Manhwa Recaps"},
        {"view_count": 36000, "channel": "Cokie Manhwa"}]}
    cov = discovery.parse_ytdlp_search(payload)
    assert cov["videos"] == 2 and cov["max_views"] == 534000
    saturated = {"trend_score": 90, "meta": {"youtube": cov}}
    thin = {"trend_score": 90, "meta": {"youtube":
                                        {"videos": 1, "max_views": 5000}}}
    assert discovery.opportunity(saturated) is False
    assert discovery.opportunity(thin) is True


def test_meta_merge_preserves_links_across_refresh(tmp_path):
    con = connect(tmp_path / "s.db")
    discovery.upsert_discovery(con, discovery.parse_trending(FIX))
    discovery.link_sources(con, 105398, "Solo Leveling",
                           {"asura": lambda t: [("Solo Leveling", "https://a/sl")]})
    discovery.upsert_discovery(con, discovery.parse_trending(FIX))  # refresh
    row = [r for r in discovery.listing(con) if r["anilist_id"] == 105398][0]
    assert row["meta"]["links"]["asura"]["url"] == "https://a/sl"


def test_scan_with_injected_dependencies(tmp_path, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)  # force yt-dlp path
    con = connect(tmp_path / "s.db")

    class FakeClient:
        def post(self, *a, **k):
            class R:
                def json(self):
                    return FIX
            return R()

    class P:
        stdout = '{"entries": [{"view_count": 7, "channel": "X"}]}'

    n = discovery.scan(con, client=FakeClient(),
                       searchers={"asura": lambda t: [(t, "https://a/x")]},
                       yt_runner=lambda cmd: P(), log=lambda *a: None)
    assert n == 2
    row = discovery.listing(con)[0]
    assert row["meta"]["links"]["asura"]["score"] >= 0.9
    assert row["meta"]["youtube"]["videos"] == 1
