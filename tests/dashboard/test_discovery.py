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
