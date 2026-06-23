"""series_median_pages: the yardstick that catches a truncated/rate-limited fetch
(a chapter that downloads only a fraction of its pages) before it ships."""
import os

from studio.catalog.db import connect
from studio.catalog import repo


def _ch(con, num, status, ep):
    con.execute(
        "INSERT INTO chapter (series_id,number,label,url,status,ep_dir,updated_at)"
        " VALUES (1,?,?,?,?,?,'t')", (num, f"Chapter {num}", f"u{num}", status, ep))


def _pages(d, n):
    os.makedirs(d, exist_ok=True)
    for i in range(1, n + 1):
        open(os.path.join(d, f"{i:03d}.jpg"), "w").close()
    return d


def test_series_median_pages(tmp_path):
    con = connect(tmp_path / "s.db")
    con.execute("INSERT INTO series (id,source,series_url,slug,title,added_at) "
                "VALUES (1,'asura','u','nano','Nano','t')")
    # fewer than 3 processed samples -> no yardstick yet
    _ch(con, 1, "rendered", _pages(str(tmp_path / "c1"), 20))
    _ch(con, 2, "rendered", _pages(str(tmp_path / "c2"), 22))
    con.commit()
    assert repo.series_median_pages(con, 1) is None
    # 3+ processed -> median; an unverified 'downloaded' chapter is EXCLUDED so a
    # bad partial can't poison the yardstick
    _ch(con, 3, "rendered", _pages(str(tmp_path / "c3"), 18))
    _ch(con, 4, "downloaded", _pages(str(tmp_path / "c4"), 3))   # excluded
    con.commit()
    med = repo.series_median_pages(con, 1)
    assert med == 20                       # median(20, 22, 18), the 3-pg excluded
    # the guard's verdict: a 3-page chapter is well under 50% of median, a full one isn't
    assert 3 < 0.5 * med
    assert not (20 < 0.5 * med)
