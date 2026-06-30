import sqlite3
from studio.catalog.models import Series, Chapter


def upsert_series(
    con: sqlite3.Connection,
    source: str,
    series_url: str,
    slug: str,
    title: str,
    *,
    added_at: str,
    niche_primary: str | None = None,
    niche_secondary: str | None = None,
    genres: str | None = None,
    synopsis: str | None = None,
) -> int:
    con.execute(
        """
        INSERT INTO series(source, series_url, slug, title, added_at,
                           niche_primary, niche_secondary, genres, synopsis)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, series_url) DO UPDATE SET
          title=excluded.title,
          niche_primary=COALESCE(excluded.niche_primary, series.niche_primary),
          niche_secondary=COALESCE(excluded.niche_secondary, series.niche_secondary),
          genres=COALESCE(excluded.genres, series.genres),
          synopsis=COALESCE(excluded.synopsis, series.synopsis)
        """,
        (source, series_url, slug, title, added_at,
         niche_primary, niche_secondary, genres, synopsis),
    )
    con.commit()
    row = con.execute(
        "SELECT id FROM series WHERE source=? AND series_url=?",
        (source, series_url),
    ).fetchone()
    return row[0]


def upsert_chapter(
    con: sqlite3.Connection,
    series_id: int,
    number: float,
    label: str,
    url: str,
    *,
    updated_at: str,
) -> int:
    con.execute(
        """
        INSERT INTO chapter(series_id, number, label, url, updated_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(series_id, number) DO UPDATE SET
          label=excluded.label,
          url=excluded.url
        """,
        (series_id, number, label, url, updated_at),
    )
    con.commit()
    row = con.execute(
        "SELECT id FROM chapter WHERE series_id=? AND number=?",
        (series_id, number),
    ).fetchone()
    return row[0]


def set_chapter_status(
    con: sqlite3.Connection,
    cid: int,
    status: str,
    *,
    error: str | None = None,
    ep_dir: str | None = None,
    updated_at: str,
) -> None:
    if ep_dir is not None:
        con.execute(
            "UPDATE chapter SET status=?, error=?, ep_dir=?, updated_at=? WHERE id=?",
            (status, error, ep_dir, updated_at, cid),
        )
    else:
        con.execute(
            "UPDATE chapter SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error, updated_at, cid),
        )
    con.commit()


def get_chapter(con: sqlite3.Connection, cid: int) -> Chapter:
    row = con.execute(
        "SELECT id, series_id, number, label, url, status, ep_dir, error, updated_at FROM chapter WHERE id=?",
        (cid,),
    ).fetchone()
    return Chapter(
        id=row[0],
        series_id=row[1],
        number=row[2],
        label=row[3],
        url=row[4],
        status=row[5],
        ep_dir=row[6],
        error=row[7],
        updated_at=row[8],
    )


def get_series(con: sqlite3.Connection, sid: int) -> Series:
    row = con.execute(
        "SELECT id, source, series_url, slug, title, added_at, last_checked, poll_priority, "
        "niche_primary, niche_secondary, genres, synopsis FROM series WHERE id=?",
        (sid,),
    ).fetchone()
    return Series(
        id=row[0],
        source=row[1],
        series_url=row[2],
        slug=row[3],
        title=row[4],
        added_at=row[5],
        last_checked=row[6],
        poll_priority=row[7],
        niche_primary=row[8],
        niche_secondary=row[9],
        genres=row[10],
        synopsis=row[11],
    )


def list_series(con: sqlite3.Connection) -> list[Series]:
    rows = con.execute(
        "SELECT id, source, series_url, slug, title, added_at, last_checked, poll_priority, "
        "niche_primary, niche_secondary, genres, synopsis FROM series"
    ).fetchall()
    return [
        Series(
            id=r[0], source=r[1], series_url=r[2], slug=r[3], title=r[4],
            added_at=r[5], last_checked=r[6], poll_priority=r[7],
            niche_primary=r[8], niche_secondary=r[9], genres=r[10], synopsis=r[11],
        )
        for r in rows
    ]


def list_chapters(con: sqlite3.Connection, series_id: int) -> list[Chapter]:
    rows = con.execute(
        "SELECT id, series_id, number, label, url, status, ep_dir, error, updated_at "
        "FROM chapter WHERE series_id=? ORDER BY number",
        (series_id,),
    ).fetchall()
    return [
        Chapter(
            id=r[0], series_id=r[1], number=r[2], label=r[3], url=r[4],
            status=r[5], ep_dir=r[6], error=r[7], updated_at=r[8],
        )
        for r in rows
    ]


def next_actionable(con: sqlite3.Connection, series_id: int) -> Chapter | None:
    row = con.execute(
        """
        SELECT id, series_id, number, label, url, status, ep_dir, error, updated_at
        FROM chapter
        WHERE series_id=?
          AND status != 'planned'
          AND status NOT LIKE '%_failed'
        ORDER BY number
        LIMIT 1
        """,
        (series_id,),
    ).fetchone()
    if row is None:
        return None
    return Chapter(
        id=row[0], series_id=row[1], number=row[2], label=row[3], url=row[4],
        status=row[5], ep_dir=row[6], error=row[7], updated_at=row[8],
    )
