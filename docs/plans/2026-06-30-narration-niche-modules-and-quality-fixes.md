# Narration Niche Modules + Recap-Quality Fixes — Implementation Plan

> **For agentic workers:** REQUIRED: Use subagent-driven-development (if subagents available) or executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the recap a per-series, niche-aware, always-on funny/arrogant narrator and fix the four ch1 quality defects (over-tense delivery, missing quotes, flash montages, flat persona) — without dropping any panel.

**Architecture:** A new per-series **niche** (A/B/C/D = manhwa type) is auto-detected at `add-series` from source genre tags (deterministic keyword map) and cached on the `series` row. Narration keeps ONE always-on base persona; the niche only modulates its *temperature*; multi-niche series blend a primary + secondary. Separately, intensity grading is recalibrated, and pacing gets a per-panel on-screen floor so groups can't flash.

**Tech Stack:** Python 3.12, SQLite (`studio/catalog`), pytest (existing suite ≈1147 tests, `.eval_venv`), Ollama/Gemma (beats), local MLX TTS. No new dependencies.

**Spec:** `docs/plans/specs/2026-06-30-narration-niche-modules-and-quality-fixes-design.md` (read it first).
**Audit (context):** `docs/2026-06-30-manhwa-fresh-vs-current-audit.md`.

**Conventions (from CLAUDE.md):**
- Run tests with `V=.eval_venv/bin/python; $V -m pytest -q`.
- Manifests are the API; `segment_id` (`g####_p##`) must stay byte-identical across script_expander → tts → timeline_planner.
- Edit the plain file, never `*-BAK.py`/`*X.py`.
- `worker.py`/`dashboard` changes need a daemon restart on the Mini; `tools/` + `pipeline.py` are subprocesses (fresh on pull).
- **Test layout (verified):** tools-module tests at top-level `tests/test_*.py`; catalog tests under `tests/catalog/`; source-adapter tests under `tests/sources/` (httpx monkeypatched to return fixture HTML — see `tests/sources/test_asura.py`).

---

## File Structure (decomposition)

**Chunk 1 — Niche detection (acquisition + catalog + classifier):**
- Create `tools/niche_modules.py` — `NICHE_REGISTERS` (4 temperature blocks) + `classify_niche()` + `pick_primary_secondary()` + `register_block()`. Single responsibility: niche classification + register prompt text.
- Create `tests/test_niche_modules.py`.
- Modify `studio/sources/base.py` — extend `SeriesMeta` with `genres`, `synopsis`.
- Modify `studio/sources/asura.py`, `webtoon.py`, `elftoon.py` — `series_meta()` parses genres + synopsis (fail-soft).
- Modify `studio/catalog/db.py` — additive migration (4 new `series` columns) inside `connect()`.
- Modify `studio/catalog/models.py` — `Series` dataclass fields.
- Modify `studio/catalog/repo.py` — persist/select the new fields.
- Modify the add-series flow (`studio/cli.py:cmd_add_series`) — extract `_persist_series` that classifies + stores.
- Create `tests/catalog/test_niche.py`, `tests/sources/test_meta.py`.

**Chunk 2 — Persona engine (narration):** `tools/narration_punchup.py` (invert gate + inject registers), `tools/gemini_narrative_pass.py` (niche into beats prompt + iconic-quote nudge), wiring via `studio/pipeline.py` (`write_series_manifest`) + `studio/cli.py` (`cmd_run` writes it; tools auto-read). *(Detailed in Chunk 2 below.)*

**Chunk 3 — Calibration + pacing:** `tools/panel_understand.py` (intensity), `tools/script_expander.py` + `tools/local_tts_from_manifest.py` (intensity→delivery), `tools/story_group.py` (splitter), `tools/timeline_planner.py` (floor), `tools/prep_qa.py` (`flash_cut` BLOCKING). *(Detailed in Chunk 3 below.)*

**Dependency order:** Chunk 1 → Chunk 2 (Chunk 2 reads `series.niche_*`). Chunk 3 is independent and may proceed in parallel.

---

## Chunk 1: Niche detection

Lands: a series gets `niche_primary`/`niche_secondary` auto-populated at `add-series`; nothing downstream consumes it yet (Chunk 2 does), so this chunk is independently testable via the catalog + classifier.

### Task 1.1: `tools/niche_modules.py` — classifier + registers

**Files:**
- Create: `tools/niche_modules.py`
- Test: `tests/test_niche_modules.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_niche_modules.py
from tools import niche_modules as nm


def _primary(genres, synopsis=""):
    ranked = nm.classify_niche(genres, synopsis)
    return ranked[0][0] if ranked else None


def _with_secondary(genres, synopsis=""):
    """(primary, secondary|None) via the module's shared margin rule."""
    return nm.pick_primary_secondary(nm.classify_niche(genres, synopsis))


def test_pure_genres_map_to_single_niche():
    assert _primary(["Isekai", "Fantasy"]) == "A"
    assert _primary(["System", "Leveling", "Dungeon"]) == "A"
    assert _primary(["Romance", "Drama"]) == "B"
    assert _primary(["Revenge", "Tragedy", "Mature"]) == "C"
    assert _primary(["Comedy", "Slice of Life"]) == "D"


def test_nano_style_murim_action_is_C_primary_A_secondary():
    # A murim/action/revenge manhwa (Nano Machine class): action+mature -> C,
    # martial-arts/murim -> A, with A >= 0.5*C so it surfaces as secondary.
    primary, secondary = _with_secondary(
        ["Action", "Martial Arts", "Murim", "Mature", "Fantasy"])
    assert primary == "C"
    assert secondary == "A"


def test_single_dominant_genre_has_no_secondary():
    primary, secondary = _with_secondary(["Comedy", "Slice of Life", "Comedy"])
    assert primary == "D"
    assert secondary is None


def test_unmappable_or_empty_returns_empty():
    assert nm.classify_niche([], "") == []
    assert nm.classify_niche(["Webtoon", "Full Color"], "") == []  # chrome tags only


def test_synopsis_is_a_weak_tiebreak_only():
    ranked = nm.classify_niche([], "A betrayed swordsman returns for bloody revenge.")
    assert ranked and ranked[0][0] == "C"
    tag_ranked = nm.classify_niche(["Romance"], "revenge revenge revenge")
    assert tag_ranked[0][0] == "B"  # one genre tag outweighs three synopsis hits


def test_registers_exist_and_are_nonempty_for_all_four():
    for key in ("A", "B", "C", "D"):
        assert key in nm.NICHE_REGISTERS
        assert isinstance(nm.NICHE_REGISTERS[key], str)
        assert nm.NICHE_REGISTERS[key].strip()


def test_ranked_scores_are_descending():
    ranked = nm.classify_niche(["Action", "Martial Arts", "Comedy"])
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_pick_primary_secondary_applies_half_margin():
    assert nm.pick_primary_secondary([("C", 5.0), ("A", 4.0)]) == ("C", "A")
    assert nm.pick_primary_secondary([("C", 5.0), ("A", 2.0)]) == ("C", None)  # 2 < 2.5
    assert nm.pick_primary_secondary([("D", 6.0)]) == ("D", None)
    assert nm.pick_primary_secondary([]) == (None, None)


def test_register_block_primary_and_secondary():
    blk = nm.register_block("C", "A")
    assert blk.startswith("PRIMARY") and "SECONDARY" in blk
    assert nm.register_block("", "") == ""        # no niche -> base voice only
    assert "SECONDARY" not in nm.register_block("D", "")  # primary only
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `V=.eval_venv/bin/python; $V -m pytest tests/test_niche_modules.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.niche_modules'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/niche_modules.py
"""Manhwa niche (a/b/c/d) classification + per-niche persona temperature blocks.

The niche is the user's "Manhwa Fresh" Niche Module:
  A = Isekai / Power-Fantasy   B = Romance / Drama
  C = Dark-Action / Revenge    D = Comedy / Slice-of-Life

classify_niche() is deterministic (a weighted keyword map over the source's genre
tags, synopsis as a weak tiebreak) — NO LLM call, so it never reintroduces the
grading non-determinism we are trying to remove. pick_primary_secondary() applies
the 0.5x margin rule in ONE place (used by the add-series caller and the tests).
"""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

# Per-niche keyword weights. Tags are matched case-insensitively as substrings
# against each genre tag (and, at lower weight, the synopsis). Weights:
# 3 = a defining signal, 2 = strong, 1 = supporting. A tag may contribute to more
# than one niche (e.g. "martial arts" -> A power-fantasy).
_NICHE_KEYWORDS: Mapping[str, Mapping[str, int]] = {
    "A": {  # Isekai / Power-Fantasy
        "isekai": 3, "reincarnat": 3, "regression": 3, "regressor": 3,
        "returner": 3, "system": 3, "leveling": 3, "level up": 3, "level-up": 3,
        "dungeon": 2, "tower": 2, "hunter": 2, "awakening": 2,
        "cultivation": 3, "murim": 2, "martial art": 2, "sect": 2,
        "status window": 3, "overpowered": 3, "rpg": 2,
        "constellation": 2, "summoner": 1, "necromancer": 1,
    },
    "B": {  # Romance / Drama
        "romance": 3, "romantic": 3, "love": 2, "drama": 2, "josei": 3,
        "shoujo": 3, "shojo": 3, "otome": 3, "melodrama": 2,
        "marriage": 2, "family drama": 2,
    },
    "C": {  # Dark-Action / Revenge
        "revenge": 3, "vengeance": 3, "action": 3, "thriller": 2, "horror": 2,
        "tragedy": 2, "mature": 2, "gore": 2, "psychological": 2, "war": 1,
        "crime": 2, "mafia": 2, "murder": 2, "villain": 2, "assassin": 2,
        "betrayal": 2, "bloody": 2, "dark": 2, "seinen": 1, "military": 1,
    },
    "D": {  # Comedy / Slice-of-Life
        "comedy": 3, "gag": 3, "parody": 3, "slice of life": 3, "slice-of-life": 3,
        "humor": 2, "humour": 2, "wholesome": 2, "healing": 2, "cooking": 1,
        "everyday": 1,
    },
}

# Tags carrying no genre signal — ignored so they don't dilute scoring.
_CHROME_TAGS = {"webtoon", "manhwa", "manhua", "manga", "full color", "colored",
                "long strip", "adaptation", "web comic", "webcomic"}

_SYNOPSIS_WEIGHT = 0.34  # a synopsis hit is worth ~1/3 of a genre-tag hit (tiebreak)


def _norm(s: str) -> str:
    return (s or "").lower().strip()


def _score_text(text: str, weight_scale: float, acc: dict) -> None:
    low = _norm(text)
    if not low or low in _CHROME_TAGS:
        return
    for niche, kw in _NICHE_KEYWORDS.items():
        for term, w in kw.items():
            if term in low:
                acc[niche] = acc.get(niche, 0.0) + w * weight_scale


def classify_niche(genres: Sequence[str],
                   synopsis: str = "") -> List[Tuple[str, float]]:
    """Return [(niche, score), ...] ranked by score descending; [] if nothing maps."""
    acc: dict = {}
    for tag in (genres or []):
        _score_text(str(tag), 1.0, acc)
    if synopsis:
        _score_text(str(synopsis), _SYNOPSIS_WEIGHT, acc)
    return sorted(
        ((n, s) for n, s in acc.items() if s > 0.0),
        key=lambda kv: (-kv[1], kv[0]),  # score desc, niche letter for stable ties
    )


def pick_primary_secondary(ranked: List[Tuple[str, float]]):
    """Apply the 0.5x margin rule -> (primary|None, secondary|None). ONE home for
    the rule so the add-series caller and the tests agree."""
    if not ranked:
        return (None, None)
    primary, p_score = ranked[0]
    secondary = None
    if len(ranked) > 1 and ranked[1][1] >= 0.5 * p_score:
        secondary = ranked[1][0]
    return (primary, secondary)


# --- per-niche persona TEMPERATURE blocks (injected by narration_punchup, Chunk 2)
# These modulate the ALWAYS-ON base voice; they never replace it.
NICHE_REGISTERS: Mapping[str, str] = {
    "A": (
        "NICHE TEMPERATURE — Isekai/Power-Fantasy (HYPE): lean into the power gap "
        "and the climb; the gap is the point. Cocky understatement about how "
        "outmatched everyone else is; confidence high, jokes welcome. Use "
        "game/system framing ONLY if the world is literally a system; for a "
        "murim/cultivation power-fantasy use realm/cultivation framing instead."
    ),
    "B": (
        "NICHE TEMPERATURE — Romance/Drama (WARM): intimacy over spectacle. Slow on "
        "glances, confessions, betrayals; texture is wry warmth, never a gamer joke "
        "over an emotional beat. Stakes are relational, not combat; keep the wit gentle."
    ),
    "C": (
        "NICHE TEMPERATURE — Dark-Action/Revenge (COLD): controlled menace, short "
        "hard clauses; let cruelty and consequence land. The wit stays but turns "
        "grim/ironic, not light — arrogance reads as intimidation, not comedy. Jokes "
        "thin out; stay restrained on grief and death."
    ),
    "D": (
        "NICHE TEMPERATURE — Comedy/Slice-of-Life (FUNNY): persona-forward, the "
        "funniest register. Frequent asides, absurd contrast, deadpan, audience "
        "winks — humor is the DEFAULT here, not seasoning."
    ),
}


def register_block(primary: str, secondary: str = "") -> str:
    """Compose the register prompt: primary governs; secondary flavors matching beats."""
    primary = (primary or "").upper().strip()
    secondary = (secondary or "").upper().strip()
    if primary not in NICHE_REGISTERS:
        return ""  # no niche -> base voice only
    out = "PRIMARY " + NICHE_REGISTERS[primary]
    if secondary in NICHE_REGISTERS and secondary != primary:
        out += ("\nSECONDARY (flavor only the beats that match it) "
                + NICHE_REGISTERS[secondary])
    return out


def demo() -> None:
    """Runnable self-check: assert representative tag-sets rank correctly."""
    assert classify_niche(["Isekai", "Fantasy"])[0][0] == "A"
    assert classify_niche(["Romance", "Drama"])[0][0] == "B"
    r = classify_niche(["Action", "Martial Arts", "Murim", "Mature"])
    assert pick_primary_secondary(r) == ("C", "A"), r
    assert classify_niche(["Comedy", "Slice of Life"])[0][0] == "D"
    assert classify_niche([], "") == []
    assert pick_primary_secondary([("C", 5.0), ("A", 2.0)]) == ("C", None)
    assert register_block("C", "A").startswith("PRIMARY") and "SECONDARY" in register_block("C", "A")
    print("niche_modules demo OK")


if __name__ == "__main__":
    demo()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `V=.eval_venv/bin/python; $V -m pytest tests/test_niche_modules.py -q && $V tools/niche_modules.py`
Expected: PASS (all tests) and `niche_modules demo OK`.
NOTE: if `test_nano_style_...` fails (A>C, or A<0.5×C so no secondary), tune `_NICHE_KEYWORDS` weights — the *test* encodes the intended ranking; adjust weights, not the test. (Hand-check: the Nano set scores C=5, A=4 → C primary, A secondary, so this should pass as written; real-tag validation is in the Chunk-1 gate.)

- [ ] **Step 5: Commit**

```bash
git add tools/niche_modules.py tests/test_niche_modules.py
git commit -m "feat(niche): deterministic niche classifier + persona temperature registers"
```

### Task 1.2: catalog schema + models + repo

**Files:**
- Modify: `studio/catalog/db.py` — add the 4 columns inside `connect()`, mirroring the existing `ALTER TABLE series ADD COLUMN` idiom at ~`:99-111`.
- Modify: `studio/catalog/models.py` — `Series` dataclass (~`:13-22`).
- Modify: `studio/catalog/repo.py` — `upsert_series` (~`:5-27`, `added_at` is keyword-only after `*`), `get_series` (~`:97-111`), `list_series` (~`:114-124`).
- Test: `tests/catalog/test_niche.py`

NOTE on the initializer: there is **no `init_db`**. The catalog entry point is
`studio.catalog.db.connect(path: Path | str) -> sqlite3.Connection` (`db.py:5`) — it
CREATES + migrates a fresh DB from a *path* and returns the connection. Mirror the
existing pattern in `tests/catalog/test_db.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/catalog/test_niche.py
import sqlite3
from studio.catalog import db as catalog_db   # connect(path) -> new, migrated connection
from studio.catalog import repo


def _fresh_con(tmp_path):
    return catalog_db.connect(tmp_path / "studio.db")


def test_series_table_has_niche_columns(tmp_path):
    con = _fresh_con(tmp_path)
    cols = {row[1] for row in con.execute("PRAGMA table_info(series)")}
    assert {"niche_primary", "niche_secondary", "genres", "synopsis"} <= cols


def test_migration_is_idempotent_and_backcompat(tmp_path):
    # legacy DB WITHOUT the new columns, then let connect() migrate it.
    path = tmp_path / "old.db"
    raw = sqlite3.connect(str(path))
    raw.execute("CREATE TABLE series (id INTEGER PRIMARY KEY, source TEXT, "
                "series_url TEXT, slug TEXT, title TEXT, added_at TEXT, "
                "last_checked TEXT, poll_priority INTEGER DEFAULT 100, "
                "UNIQUE(source, series_url))")
    raw.execute("INSERT INTO series(source, series_url, slug, title, added_at) "
                "VALUES ('asura','u','s','t','now')")
    raw.commit(); raw.close()
    catalog_db.connect(path)          # 1st: ALTER-ADD the columns, must not crash
    con = catalog_db.connect(path)    # 2nd: idempotent, must not crash
    cols = {row[1] for row in con.execute("PRAGMA table_info(series)")}
    assert {"niche_primary", "niche_secondary", "genres", "synopsis"} <= cols
    assert con.execute("SELECT title FROM series WHERE id=1").fetchone()[0] == "t"


def test_upsert_and_get_roundtrip_niche(tmp_path):
    con = _fresh_con(tmp_path)
    sid = repo.upsert_series(con, source="asura", series_url="u", slug="s",
                             title="t", added_at="now",
                             niche_primary="C", niche_secondary="A",
                             genres="Action, Martial Arts", synopsis="syn")
    s = repo.get_series(con, sid)
    assert s.niche_primary == "C"
    assert s.niche_secondary == "A"
    assert s.genres == "Action, Martial Arts"
    assert s.synopsis == "syn"


def test_upsert_does_not_blank_existing_niche_on_metaless_redip(tmp_path):
    con = _fresh_con(tmp_path)
    repo.upsert_series(con, source="asura", series_url="u", slug="s", title="t",
                       added_at="now", niche_primary="C", genres="Action")
    # re-discovery with no metadata must NOT wipe the stored niche (COALESCE)
    sid = repo.upsert_series(con, source="asura", series_url="u", slug="s",
                             title="t2", added_at="now")
    s = repo.get_series(con, sid)
    assert s.niche_primary == "C"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `V=.eval_venv/bin/python; $V -m pytest tests/catalog/test_niche.py -q`
Expected: FAIL — missing columns / `upsert_series` rejects the new kwargs / `Series` has no `niche_primary`.

- [ ] **Step 3: Write the implementation**

In `studio/catalog/db.py`, inside `connect()` after the `series` create and alongside the existing `scols = {r[1] for r in con.execute("PRAGMA table_info(series)")}` block (~`:99-111`), add:

```python
for _col, _typ in (("niche_primary", "TEXT"), ("niche_secondary", "TEXT"),
                   ("genres", "TEXT"), ("synopsis", "TEXT")):
    if _col not in scols:
        con.execute(f"ALTER TABLE series ADD COLUMN {_col} {_typ}")
con.commit()
```
(Reuse the existing `scols` set if it's already computed just above; otherwise compute it as the existing code does.)

In `studio/catalog/models.py`, extend `Series` with four defaulted fields:

```python
    niche_primary: str | None = None
    niche_secondary: str | None = None
    genres: str | None = None
    synopsis: str | None = None
```

In `studio/catalog/repo.py`:
- `upsert_series(...)` gains optional kwargs `niche_primary=None, niche_secondary=None, genres=None, synopsis=None`; include them in the INSERT column list/values and in `ON CONFLICT(...) DO UPDATE SET` using `COALESCE(excluded.<col>, series.<col>)` so a metadata-less re-discovery never blanks a stored niche.
- `get_series` / `list_series` add the 4 columns to their SELECT (keep SELECT order aligned with the `Series(...)` constructor) and pass them through.

(Write the exact SQL by following the existing statements in the file.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `V=.eval_venv/bin/python; $V -m pytest tests/catalog/test_niche.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add studio/catalog/db.py studio/catalog/models.py studio/catalog/repo.py tests/catalog/test_niche.py
git commit -m "feat(catalog): niche/genres/synopsis columns on series (additive migration)"
```

### Task 1.3: `SeriesMeta` + adapters parse genres/synopsis (fail-soft)

**Files:**
- Modify: `studio/sources/base.py` (`SeriesMeta`, ~`:40-47`).
- Modify: `studio/sources/asura.py`, `studio/sources/elftoon.py` (`series_meta()` + a module-level `_parse_genres`/`_parse_synopsis`).
- Modify: `studio/sources/webtoon.py` (`series_meta()` builds from gallery-dl `-j` entry metadata at ~`:139-158`; expose genres if present, else `()` — fail-soft).
- Test: `tests/sources/test_meta.py`
- Fixtures (under `tests/sources/fixtures/`):
  - `elftoon_series.html` — **already exists** and contains the genre block → use as-is for the elftoon parse test.
  - `asura_series.html` — **exists but is a 17-line stub trimmed to the chapter list with NO genre block** → re-capture or extend it to include the genres/synopsis markup before the asura parse test can pass.
  - webtoon — no HTML series fixture (JSON path); test the `()` fail-soft fallback only.

- [ ] **Step 1: Write the failing test**

```python
# tests/sources/test_meta.py
from pathlib import Path
from studio.sources.base import SeriesMeta
from studio.sources import asura, elftoon

FIXTURES = Path(__file__).parent / "fixtures"


def test_seriesmeta_defaults_are_safe():
    m = SeriesMeta(source="asura", series_url="u", title="t", slug="s")
    assert m.genres == ()       # default empty tuple, never None
    assert m.synopsis == ""


def test_elftoon_parse_genres_from_fixture():
    html = (FIXTURES / "elftoon_series.html").read_text(encoding="utf-8")
    genres = elftoon._parse_genres(html)
    assert isinstance(genres, tuple) and len(genres) >= 1
    assert any("action" in g.lower() for g in genres)  # fixture is an action title


def test_asura_parse_genres_from_fixture():
    # requires the extended asura_series.html fixture (see Files note)
    html = (FIXTURES / "asura_series.html").read_text(encoding="utf-8")
    genres = asura._parse_genres(html)
    assert isinstance(genres, tuple) and len(genres) >= 1


def test_parse_genres_failsoft_on_garbage():
    assert asura._parse_genres("<html>no genres here</html>") == ()
    assert elftoon._parse_genres("") == ()
```

- [ ] **Step 2: Run to verify it fails**

Run: `V=.eval_venv/bin/python; $V -m pytest tests/sources/test_meta.py -q`
Expected: FAIL — `SeriesMeta` has no `genres`; `asura._parse_genres` / `elftoon._parse_genres` not defined.

- [ ] **Step 3: Implement**

`studio/sources/base.py`:

```python
@dataclass(frozen=True)
class SeriesMeta:
    source: str
    series_url: str
    title: str
    slug: str
    genres: tuple[str, ...] = ()
    synopsis: str = ""
```

For asura + elftoon, add module-level pure parsers and call them from `series_meta()`:

```python
def _parse_genres(html: str) -> tuple[str, ...]:
    try:
        tree = HTMLParser(html)          # selectolax, already imported in these adapters
        nodes = tree.css("<the genre-tag selector for this site>")
        return tuple(n.text(strip=True) for n in nodes if n.text(strip=True))
    except Exception:
        return ()                        # fail-soft: markup churn must not break discovery


def _parse_synopsis(html: str) -> str:
    try:
        node = HTMLParser(html).css_first("<the synopsis selector>")
        return node.text(strip=True) if node else ""
    except Exception:
        return ""
```
Determine the exact CSS selectors from the fixtures (`elftoon_series.html` already has the genre block; extend `asura_series.html` first). Then in each `series_meta()`, fetch the page HTML once and pass it to both parsers, populating the new `SeriesMeta` fields. For `webtoon.py`, read genres from the gallery-dl entry dict if the key exists, else `()`.

- [ ] **Step 4: Run to verify it passes**

Run: `V=.eval_venv/bin/python; $V -m pytest tests/sources/test_meta.py -q`
Expected: PASS.

- [ ] **Step 5: Live smoke (manual, non-gating)**

Run live `series_meta()` for one tracked series per source; eyeball that `genres` is non-empty where the page has tags (asura/elftoon should; webtoon may be empty — acceptable, falls back to default voice). Record results in the commit message.

- [ ] **Step 6: Commit**

```bash
git add studio/sources/base.py studio/sources/asura.py studio/sources/webtoon.py studio/sources/elftoon.py tests/sources/test_meta.py tests/sources/fixtures/
git commit -m "feat(sources): parse genres+synopsis into SeriesMeta (fail-soft)"
```

### Task 1.4: wire add-series → classify → store

**Files:**
- Modify: `studio/cli.py` — `cmd_add_series` (~`:101-118`; `series_meta()` at `:104`, `upsert_series(...)` at `:106-113`). Extract a testable `_persist_series(con, meta) -> int`.
- Test: extend `tests/catalog/test_niche.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/catalog/test_niche.py
from studio.sources.base import SeriesMeta
from studio import cli


def test_persist_series_classifies_and_stores_niche(tmp_path):
    con = catalog_db.connect(tmp_path / "studio.db")
    # Nano-class tags -> C primary + A secondary (proven set from Task 1.1)
    meta = SeriesMeta(source="asura", series_url="u", title="Nano", slug="nano",
                      genres=("Action", "Martial Arts", "Murim", "Mature", "Fantasy"),
                      synopsis="A murim revenge story.")
    sid = cli._persist_series(con, meta)
    s = repo.get_series(con, sid)
    assert s.niche_primary == "C"
    assert s.niche_secondary == "A"
    assert "Action" in (s.genres or "")   # raw tags stored for audit
```

(Build `SeriesMeta` directly and call `_persist_series` — no stub adapter / monkeypatch needed once the function is extracted.)

- [ ] **Step 2: Run to verify it fails**

Run: `V=.eval_venv/bin/python; $V -m pytest tests/catalog/test_niche.py::test_persist_series_classifies_and_stores_niche -q`
Expected: FAIL — `cli._persist_series` not defined.

- [ ] **Step 3: Implement** — extract `_persist_series` from `cmd_add_series` and call it there:

```python
# studio/cli.py
from tools.niche_modules import classify_niche, pick_primary_secondary

def _persist_series(con, meta) -> int:
    primary, secondary = pick_primary_secondary(classify_niche(meta.genres, meta.synopsis))
    return repo.upsert_series(   # cli.py imports `from studio.catalog import repo`
        con, source=meta.source, series_url=meta.series_url, slug=meta.slug,
        title=meta.title, added_at=_now_iso(),     # reuse cmd_add_series' existing timestamp call
        niche_primary=primary, niche_secondary=secondary,
        genres=", ".join(meta.genres), synopsis=meta.synopsis,
    )
```
`cmd_add_series` then calls `sid = _persist_series(con, meta)` in place of its inline `upsert_series(...)`. (Match the real `added_at`/timestamp expression already used in `cmd_add_series`.)

- [ ] **Step 4: Run to verify it passes**

Run: `V=.eval_venv/bin/python; $V -m pytest tests/catalog/test_niche.py -q`
Expected: PASS.

- [ ] **Step 5: Backfill note + commit**

Already-tracked series have null niche until re-added. Provide a one-off script (NOT a migration — it's network-dependent): for each series, `series_meta()` + `classify_niche` + `UPDATE series SET niche_primary=?, niche_secondary=?, genres=?, synopsis=?`. Then:

```bash
git add studio/cli.py tests/catalog/test_niche.py
git commit -m "feat(add-series): classify + store per-series niche from source genres"
```

### Chunk 1 done — gate

- [ ] Full suite green: `V=.eval_venv/bin/python; $V -m pytest -q`
- [ ] `classify_niche` validated against a REAL scraped tag-set for Nano Machine (resolves to C primary + A secondary, or weights tuned until it does).

---

## Chunk 2: Persona engine

Lands: the always-on funny/arrogant base voice (persona is the default, not "occasional seasoning"), modulated by the per-series niche register, reaching narration in both the pipeline (prepare) and worker (regen) paths via a `manifest.series.json` in the episode dir. Fixes defect #4 (flat/no persona) and supports defect #2 (quotes). Depends on Chunk 1 (reads `series.niche_*`).

**Mechanism (niche → narration):** the niche travels in `<ep_dir>/manifest.series.json`
(matching the existing `manifest.cast.json` / `manifest.story.json` pattern), written by
**`cli.cmd_run`** — the single writer (it has `con`; the worker's prepare and manual runs both
route through `studio run` = `cmd_run`). The narration tools **auto-read** it from the episode
dir (`narration_punchup` via `--episode-dir`, `gemini_narrative_pass` via `dirname(--out)`), so the
prepare path AND the worker regen path (`_regen_flagged`, which passes `--episode-dir`/`--out`
into the same `ep`) both pick it up with **no `worker.py` change and no daemon restart**. This
sidesteps `pipeline._stage_beated(ep_dir, cfg)` (`:217`) having no DB handle.

### Task 2.1: invert the persona gate (`CINEMATIC_RULES`)

**Files:**
- Modify: `tools/narration_punchup.py` — `CINEMATIC_RULES` constant (~`:203-228`).
- Test: `tests/test_narration_punchup_persona.py`

- [ ] **Step 1: Write the failing test** (guards the inversion at the prompt-contract level)

```python
# tests/test_narration_punchup_persona.py
import tools.narration_punchup as np


def test_cinematic_rules_persona_is_default_not_seasoning():
    rules = np.CINEMATIC_RULES.lower()
    # the OLD gate (persona off on dramatic beats) must be gone:
    assert "occasional seasoning" not in rules
    assert "never the default" not in rules
    assert "purely cinematic" not in rules
    # the NEW contract (voice always on; gravity only drops jokes) must be present:
    assert "always on" in rules
    assert "drop the jokes" in rules or "drops the jokes" in rules
    # grounding guardrails retained:
    assert "weather" in rules and "caption" in rules
```

- [ ] **Step 2: Run to verify it fails** — `$V -m pytest tests/test_narration_punchup_persona.py -q` → FAIL (old phrases present, new absent).

- [ ] **Step 3: Implement** — replace the `CINEMATIC_RULES` string (~`:203-228`) with:

```python
CINEMATIC_RULES = """THE CHANNEL VOICE IS THE BASELINE — write EVERY line in the
persona: internet-native, dry, confident, a little arrogant — a sharp friend
recapping the story, not a movie trailer narrator. This voice is ALWAYS ON, even on
grave beats; it never switches off. Use the DRAMATIC/CONNECTIVE/COMIC tag ONLY to set
the TEMPERATURE of that voice, never to remove it:
- DRAMATIC (intense/explosive, somber, tragic, danger): keep the voice and its
  confidence, but DROP THE JOKES — no winks or deflating asides; let the menace,
  stakes, and consequence land in the same dry, characterful voice.
- CONNECTIVE / mundane-aside: the voice runs warm and witty — this is where asides,
  light hyperbole, and intimate stand-ins ("our guy"/"our boy") land most.
- COMIC (mockery, humiliation, a visual gag, a face-slap): the beat is already a joke
  — add ONE sharp recap-channel punch so it lands. The punch must be clearly
  figurative/framing, never a new story event.
Cinematic phrasing (strong verbs, rhythm, stakes) is the floor for EVERY line; it does
NOT mean adding weather, lighting, hair, mist, or trailer-grade atmosphere the viewer
can already see. The NICHE TEMPERATURE block (when present) further tunes how
hot/cold/funny this voice runs.
STORY CAPTIONS / narration-box text: WEAVE them into the line in the story's own
first-person voice — you MAY rephrase for flow, but keep their MEANING and any key
phrase, and never read a caption robotically as a bare standalone fragment.
Keep every grounding rule: no invented facts, cast names verbatim, caption meaning
preserved, mood tags preserved, no chrome."""
```

NOTE: `classify_beats` (`:156`) is UNCHANGED — it still returns DRAMATIC/CONNECTIVE/COMIC; only the *meaning* the prompt assigns to those labels changes (temperature, not on/off). Leave `_comic_cue_score`/`classify_panel_lines` as-is.

- [ ] **Step 4: Run to verify it passes** — `$V -m pytest tests/test_narration_punchup_persona.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/narration_punchup.py tests/test_narration_punchup_persona.py
git commit -m "fix(narration): persona voice is always-on; gravity drops jokes, not the voice"
```

### Task 2.2: inject the niche register into `build_prompt` + auto-read

**Files:**
- Modify: `tools/narration_punchup.py` — `build_prompt` (`:244-261`), the argparse (`:707-737`), the `_prompt` thunk (`:786-790`), and a new `_load_niche(...)` reader.
- Test: extend `tests/test_narration_punchup_persona.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import tools.narration_punchup as np


def test_build_prompt_injects_niche_register():
    lines = [{"group_id": 1, "narration": "x"}]   # build_prompt's cinematic branch needs this schema
    with_niche = np.build_prompt(lines, ["Hero"], "cinematic",
                                 niche="C", niche_secondary="A")
    assert "Dark-Action/Revenge" in with_niche      # primary C register text
    assert "SECONDARY" in with_niche                 # secondary A flavor
    base_only = np.build_prompt(lines, ["Hero"], "cinematic")
    # the C register text proves injection; its ABSENCE proves base-only.
    # (Do NOT assert on "NICHE TEMPERATURE" — CINEMATIC_RULES itself mentions that phrase.)
    assert "Dark-Action/Revenge" not in base_only


def test_load_niche_reads_episode_manifest(tmp_path):
    (tmp_path / "manifest.series.json").write_text(
        json.dumps({"niche_primary": "C", "niche_secondary": "A"}))
    assert np._load_niche(str(tmp_path), "", "") == ("C", "A")
    # explicit args win over the manifest:
    assert np._load_niche(str(tmp_path), "D", "") == ("D", "")
    # missing manifest -> empty (base voice), never crash:
    assert np._load_niche(str(tmp_path / "nope"), "", "") == ("", "")
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`build_prompt` has no `niche` param; `_load_niche` undefined).

- [ ] **Step 3: Implement**

In `build_prompt` (`:244`), add params and inject `register_block` after the genre addon (after `:257`):

```python
def build_prompt(lines, cast_names, humor, genre="", classes=None,
                 story_context="", niche="", niche_secondary=""):
    ...
    addon = GENRE_ADDONS.get(genre_key(genre), "")
    guide = BASE_PERSONA + ("\n\n" + addon if addon else "")
    from niche_modules import register_block          # tools/ is on sys.path
    nblock = register_block(niche, niche_secondary)
    if nblock:
        guide += "\n\n" + nblock
    guide += "\n\n" + RECAP_STYLE_RULES
    ...
```

Add a reader near the other helpers:

```python
def _load_niche(episode_dir, explicit_primary="", explicit_secondary=""):
    """Explicit args win; else read <episode_dir>/manifest.series.json; else ('','')."""
    if explicit_primary:
        return (explicit_primary, explicit_secondary)
    try:
        with open(os.path.join(episode_dir, "manifest.series.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        return (str(d.get("niche_primary") or ""),
                str(d.get("niche_secondary") or ""))
    except Exception:
        return ("", "")
```

In argparse (after `--genre`, ~`:723`):

```python
    ap.add_argument("--niche", default="",
                    help="manhwa niche A/B/C/D; default reads --episode-dir/manifest.series.json")
    ap.add_argument("--niche-secondary", default="")
```

In `main()`, resolve once and thread into the `_prompt` thunk (`:786-790`):

```python
    niche_p, niche_s = _load_niche(args.episode_dir, args.niche, args.niche_secondary)
    def _prompt(batch, index):
        return build_prompt(batch, cast_names, args.humor, genre=args.genre,
                            classes=classes, story_context=story_context,
                            niche=niche_p, niche_secondary=niche_s)
```

- [ ] **Step 4: Run to verify it passes** — `$V -m pytest tests/test_narration_punchup_persona.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/narration_punchup.py tests/test_narration_punchup_persona.py
git commit -m "feat(narration): inject per-series niche temperature register into punchup"
```

### Task 2.3: niche register into `gemini_narrative_pass` (grounded line in-voice)

**Files:**
- Modify: `tools/gemini_narrative_pass.py` — system assembly (`:1114-1115`), argparse (after `--cast`, `:927`), and a `_load_niche` mirror (or import the one from `narration_punchup`/a shared spot — simplest: a tiny local copy keyed off `--out`'s dir).
- Test: `tests/test_gemini_niche.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gemini_niche.py
import tools.gemini_narrative_pass as g


def test_system_prompt_includes_niche_register_when_set():
    # the assembler should append the register block; expose it as a small helper
    sys_with = g._append_niche("BASE SYSTEM", niche="C", niche_secondary="A")
    assert "Dark-Action/Revenge" in sys_with and "SECONDARY" in sys_with
    assert g._append_niche("BASE SYSTEM", "", "") == "BASE SYSTEM"  # no-op when unset
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`_append_niche` undefined).

- [ ] **Step 3: Implement** — add a helper and call it where the system is assembled (`:1114-1115`):

```python
def _append_niche(system, niche="", niche_secondary=""):
    from niche_modules import register_block
    blk = register_block(niche, niche_secondary)
    return system + ("\n\n" + blk if blk else "")
```

```python
    system = (system + "\n\n" + SAFE_NARRATION_RULES + "\n\n"
              + _DIALOGUE_RULE + "\n\n" + RECAP_STYLE_RULES)
    system = _append_niche(system, args.niche, args.niche_secondary)
```

Add args after `--cast` (`:927`, inside `build_arg_parser()` — `main()` calls
`build_arg_parser().parse_args()`, so `args.niche` is available before the `:1114` assembly),
defaulting to a read of `<dir of --out>/manifest.series.json` (mirror `narration_punchup._load_niche`,
resolving the episode dir as `os.path.dirname(args.out)`):

```python
    ap.add_argument("--niche", default="")
    ap.add_argument("--niche-secondary", default="")
```

Resolve in `main()` before assembling `system` (auto-read when args are empty), so both explicit-arg and manifest paths work.

- [ ] **Step 4: Run to verify it passes** — `$V -m pytest tests/test_gemini_niche.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/gemini_narrative_pass.py tests/test_gemini_niche.py
git commit -m "feat(narration): niche register into gemini_narrative_pass system prompt"
```

### Task 2.4: write `manifest.series.json` (single writer = CLI) + auto-read

The niche is written ONCE, by the CLI run path that everything routes through. The tools
auto-read it; no worker edits, no daemon restart.

**Files:**
- Modify: `studio/pipeline.py` — add a module-level `write_series_manifest(ep_dir, niche_primary, niche_secondary)` (pure: dumps `{"niche_primary":..., "niche_secondary":...}` to `<ep_dir>/manifest.series.json`). (It lives here, not in `_stage_beated(ep_dir, cfg)` at `:217`, because that stage has no DB handle.)
- Modify: `studio/cli.py` — `cmd_run` has `con`. Add `s = repo.get_series(con, args.series_id)` before the per-chapter loop; inside the loop, after the skip-guards (~`:215-227`) and before `run_chapter` (`:230`), call `write_series_manifest(ep_dir, s.niche_primary, s.niche_secondary)`. (`cli.py` already imports `repo` from Task 1.4.)
- **NO `studio/worker.py` change.** The worker's prepare shells out to `studio run` (= `cmd_run`), which writes the manifest in a fresh subprocess; `_stage_beated` passes `--episode-dir` to punchup (`pipeline.py:281`) and `--out` (in `ep`) to gemini → both auto-read it. The regen path `_regen_flagged` likewise passes `--episode-dir`/`--out` into the same `ep` → auto-reads the already-persisted file. (An earlier draft wired `_h_prepare`/`_regen_flagged` directly — WRONG: `_h_prepare` narrates inside the `studio run` subprocess before `ep` is resolved, and `_regen_flagged(ep, cfg, project, location, corr_path, env, log)` has no `con`/`series`/`s` in scope → `NameError`.)
- Test: `tests/test_series_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_series_manifest.py
import json
from studio.pipeline import write_series_manifest


def test_write_series_manifest_roundtrip(tmp_path):
    write_series_manifest(str(tmp_path), "C", "A")
    d = json.loads((tmp_path / "manifest.series.json").read_text())
    assert d["niche_primary"] == "C" and d["niche_secondary"] == "A"


def test_write_series_manifest_handles_empty(tmp_path):
    write_series_manifest(str(tmp_path), None, None)  # no niche -> still writes a file
    d = json.loads((tmp_path / "manifest.series.json").read_text())
    assert d["niche_primary"] in ("", None)
```

- [ ] **Step 2: Run to verify it fails** — `$V -m pytest tests/test_series_manifest.py -q` → FAIL (`write_series_manifest` undefined).

- [ ] **Step 3: Implement**

```python
# studio/pipeline.py
def write_series_manifest(ep_dir, niche_primary, niche_secondary):
    import json, os
    with open(os.path.join(ep_dir, "manifest.series.json"), "w", encoding="utf-8") as f:
        json.dump({"niche_primary": niche_primary or "",
                   "niche_secondary": niche_secondary or ""}, f)
```

In `studio/cli.py` `cmd_run`: fetch the series once, then write the manifest per chapter before `run_chapter`:

```python
from studio.pipeline import write_series_manifest
s = repo.get_series(con, args.series_id)          # before the loop (repo imported :28; get_series used :148)
...
for ch in selected:                                # existing loop var (:213)
    ...                                            # existing skip-guards (~:215-227)
    write_series_manifest(ch.ep_dir, s.niche_primary, s.niche_secondary)
    run_chapter(...)                               # existing call (:230)
```
Use **`ch.ep_dir`** (the `Chapter` dataclass attribute) — `cmd_run` has NO local `ep_dir` variable (unlike `cmd_fetch`/`cmd_qa`); do not rebuild the path. Placement after the skip-guards guarantees `ch.ep_dir` is non-None (downloaded+).

- [ ] **Step 4: Run to verify it passes** — `$V -m pytest tests/test_series_manifest.py -q` → PASS. Then full suite: `$V -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add studio/pipeline.py studio/cli.py tests/test_series_manifest.py
git commit -m "feat(cli+pipeline): write per-series niche manifest; narration auto-reads it"
```

### Task 2.5: quotes (defect #2) — regression guard + nudge + diagnostic

The fragment/quote code is **already correct** (`sfx_scrub.is_fragment_quote` keeps "Kill him!", drops only trailing-off "Ancestor...?"). So this task (a) codifies that as a regression guard, (b) adds a light quoting nudge to the persona so gemma actually produces iconic quotes, and (c) a diagnostic on real output. No detector rewrite unless the diagnostic proves otherwise.

**Files:**
- Test: `tests/test_quote_survival.py`
- Modify (nudge): `tools/gemini_narrative_pass.py` `_DIALOGUE_RULE` (`:155`) — the BEATS pass, where quotes are actually produced. (NOT `BASE_PERSONA`/punchup: punchup runs `validate_line(..., forbid_quotes=True)` at `:450,625` and rejects any quoted rewrite, so a nudge there can't ship a quote.)

- [ ] **Step 1: Write the regression-guard test**

```python
# tests/test_quote_survival.py
from tools.sfx_scrub import is_droppable_quote, scrub_sfx_quotes


def test_iconic_short_quotes_survive():
    for q in ("Kill him!", "Ancestor!", "Damn you.", "I can't move."):
        assert not is_droppable_quote(q), q
    # garble / trailing-off still dropped:
    assert is_droppable_quote("EUAACK...!! ACK!!!")
    assert is_droppable_quote("Ancestor...?")
    # a kept quote stays in the line:
    assert "Kill him" in scrub_sfx_quotes('The order rang out: "Kill him!"')
```

- [ ] **Step 2: Run** — `$V -m pytest tests/test_quote_survival.py -q`. Expected: PASS as-is (codifies current behavior). If any iconic case FAILS, fix the predicate in `sfx_scrub.py` minimally and re-run.

- [ ] **Step 3: Nudge the BEATS pass to quote** — append ONE clause to `_DIALOGUE_RULE`
  (`gemini_narrative_pass.py:155`), which already invites quotes. Keep the existing
  "PARAPHRASE"/"onomatopoeia"/"fragment" phrasing intact (existing tests assert on it — add, don't
  replace), e.g. append: `"When a real line is short and iconic — a threat, a taunt, a name — prefer
  QUOTING it (clean sentence case, attributed) over paraphrasing."` The grounded quoted line then
  survives punchup via its fallback-to-original. (No new assertion beyond the existing
  `_DIALOGUE_RULE` tests; the real proof is the Step 4 diagnostic.)

- [ ] **Step 4: Diagnostic on real output (non-gating)** — after a re-narration of a known chapter (Chunk 2 deployed), inspect `manifest.beats.json` for `panel_narration[].line` containing quotes:

Run: `V=.eval_venv/bin/python; $V -c "import json,sys; b=json.load(open(sys.argv[1])); ls=[p['line'] for be in b['beats'] for p in be.get('panel_narration',[])]; q=[l for l in ls if '\"' in l or chr(8220) in l]; print(len(q),'quoted of',len(ls)); print(q[:8])" <ep>/manifest.beats.json`
Expected: > 0 quoted lines on a dialogue-heavy chapter. If 0, the gap is gemma compliance → strengthen the nudge / `_DIALOGUE_RULE`, not the scrub.

- [ ] **Step 5: Commit**

```bash
git add tools/gemini_narrative_pass.py tests/test_quote_survival.py
git commit -m "fix(narration): quote-survival guard + beats-side iconic-quote nudge (defect #2)"
```

### Chunk 2 done — gate

- [ ] Full suite green: `$V -m pytest -q`
- [ ] Deploy note: all Chunk-2 sites are subprocesses (`narration_punchup.py`/`gemini_narrative_pass.py`/`pipeline.py`/`cli.py`) — **fresh on `git pull`, NO daemon restart** (the niche mechanism does not touch `worker.py`; the worker's prepare shells to `studio run` = `cmd_run`, which writes the manifest in a fresh subprocess).
- [ ] Manual: a re-narrated chapter (delete `manifest.panels.understood.json` / reset to `visioned`) carries the niche voice — verify by listening, not JSON (acceptance §6.2).

---

## Chunk 3: Calibration + pacing

Lands: intensity is graded/delivered less hot (defect #1), big groups split so narration scales with panels and no panel flashes sub-floor (defect #3), with a BLOCKING gate so a flash montage can never ship. Independent of Chunks 1–2 (no niche dependency).

### Task 3.1: intensity recalibration (defect #1)

Two levers: recalibrate the grader (fewer `intense`) and soften the intensity→mood-tag escalation. (A third, optional, lever — the mood→exaggeration curve — is in Step 6, used only if needed.)

**Files:**
- Modify: `tools/panel_understand.py` — the intensity guidance line (`:61`).
- Modify: `tools/script_expander.py` — `_escalate_tag_for_intensity` (`:746-756`).
- Test: `tests/test_intensity_calibration.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_intensity_calibration.py
import tools.script_expander as se
import tools.panel_understand as pu


def test_intense_panel_no_longer_auto_escalates_to_tense():
    # rank 2 = "intense": a single intense panel must NOT force the beat's mood up
    assert se._escalate_tag_for_intensity("serious", 2) == "serious"
    # rank 3 = "explosive": a genuine peak still escalates
    assert se._escalate_tag_for_intensity("serious", 3) == "excited"
    # non-escalatable tags untouched at any rank
    assert se._escalate_tag_for_intensity("whisper", 3) == "whisper"


def test_panel_understand_prompt_reserves_high_intensity_for_peaks():
    # the SYSTEM prompt must instruct reserving intense/explosive for real peaks
    prompt = pu._build_system_prompt() if hasattr(pu, "_build_system_prompt") else pu.SYSTEM
    low = prompt.lower()
    assert "reserve" in low and "intense" in low and "peak" in low
```

(If `panel_understand` exposes the SYSTEM prompt by a different name, point the test at the real constant/function — the prompt is assembled at `:49-94`.)

- [ ] **Step 2: Run to verify it fails** — `$V -m pytest tests/test_intensity_calibration.py -q` → FAIL (rank-2 currently returns "tense"; prompt lacks the calibration wording).

- [ ] **Step 3a: recalibrate the grader** — in `tools/panel_understand.py:61`, expand the intensity guidance:

```python
"  intensity: calm | tense | intense | explosive. RESERVE 'intense' and "
"'explosive' for genuine PEAKS — a real clash, a shocking reveal, mortal "
"danger. Grade routine action, travel, dialogue, and ordinary reactions "
"(e.g. a stumble or a fall) as 'calm' or 'tense'. Most panels are calm/tense.\n"
```

- [ ] **Step 3b: soften the escalation** — in `tools/script_expander.py:746-756`, drop the rank-2 bump so a lone intense panel no longer forces "tense":

```python
def _escalate_tag_for_intensity(tag: str, rank: int) -> str:
    t = (tag or "serious").strip().lower()
    if t not in _ESCALATABLE_TAGS:
        return t
    if rank >= 3:            # only a genuine explosive peak nudges a neutral tag up
        return "excited"
    return t                 # rank<=2 (incl. 'intense'): keep the keyword-inferred mood
```

NOTE: this changes the rank-2 mapping — `tests/test_verbatim_script.py:96` asserts the OLD
`_escalate_tag_for_intensity("calm", 2) == "tense"` and MUST be updated to `== "calm"` (the
other assertions at `:95,:97,:99-101` stay valid). Grep for any other callers of the old mapping.

- [ ] **Step 4: Run to verify it passes** — `$V -m pytest tests/test_intensity_calibration.py -q` → PASS. Then full suite.

- [ ] **Step 5: Commit**

```bash
git add tools/panel_understand.py tools/script_expander.py tests/test_intensity_calibration.py
git commit -m "fix(narration): reserve intense/explosive for true peaks; stop lone-intense escalation"
```

- [ ] **Step 6 (secondary lever, only if delivery still too hot after a real re-render):** flatten the mood→exaggeration curve in `tools/local_tts_from_manifest.py` — raise the breakpoints in `exaggeration_to_instruction` (`:165-178`) / `exaggeration_to_speed` (`:181-194`) so the top "intense/explosive" tiers (e ≥ 0.85) are reached less readily, and/or lower the high end of `_EMOTION_BY_KEYWORD` (`:90-96`). Gate this on the distribution check below; don't pre-flatten.

### Task 3.2: split big groups so narration scales with panels (defect #3, splitter)

**Files:**
- Modify: `tools/story_group.py` — `DEFAULT_MAX_BEAT_LEN` (`:42`). Call site is already wired: `pipeline.py:211` does NOT pass `--max-beat-len`, so it uses the CLI default `DEFAULT_MAX_BEAT_LEN` (`story_group.py:455`) → `group_panels(max_beat_len=mbl)` → `repair_to_shots` (`:143-179`, splits at `len(cur["scene_files"]) >= limit`, `:171-175`). So lowering the constant takes effect in production.
- Test: extend `tests/test_story_group.py` (it already covers splitting; also UPDATE its existing `assert sg.DEFAULT_MAX_BEAT_LEN == 15` at `:69`).

- [ ] **Step 1: Append the failing tests** to `tests/test_story_group.py`:

```python
def test_oversized_beat_splits_at_cap():
    scene_order = [f"p{i}.jpg" for i in range(12)]
    model_beats = [{"scene_files": scene_order}]   # one 12-panel beat (canonical shape, cf. :47-49)
    shots = sg.repair_to_shots(scene_order, model_beats, max_beat_len=8)
    assert len(shots) >= 2
    assert all(len(s["scene_files"]) <= 8 for s in shots)
    assert sum(len(s["scene_files"]) for s in shots) == 12   # every panel covered once


def test_default_cap_is_tighter():
    assert sg.DEFAULT_MAX_BEAT_LEN == 8
```

(`repair_to_shots` reads NO beat-id key — it uses `enumerate(model_beats)` for the index; the only required input key is `scene_files`. Do NOT add `beat_index`.)

- [ ] **Step 2: Run to verify it fails** — `$V -m pytest tests/test_story_group.py -q` → FAIL (`DEFAULT_MAX_BEAT_LEN == 15`; the existing `:69` assert and the new `test_default_cap_is_tighter` disagree).

- [ ] **Step 3: Implement** — set `DEFAULT_MAX_BEAT_LEN = 8` (`:42`), AND update the existing `tests/test_story_group.py:69` assertion from `== 15` to `== 8`.

- [ ] **Step 4: Run to verify it passes** — `$V -m pytest tests/test_story_group.py -q` → PASS. Then full suite. (Watch for continuity regressions from over-fragmentation; if any, raise to `10` and adjust both asserts.)

- [ ] **Step 5: Commit**

```bash
git add tools/story_group.py tests/test_story_group.py
git commit -m "fix(pacing): tighter beat cap (15->8) so narration scales with panel count"
```

### Task 3.3: per-panel on-screen floor (defect #3, enforcer)

The floor extends a segment to `max(audio, n_kept × floor)` (no panel sub-floor, none dropped). Per spec §3.3 / §10 this is a *generalization* of the planner's existing `max(base_min, audio+pad)` hold — same shape, **no new audio machinery, no cross-segment overlap**, so the `text_sha` gate / `total_drift` QA / renderer are untouched.

**Files:**
- Modify: `tools/timeline_planner.py` — add a pure `_floor_shot_dur` helper + a `floor` kwarg to `build_cuts` (split region `:973-984`), and have the build_cuts caller reassign the default-path `dur` to the floored total so the emit + `time_cursor += float(dur)` (`:1837`) stay aligned. (`:1636` is the unrelated group-mode increment — not this path.)
- Test: `tests/test_timeline_floor.py`

- [ ] **Step 1: Write the failing test** (pure helper — no image I/O)

```python
# tests/test_timeline_floor.py
from tools.timeline_planner import _floor_shot_dur


def test_floor_extends_when_audio_too_short():
    assert _floor_shot_dur(12, 2.5, 1.2) == 12 * 1.2   # 12 panels can't fit in 2.5s -> extend
    assert _floor_shot_dur(2, 10.0, 1.2) == 10.0       # ample audio -> unchanged
    assert _floor_shot_dur(0, 5.0, 1.2) == 5.0         # no panels -> unchanged
    assert _floor_shot_dur(3, 5.0, 0.0) == 5.0         # floor disabled -> unchanged
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`_floor_shot_dur` undefined).

- [ ] **Step 3: Implement**

```python
# tools/timeline_planner.py  (module-level, near build_cuts)
PANEL_FLOOR_SEC = 1.2   # keep == prep_qa flash_cut threshold

def _floor_shot_dur(n_kept: int, shot_dur: float, floor: float) -> float:
    """Extend a segment so each of n_kept panels gets >= floor seconds; never shrink."""
    if n_kept and floor and shot_dur / n_kept < floor:
        return float(n_kept) * float(floor)
    return float(shot_dur)
```

In `build_cuts`, after `k = len(files)` (`:973`, the post-dedup distinct count), floor `shot_dur` before splitting, and add a `floor=0.0` kwarg:

```python
def build_cuts(files, shot_dur, ..., floor: float = 0.0):
    ...
    k = len(files)
    if k == 0:
        return []
    shot_dur = _floor_shot_dur(k, shot_dur, floor)   # <-- extend if too tight
    per = shot_dur / float(k)
    ...
```

At the **default per-panel build_cuts call** — the multi-cut path at `:1764` (NOT the `single_hold` at `:1761`, which is one held panel and can't flash) — pass `floor=PANEL_FLOOR_SEC`. On this path the segment-duration variable is **`dur`** (from `compute_duration_sec` at `:1728`); it flows to the emitted `duration_sec` (`:1813`), `end_sec` (`:1814`), the `cut_gap`/`total_drift` checks, and `time_cursor += float(dur)` (`:1837`). So reassign **`dur`** to the cuts' floored total — and do it **AFTER the per-cut motion loop (`:1776-1795`), just before the item dict (`:1797`)**, because that loop's `cut_dur = float(c.get("duration_sec") or dur)` (`:1789`) falls back to `dur` (cuts carry `"dur"`, not `"duration_sec"`) and must see the per-cut size, not the floored total:

```python
cuts = build_cuts(files, dur, ..., floor=PANEL_FLOOR_SEC)   # the :1764 multi_cut call
... existing per-cut motion loop (:1776-1795) ...
dur = sum(float(c["dur"]) for c in cuts) if cuts else dur     # floored total -> keeps :1813/:1814/:1837 aligned
```

CRITICAL: do **NOT** use `group_clip_dur` — that variable belongs only to the group-mode branch (`:1532-1636`, which `continue`s at `:1637` and never reaches this path); referencing it here is a `NameError`. And leaving `dur` at the old `audio+pad` value while the cuts tile to `k×floor` would trip `cut_gap` (`:1286-1295`) + `total_drift` and desync audio. With `dur` reassigned to the cuts' total, per-segment audio placement is preserved (the spec invariant). `build_cuts`' `len(files)==1` early return (`:952-953`) bypasses the floor — fine (single panels hold for `audio+pad ≥ base_min`, never flash).

- [ ] **Step 4: Run to verify it passes** — `$V -m pytest tests/test_timeline_floor.py -q` → PASS. Then full suite (the existing planner tests exercise the integrated path; fix any that assumed `total == audio` for dense groups — they should now allow `max(audio, k×floor)`).

- [ ] **Step 5: Commit**

```bash
git add tools/timeline_planner.py tests/test_timeline_floor.py
git commit -m "fix(pacing): per-panel on-screen floor (extend segment, never flash, never drop)"
```

### Task 3.4: `flash_cut` → BLOCKING (defect #3, backstop)

After the floor, no cut is sub-1.2s; promoting `flash_cut` to ERROR is the backstop that fails loud if the floor is ever bypassed.

**Files:**
- Modify: `tools/prep_qa.py` — `flash_cut` severity (`:1275`, WARN→ERROR) + a held-card guard.
- Modify: `studio/worker.py` — add `"flash_cut"` to `_CRITICAL_QA_CODES` (`:187-200`). The worker parks on `ERROR codes ∩ _CRITICAL_QA_CODES` → `NonRetryableError` (`:276`,`:285-288`,`:595`); it keys off the CODE SET, not prep_qa's exit code, so **both** edits are required. **This is the one Chunk-3 site needing a daemon restart.**
- Test: extend `tests/test_prep_qa.py` (it has the `plan_flags` harness); update stale fixtures (Step 4).

- [ ] **Step 1: Write the failing test** — mirror the existing `tests/test_prep_qa.py:350-362` (`_item`/`_plan` builders + `plan_flags(plan, clean_files=..., audio_exists=...)`), build a plan with a sub-1.2s cut, and assert the `flash_cut` flag is ERROR:

```python
# add to tests/test_prep_qa.py (uses the existing _item/_plan helpers + the plan_flags entry point)
def test_flash_cut_is_blocking_error():
    # _item(seg, files, dur=...) BUILDS one cut from `files`; dur=0.3 -> a sub-1.2s flash cut.
    plan = _plan([_item("g0001_p01", ["p.jpg"], dur=0.3)])
    flags = prep_qa.plan_flags(plan, clean_files={"p.jpg"},      # suppress incidental missing_file noise
                               audio_exists=lambda p: True)       # audio_exists is a CALLABLE (cf. :357)
    sev = {f["code"]: f["severity"] for f in flags}
    assert sev.get("flash_cut") == prep_qa.ERROR
```

- [ ] **Step 2: Run to verify it fails** — `$V -m pytest tests/test_prep_qa.py -q` → FAIL (`flash_cut` is currently WARN).

- [ ] **Step 3: Implement**
  - `tools/prep_qa.py:1273-1279`: change `WARN` → `ERROR`, AND add a held guard mirroring the sibling `repeat_cut` (`:1280`, which already does `and not c.get("held")`): make the condition `if dur < 1.2 and not c.get("held"):`. (The flash_cut loop currently has NO exempt check, so without this a legitimately-short held card would false-block.)
  - `studio/worker.py:187-200`: add `"flash_cut"` to `_CRITICAL_QA_CODES`.

- [ ] **Step 4: Update stale fixtures** that hardcode `flash_cut` as WARN/non-blocking (they don't hard-break — the worker reads the fixture's severity, not live prep_qa — but become semantically false): `tests/dashboard/test_worker.py:198` (`_qa_error_codes` exclusion) and `:415` ("ordinary WARN ok" autopilot-advances), and `tests/test_narration_heal.py:55` ("other WARN -> skip"). Either switch those fixtures to a different still-WARN code, or add a positive test that a `flash_cut` ERROR now parks autopilot. (`tests/test_prep_qa.py:362` only checks presence → survives.)

- [ ] **Step 5: Run to verify it passes** — `$V -m pytest tests/test_prep_qa.py tests/dashboard/test_worker.py tests/test_narration_heal.py -q` → PASS. Full suite.

- [ ] **Step 6: Commit**

```bash
git add tools/prep_qa.py studio/worker.py tests/test_prep_qa.py tests/dashboard/test_worker.py tests/test_narration_heal.py
git commit -m "fix(qa): flash_cut BLOCKING + non-retryable; held-card guard; update fixtures"
```

### Chunk 3 done — gate

- [ ] Full suite green: `$V -m pytest -q`
- [ ] Deploy note: `panel_understand.py`/`script_expander.py`/`story_group.py`/`timeline_planner.py`/`prep_qa.py`/`local_tts_from_manifest.py` are subprocesses (fresh on pull); **`worker.py` (`_CRITICAL_QA_CODES`) change needs a daemon restart** (`launchctl kickstart -k gui/$(id -u)/com.originpower.worker`).
- [ ] Manual validation on a real chapter (reset to `visioned`, re-prepare): (a) intensity distribution is no longer majority `intense` (re-run the diagnostic that found 57%); (b) the former 12-panels-in-2.5s group now shows each panel ≥1.2s with every panel present; (c) no `flash_cut` in `prep_qa.json`; (d) listen — routine beats are not delivered at climax intensity (acceptance §6.2–6.3, §6.5).

---

## Plan complete

All three chunks land working, tested software. Execution order: Chunk 1 → Chunk 2 (reads `series.niche_*`); Chunk 3 anytime (independent). Recommended: run each chunk's tasks in order via subagent-driven-development in an isolated worktree, full suite green at each chunk gate, then the manual/listen validations before deploying to the Mini.
