# Teaser Planner Implementation Plan

> **For agentic workers:** REQUIRED: Use subagent-driven-development (if subagents available) or executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bundle-level arc teaser — a high-stakes window selected per-manhwa from the chapters in a bundle, rendered as a short `teaser.mp4` and prepended to the bundle concat.

**Architecture:** A new `tools/teaser_planner.py` does the novel work (deterministic window scoring over cached `understood.json` + one stub-injectable model call to pick + write narration) and materializes a **synthetic episode dir** (`dist/bundle_<id>/teaser/`) with symlinked scenes + `manifest.{beats,cast,groups,scenes}.json`. A new worker handler `_h_teaser` then runs the existing render/TTS **tools** (script_expander → local_tts_from_manifest → timeline_planner → render_prep → remotion) on that dir — NOT the chapter-keyed `_h_*` handlers — and `_h_concat` prepends the result. A `bundle.teaser_state` column gates concat.

**Tech Stack:** Python 3.12 (`.eval_venv`), SQLite catalog, FastAPI/Jinja dashboard, Remotion render, qwen-mlx TTS. Tests: `.eval_venv/bin/python -m pytest -q`.

**Spec:** `docs/plans/specs/2026-06-28-teaser-planner-implementation-spec.md` (approved). Scope = teaser only; flashback inserts / auto-batcher / tail-handling are deferred.

---

## Conventions for every task

- **Venv:** `V=.eval_venv/bin/python`. Run tests with `$V -m pytest <path> -q`.
- **Tool tests** load the module by path (tools/ is not a package):
  ```python
  import importlib.util
  from pathlib import Path
  _SPEC = importlib.util.spec_from_file_location(
      "teaser_planner", Path(__file__).resolve().parent.parent / "tools" / "teaser_planner.py")
  tp = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(tp)
  ```
- **Studio tests** import normally: `from studio.catalog.db import connect`, `from studio.config import Config, load`.
- **Model calls are injected** as a `model_call(payload) -> dict|None` callable (the `story_group.group_panels(panels, call_fn)` pattern) so tests never hit a real LLM.
- **New tool exposes `build_arg_parser()`** + `main()` so argv is testable without running.
- **Commit after every green step.** Branch is `main` (the repo's normal flow); commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `studio.toml` | `[teaser]` config block | Modify |
| `studio/config.py` | `Config` teaser_* fields + `load()` wiring | Modify |
| `studio/catalog/db.py` | `bundle.teaser_state` column (CREATE + migration) | Modify |
| `tools/teaser_planner.py` | scorer + select/write + synthetic-dir builder + `main()` | **Create** |
| `studio/dashboard/gates.py` | `teaser_allowed` + `concat_allowed` teaser gating | Modify |
| `studio/dashboard/jobs.py` | `LANES` entry for `plan_teaser` | Modify |
| `studio/worker.py` | `_h_teaser` handler + `_h_concat` prepend + `HANDLERS` | Modify |
| `studio/dashboard/app.py` | `/videos` teaser context + plan/approve/decline routes | Modify |
| `studio/dashboard/templates/videos.html` | Plan-teaser button + teaser review card + badge | Modify |
| `tests/test_teaser_planner.py` | scorer + select/write + dir-builder unit tests | **Create** |
| `tests/dashboard/test_gates.py` | teaser gating tests | Modify |
| `tests/test_pipeline.py` or `tests/test_teaser_worker.py` | `_h_teaser`/`_h_concat` prepend tests | **Create** |

---

## Chunk 1: Config + DB foundation

### Task 1: `[teaser]` config block

**Files:**
- Modify: `studio.toml`, `studio/config.py`
- Test: `tests/test_config_teaser.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_teaser.py
from studio.config import load, REPO_ROOT

def test_teaser_config_defaults_and_toml():
    cfg = load(REPO_ROOT / "studio.toml")
    assert cfg.teaser_enabled is True
    assert cfg.teaser_shortlist_n == 4
    assert cfg.teaser_min_panels == 4
    assert cfg.teaser_max_hook_panels == 10
    assert cfg.teaser_max_hook_scan_chapters == 12
    assert cfg.teaser_max_seconds == 90
    assert 0.0 < cfg.teaser_payoff_tail_frac < 1.0
```

- [ ] **Step 2: Run, verify fail** — `$V -m pytest tests/test_config_teaser.py -q` → AttributeError (no `teaser_enabled`).

- [ ] **Step 3: Implement.** In `studio/config.py` add dataclass fields (follow the `punchup`/`_env_bool` idiom, config.py:30/59-68):
```python
    teaser_enabled: bool = False
    teaser_shortlist_n: int = 4
    teaser_min_panels: int = 4
    teaser_max_hook_panels: int = 10
    teaser_max_hook_scan_chapters: int = 12
    teaser_max_seconds: int = 90
    teaser_payoff_tail_frac: float = 0.20
    teaser_model: str = "gemini-2.5-flash"   # mirrors the beats backend (Vertex Gemini / ollama Gemma) — NOT an OpenAI id; the model call is _call_model_with_backoff from gemini_narrative_pass
```
In `load()`, after `t = data.get("tts", {})`, add `tr = data.get("teaser", {})`, and in the `Config(...)` return:
```python
        teaser_enabled=_env_bool("STUDIO_TEASER_ENABLED", bool(tr.get("enabled", False))),
        teaser_shortlist_n=int(os.environ.get("STUDIO_TEASER_SHORTLIST_N") or tr.get("shortlist_n", 4)),
        teaser_min_panels=int(tr.get("min_panels", 4)),
        teaser_max_hook_panels=int(tr.get("max_hook_panels", 10)),
        teaser_max_hook_scan_chapters=int(tr.get("max_hook_scan_chapters", 12)),
        teaser_max_seconds=int(tr.get("max_seconds", 90)),
        teaser_payoff_tail_frac=float(tr.get("payoff_tail_frac", 0.20)),
        teaser_model=(os.environ.get("STUDIO_TEASER_MODEL") or tr.get("model", "gemini-2.5-flash")),
```
In `studio.toml` add:
```toml
[teaser]
enabled = true
shortlist_n = 4
min_panels = 4
max_hook_panels = 10
max_hook_scan_chapters = 12   # cost guard only
max_seconds = 90              # cost guard only
payoff_tail_frac = 0.20       # spoiler guard: never pull from the last 20%
model = "gemini-2.5-flash"    # beats backend model (Vertex/ollama), not OpenAI
```

- [ ] **Step 4: Run, verify pass** — `$V -m pytest tests/test_config_teaser.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add studio/config.py studio.toml tests/test_config_teaser.py && git commit -m "feat(teaser): [teaser] config block"`.

### Task 2: `bundle.teaser_state` column

**Files:**
- Modify: `studio/catalog/db.py`
- Test: `tests/catalog/test_teaser_state.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/catalog/test_teaser_state.py
from studio.catalog.db import connect

def test_bundle_has_teaser_state_default_none(tmp_path):
    con = connect(tmp_path / "s.db")
    # real series columns: (source, series_url, slug, title, added_at NOT NULL, ...)
    con.execute("INSERT INTO series (source, series_url, slug, title, added_at) "
                "VALUES ('x','u','s','T', datetime('now'))")
    sid = con.execute("SELECT id FROM series").fetchone()[0]
    con.execute("INSERT INTO bundle (series_id, kind) VALUES (?, 'manual')", (sid,))
    con.commit()
    row = con.execute("SELECT teaser_state FROM bundle").fetchone()
    assert row[0] == "none"
```
(`series` real columns per db.py:10-20 = `source, series_url, slug, title, added_at(NOT NULL,no default), last_checked, poll_priority` — there is NO `kind`/`url`. Or use `repo.upsert_series(con,'x','u','s','T',added_at=...)`.)

- [ ] **Step 2: Run, verify fail** — `$V -m pytest tests/catalog/test_teaser_state.py -q` → OperationalError (no such column).

- [ ] **Step 3: Implement.** In `studio/catalog/db.py`: (a) add to the inline `CREATE TABLE bundle` after `state TEXT NOT NULL DEFAULT 'collecting',`:
```sql
          teaser_state TEXT NOT NULL DEFAULT 'none',
```
(b) add a migration block before the final `con.commit()` (mirror the `autopilot` precedent, db.py:95-111):
```python
    bcols = {r[1] for r in con.execute("PRAGMA table_info(bundle)")}
    if "teaser_state" not in bcols:
        # arc-teaser sequencing: none|planned|approved|declined
        con.execute("ALTER TABLE bundle ADD COLUMN teaser_state TEXT "
                    "NOT NULL DEFAULT 'none'")
```

- [ ] **Step 4: Run, verify pass** — `$V -m pytest tests/catalog/test_teaser_state.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add studio/catalog/db.py tests/catalog/test_teaser_state.py && git commit -m "feat(teaser): bundle.teaser_state column + migration"`.

---

## Chunk 2: teaser_planner — Stage 1 window scorer (pure)

`tools/teaser_planner.py` data model: a **panel** is a dict
`{chapter_number:int, scene_file:str(abs), description, action, dialogue, panel_kind, intensity, subjects:list}`.
A **window** is `{start:int, end:int, panels:list, score:float, signals:dict}` over a contiguous slice.

### Task 3: panel eligibility + flattening helpers

**Files:** Create `tools/teaser_planner.py`; Create `tests/test_teaser_planner.py`

- [ ] **Step 1: Failing test**
```python
def test_eligible_panels_skips_chrome_empty_error():
    panels = [
        {"scene_file": "a", "panel_kind": "story", "intensity": "calm"},
        {"scene_file": "b", "panel_kind": "chrome", "intensity": "calm"},
        {"scene_file": "c", "panel_kind": "empty", "intensity": "calm"},
        {"scene_file": "d", "panel_kind": "system", "intensity": "tense"},
        {"scene_file": "e", "panel_kind": "story", "error": "parse_failed"},
    ]
    out = tp.eligible_panels(panels)
    assert [p["scene_file"] for p in out] == ["a", "d"]
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `eligible_panels(panels)` — keep `panel_kind in {"story","caption","system"}` and no `error` key. Add module header, `INTENSITY_RANK = {"calm":0,"tense":1,"intense":2,"explosive":3}`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

### Task 4: signal scoring of one window

- [ ] **Step 1: Failing test**
```python
def test_score_window_high_stakes_beats_calm():
    hot = [{"scene_file": f"h{i}", "panel_kind": "story", "intensity": "explosive",
            "description": "the entrance exam begins", "action": "a clan heir humiliates him",
            "dialogue": "you have no badge", "subjects": ["heir","prince"]} for i in range(5)]
    calm = [{"scene_file": f"c{i}", "panel_kind": "story", "intensity": "calm",
             "description": "they eat lunch quietly", "action": "", "dialogue": "",
             "subjects": ["prince"]} for i in range(5)]
    assert tp.score_window(hot)["score"] > tp.score_window(calm)["score"]
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `score_window(panels) -> {"score":float,"signals":dict}`. Module-level keyword/regex sets (no per-series config):
```python
_STAKES_RE = re.compile(r"\b(exam|test|trial|rank|survival|expel|expuls|execut|contract|tournament|duel)\w*", re.I)
_SOCIAL_RE = re.compile(r"\b(humiliat|mock|laugh|badge|token|reject|outcast|disgrace|shame|peasant)\w*", re.I)
_POWER_RE  = re.compile(r"\b(system|status window|skill|rank up|awaken|hidden|impossible|level|power|technique)\w*", re.I)
_ENEMY_RE  = re.compile(r"\b(elder|heir|clan|authority|enemy|assassin|master|commander|villain)\w*", re.I)
```
Score = weighted sum of: stakes/social/power/enemy keyword hits (capped per signal), intensity peak (max INTENSITY_RANK), visual variety (count distinct panel_kind + spread of intensities), minus a clarity penalty when distinct `subjects` across the window exceed ~6. Document weights as module constants (the one calibration knob). Build the scored text from `description + " " + action + " " + dialogue` per panel.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

### Task 5: spoiler guard + window enumeration + shortlist

- [ ] **Step 1: Failing tests**
```python
def test_payoff_tail_excluded():
    seq = [{"scene_file": f"p{i}", "panel_kind": "story", "intensity": "calm",
            "description": "x", "action": "", "dialogue": "", "subjects": []} for i in range(10)]
    wins = tp.score_windows(seq, min_panels=2, max_panels=3, payoff_tail_frac=0.2, shortlist_n=5)
    # last 20% (p8,p9) must not appear in any returned window
    assert all(p["scene_file"] not in ("p8", "p9") for w in wins for p in w["panels"])

def test_windows_respect_max_panels_and_nonoverlap():
    seq = [{"scene_file": f"p{i}", "panel_kind": "story", "intensity": "tense",
            "description": "exam", "action": "fight", "dialogue": "", "subjects": []} for i in range(12)]
    wins = tp.score_windows(seq, min_panels=4, max_panels=10, payoff_tail_frac=0.2, shortlist_n=3)
    assert wins and all(4 <= len(w["panels"]) <= 10 for w in wins)
    # non-overlapping shortlist
    spans = sorted((w["start"], w["end"]) for w in wins)
    assert all(spans[i][1] <= spans[i+1][0] for i in range(len(spans)-1))

def test_no_windows_when_too_few_panels():
    seq = [{"scene_file": "p0", "panel_kind": "story", "intensity": "calm",
            "description": "x", "action": "", "dialogue": "", "subjects": []}]
    assert tp.score_windows(seq, min_panels=4, max_panels=10, payoff_tail_frac=0.2, shortlist_n=3) == []
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `score_windows(panels, *, min_panels, max_panels, payoff_tail_frac, shortlist_n)`:
  - `pool = eligible_panels(panels)`; compute the spoiler-excluded set: indices in the last `payoff_tail_frac` of the FULL sequence, plus the index of the single global max-intensity panel.
  - enumerate contiguous windows size `min_panels..max_panels` that contain no excluded index; `score_window` each.
  - greedily pick top-scoring non-overlapping windows until `shortlist_n` (sort by score desc, skip any overlapping an already-picked span).
  - return `[]` if the eligible pool is shorter than `min_panels`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

## Chunk 3: teaser_planner — Stage 2 select/write + synthetic dir + main

### Task 6: `select_and_write` with stubbed model

**Files:** `tools/teaser_planner.py`, `tests/test_teaser_planner.py`

- [ ] **Step 1: Failing test**
```python
def test_select_and_write_builds_teaser_manifest(tmp_path):
    win = {"start": 0, "end": 4, "score": 9.0, "signals": {},
           "panels": [{"chapter_number": 5, "scene_file": "/abs/ch5/scenes/scene_0007.jpg",
                       "panel_kind": "story", "intensity": "tense",
                       "description": "exam begins", "action": "heir mocks him",
                       "dialogue": "you have no badge", "subjects": ["heir","prince"]}]}
    def stub(payload):
        assert "windows" in payload and "loglines" in payload
        return {"chosen_index": 0,
                "panel_narration": [{"scene_file": "scene_0007.jpg", "line": "The exam begins."}],
                "rewind_line": "But to see how he got here, we go back to the start.",
                "reason": "public test + humiliation", "spoiler_boundary": "no identity reveal"}
    out = tp.select_and_write([win], loglines=["a hunted prince"], model_call=stub)
    assert out["rewind_line"].startswith("But to see")
    assert out["source_chapters"] == [5]
    assert out["panel_narration"][0]["scene_file"] == "scene_0007.jpg"
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `select_and_write(windows, *, loglines, model_call) -> dict|None`:
  - build `payload = {"windows": [<understood text per panel, basenames>], "loglines": loglines}`,
  - `resp = model_call(payload)`; `None`/missing → return `None`,
  - chosen = `windows[resp["chosen_index"]]`,
  - assemble the `manifest.teaser.json` dict: `source_chapters` (sorted unique chapter_number), `scene_files` (basenames in window order), `panel_narration` (from resp, aligned to the window's basenames), `reason`, `rewind_line`, `spoiler_boundary`, `scores`.
  - run the spoiler/fragment post-pass: wrap as `{"beats":[{"panel_narration": pn}]}`, call `recap_style.neutralize_identity_reveal_leaks(beats_obj, cast_obj, vision_by_file)` and `recap_style.repair_spoken_fragments(beats_obj)` (import recap_style by the tools/ sys.path idiom already used by gemini_narrative_pass), then read the repaired `panel_narration` back. (cast_obj/vision_by_file are passed in from `main()`; default `{}`/`{}` keeps the post-pass a safe no-op in this unit test.)
  - return the teaser dict.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

### Task 7: synthetic-dir builder (manifests + scene symlinks)

- [ ] **Step 1: Failing test**
```python
def test_materialize_teaser_dir(tmp_path):
    # a fake source chapter with one scene + a scenes manifest entry
    src = tmp_path / "ch5"; (src / "scenes").mkdir(parents=True)
    (src / "scenes" / "scene_0007.jpg").write_bytes(b"\xff\xd8\xff")  # tiny jpg stub
    (src / "manifest.scenes.json").write_text(json.dumps(
        {"scenes": [{"out_file": "scene_0007.jpg", "box_px_xyxy": [0,0,100,200],
                     "chunk_global_y0": 0, "w": 100, "h": 200}]}))
    teaser = {"source_chapters": [5], "scene_files": ["scene_0007.jpg"],
              "panel_narration": [{"scene_file": "scene_0007.jpg", "line": "The exam begins."}],
              "rewind_line": "...", "reason": "...", "spoiler_boundary": "..."}
    # map each scene_file -> its source ep_dir
    src_of = {"scene_0007.jpg": str(src)}
    out_dir = tmp_path / "teaser"
    tp.materialize_teaser_dir(teaser, src_of, out_dir, cast={"cast": []})
    assert (out_dir / "scenes" / "scene_0007.jpg").exists()           # symlink/copy
    beats = json.loads((out_dir / "manifest.beats.json").read_text())
    assert beats["beats"][0]["panel_narration"][0]["line"] == "The exam begins."
    groups = json.loads((out_dir / "manifest.groups.json").read_text())
    assert groups["shots"][0]["scene_files"] == ["scene_0007.jpg"]
    scenes = json.loads((out_dir / "manifest.scenes.json").read_text())
    assert scenes["scenes"][0]["out_file"] == "scene_0007.jpg"
    assert (out_dir / "manifest.cast.json").exists()
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `materialize_teaser_dir(teaser, src_of, out_dir, cast)`:
  - `mkdir out_dir/scenes`; for each `scene_file`, `os.symlink` the source `<src_ep>/scenes/<scene_file>` into `out_dir/scenes/` (fall back to copy if symlink fails).
  - write `manifest.beats.json` = `{"beats":[{"group_id":1, "scene_files":[...], "panel_narration":[...], "narration": <join of lines>}]}`.
  - write `manifest.groups.json` = `{"shots":[{"shot_id":1, "scene_files":[...], "segment":"present", "arc_label":"teaser"}]}`.
  - write `manifest.scenes.json` = `{"scenes":[<source entry for each scene_file, copied from its source manifest.scenes.json by out_file>]}`. (Read each `src_of[sf]/manifest.scenes.json`, find the entry whose `out_file==sf`, copy it.)
  - write `manifest.cast.json` = the merged `cast` dict.
  - write `manifest.teaser.json` = `teaser`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

### Task 8: `build_arg_parser` + `main()` + cast merge + panel loading

- [ ] **Step 1: Failing test** (argv contract, story_group/panel_narration idiom):
```python
def test_build_arg_parser_required_flags():
    p = tp.build_arg_parser()
    args = p.parse_args(["--bundle-id", "12", "--chapter-dirs", "/a", "/b",
                         "--out-dir", "/o"])
    assert args.bundle_id == 12 and args.chapter_dirs == ["/a", "/b"]

def test_load_bundle_panels_tags_chapter_and_abspath(tmp_path):
    ch = tmp_path / "ch5"; ch.mkdir()
    (ch / "manifest.panels.understood.json").write_text(json.dumps(
        {"panels": [{"scene_file": "scene_0001.jpg", "panel_kind": "story",
                     "intensity": "tense", "description": "d", "action": "a",
                     "dialogue": "", "subjects": []}]}))
    panels = tp.load_bundle_panels([str(ch)])
    assert panels[0]["chapter_number"]  # derived from dir name or order
    assert panels[0]["scene_file"].endswith("ch5/scenes/scene_0001.jpg")
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement:**
  - `load_bundle_panels(chapter_dirs, *, max_scan_chapters=0)`: for the first `max_scan_chapters` dirs (0 = all — wires `teaser_max_hook_scan_chapters`), read `manifest.panels.understood.json`; tag each panel with `chapter_number` (parse from dir name via the `recap_style`-style number regex, fallback to enumerate order) and `scene_file` → abs `<dir>/scenes/<basename>`. Skip dirs missing the understood manifest (log a warning). Return the flattened reading-order list.
  - `load_loglines(chapter_dirs)`: read each `manifest.story.json` `logline` (written by story_group, story_group.py:556-565), return the non-empty list (context for the model call).
  - `merge_cast(chapter_dirs)`: union of `manifest.cast.json` `cast` members across chapters, dedup by `canonical_name`. Returns `{"cast": [...]}`.
  - `build_arg_parser()`: `--bundle-id`(int,req), `--chapter-dirs`(nargs="+",req), `--out-dir`(req), `--backend`(default "vertex"), `--model`, **`--ollama-model`** (default "gemma4:26b" — gemini_narrative_pass.py:855; the ollama path uses this, NOT `--model`), `--project`/`--location`, and `--shortlist-n`/`--min-panels`/`--max-hook-panels`/`--payoff-tail-frac`/`--max-scan-chapters`/`--max-seconds` (cost guards, defaults match config).
  - `main()`: `panels = load_bundle_panels(chapter_dirs, max_scan_chapters=args.max_scan_chapters)`; if `< min_panels` eligible OR `len(chapter_dirs)<2` → print `"[teaser] no teaser"` and `return 0` (writes nothing). Else `score_windows` → if empty, no-teaser. Build the real `model_call` (Vertex/ollama via `_call_model_with_backoff` imported from gemini_narrative_pass — pass `model=args.ollama_model` when `backend=="ollama"`, else `args.model` — with the `TEASER_PROMPT` enforcing the 6 recap rules, uncapped, spoiler-safe, returning the resp schema) → `select_and_write(windows, loglines=load_loglines(chapter_dirs), model_call=...)` → `materialize_teaser_dir(teaser, src_of, out_dir, cast=merge_cast(chapter_dirs))`. Print the out path. `max_seconds` is a reserved soft cap (the narration stays uncapped; a future duration trim may use it — leave it parsed but document it reserved).
  - The TEASER prompt: a module constant `TEASER_PROMPT` instructing: pick the strongest window by index, write rolling per-panel narration under the recap rules, a strong uncapped cold-open first line, a `rewind_line`, `reason`, `spoiler_boundary`; never name an unrevealed identity; never reference events past the window.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

## Chunk 4: worker + gates + jobs wiring

### Task 9: `teaser_allowed` gate + `concat_allowed` teaser gating

**Files:** `studio/dashboard/gates.py`, `tests/dashboard/test_gates.py`

- [ ] **Step 1: Failing test**
```python
def test_concat_blocked_when_teaser_planned(tmp_path):
    con = connect(tmp_path / "s.db")
    con.execute("INSERT INTO series (source, series_url, slug, title, added_at) "
                "VALUES ('x','u','s','T', datetime('now'))")
    sid = con.execute("SELECT id FROM series").fetchone()[0]
    con.execute("INSERT INTO bundle (series_id, kind, teaser_state) VALUES (?, 'manual', 'planned')", (sid,))
    bid = con.execute("SELECT id FROM bundle").fetchone()[0]
    gates.approve(con, "concat", bundle_id=bid)
    assert gates.concat_allowed(con, bid)[0] is False        # 'planned' blocks
    con.execute("UPDATE bundle SET teaser_state='approved' WHERE id=?", (bid,)); con.commit()
    assert gates.concat_allowed(con, bid)[0] is True
    con.execute("UPDATE bundle SET teaser_state='declined' WHERE id=?", (bid,)); con.commit()
    assert gates.concat_allowed(con, bid)[0] is True
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** In `gates.py` add `teaser_allowed(con, bundle_id)` (mirror `concat_allowed`, checks `_has_approval(con,"teaser",bundle_id=...)`). Extend `concat_allowed` — **must be None-safe** (the existing `test_concat_gate` calls it with no bundle row):
```python
def concat_allowed(con, bundle_id):
    row = con.execute("SELECT teaser_state FROM bundle WHERE id=?", (bundle_id,)).fetchone()
    if row and row[0] == "planned":
        return False, "teaser planned but not reviewed"
    if not _has_approval(con, "concat", bundle_id=bundle_id):
        return False, "needs concat approval"
    return True, ""
```
Note the `if row and ...` guard — `fetchone()` is `None` when no bundle row exists, and `row[0]` would crash the existing gate test otherwise (regression guard).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

### Task 10: `plan_teaser` lane + `_h_teaser` handler + HANDLERS

**Files:** `studio/dashboard/jobs.py`, `studio/worker.py`, `tests/test_teaser_worker.py` (Create)

- [ ] **Step 1: Failing test** (monkeypatch the tool subprocess layer so no real render runs):
```python
def test_h_teaser_plans_and_sets_state(tmp_path, monkeypatch):
    # build a bundle of 2 chapters with understood/cast/scenes manifests on disk
    ...  # connect(con), repo.upsert_series/upsert_chapter/set_chapter_status(ep_dir=...)
    import studio.worker as w
    import studio.pipeline as pl
    calls = []
    # the handler spans BOTH layers: render_prep/remotion go through worker._stream,
    # and script_expander/local_tts/timeline_planner go through pipeline._run_tool.
    # Monkeypatch BOTH or the first three run as real subprocesses and hang.
    monkeypatch.setattr(w, "_stream", lambda argv, log, **k: calls.append(argv) or 0)
    monkeypatch.setattr(pl, "_run_tool", lambda script, args, **k: calls.append([script, *args]) or None)
    # make the planning subprocess "succeed" + write a teaser.mp4 (touch the expected outputs in the stub)
    ...
    w._h_teaser(con, {"bundle_id": bid, "payload": {}}, io.StringIO())
    assert con.execute("SELECT teaser_state FROM bundle WHERE id=?", (bid,)).fetchone()[0] == "planned"
```
(Use the `tests/test_pipeline.py` `_run_tool`/`_stream` monkeypatch + `connect`+`repo` row idioms. The stub must create the files the next step expects: `out_dir/manifest.teaser.json` after planning, and `out_dir/render/segment_none.mp4` after the render mocks, so the copy-to-`teaser.mp4` step succeeds.)

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
  - `jobs.py` `LANES`: add `"plan_teaser"` on a CPU/API lane (it does model + render → put on the render-like lane, e.g. `"cpu"`, matching `concat`/`render_segment`). **Must** be listed or the job queues forever.
  - `worker.py` `_h_teaser(con, job, log)`:
    1. `bid = job["bundle_id"]`; resolve `chapter_dirs = [Path(_chapter(con,cid)["ep_dir"]) for cid in bundles.bundle_chapters(con, bid)]`.
    2. `out_dir = REPO/"dist"/f"bundle_{bid}"/"teaser"`.
    3. Run the planner. Build argv config-synced to the backend (mirror `_stage_beated`): always `--bundle-id`, `--chapter-dirs`, `--out-dir`, `--backend cfg.beats_backend`, `--max-scan-chapters cfg.teaser_max_hook_scan_chapters`, `--shortlist-n cfg.teaser_shortlist_n`, etc.; then for ollama add `--ollama-model cfg.beats_model`, for vertex add `--model cfg.teaser_model --project … --location …`:
```python
argv = [PY, str(REPO/"tools"/"teaser_planner.py"), "--bundle-id", str(bid),
        "--chapter-dirs", *map(str, chapter_dirs), "--out-dir", str(out_dir),
        "--backend", cfg.beats_backend,
        "--max-scan-chapters", str(cfg.teaser_max_hook_scan_chapters),
        "--shortlist-n", str(cfg.teaser_shortlist_n)]
if cfg.beats_backend == "ollama":
    argv += ["--ollama-model", cfg.beats_model]
else:
    argv += ["--model", cfg.teaser_model, "--project", project, "--location", location]
if _stream(argv, log) != 0:
    raise RuntimeError("teaser_planner failed")
```
If the planner wrote no `manifest.teaser.json` (no-teaser case), leave `teaser_state='none'` and return (concat stays unblocked) — do NOT set `declined`. 
    4. If a teaser was planned, run the render TOOLS on `out_dir` (mirror `_h_render_segment` body + the scripted/voiced/planned argv from the spec): script_expander → local_tts_from_manifest (`python_exe=cfg.tts_python`) → timeline_planner → render_prep → remotion. Reuse `studio.pipeline._run_tool` for the first three (it sets PYTHONPATH) and `_stream` for render_prep (cwd=`REPO`) + remotion (cwd=`REPO/"remotion"`). When `cfg.beats_backend=="ollama"`, pass the model via `--ollama-model cfg.beats_model` (the script's ollama path ignores `--model`). **Cut-judge cache (optional, fail-soft):** render_prep reads/writes `out_dir/scenes_clean/.cut_judge_cache.json` keyed by scene **basename** (render_prep.py:1370/2059). Because the teaser keeps identical scene basenames, seed it by copying each source chapter's `scenes_clean/.cut_judge_cache.json` entries (by basename) into `out_dir/scenes_clean/.cut_judge_cache.json` before render_prep — a plain basename-keyed merge (no symlink path-remap needed). Skipping this only re-pays Gemma judging; no correctness impact.
    5. Copy `out_dir/render/segment_none.mp4` → `REPO/"dist"/f"bundle_{bid}"/"teaser.mp4"`.
    6. `con.execute("UPDATE bundle SET teaser_state='planned' WHERE id=?", (bid,)); con.commit()`.
    Wrap in `record_stage(con, chapter_id=None, stage="plan_teaser")`.
  - `HANDLERS`: add `"plan_teaser": _h_teaser`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

### Task 11: `_h_concat` prepends approved teaser

**Files:** `studio/worker.py`, `tests/test_teaser_worker.py`

- [ ] **Step 1: Failing test**
```python
def test_h_concat_prepends_teaser_when_approved(tmp_path, monkeypatch):
    # setup: bundle with teaser_state='approved'; gates.approve(con,"concat",bundle_id=bid)
    # (else the gate raises before the prepend); each chapter needs a rendered
    # ep_dir/render/segment_*.mp4 on disk (else _h_concat raises "no rendered segment");
    # and dist/bundle_<id>/teaser.mp4 must exist.
    ...
    captured = {}
    monkeypatch.setattr(w, "_stream", lambda argv, log, **k: captured.setdefault("argv", argv) or 0)
    monkeypatch.setattr(w.bundles, "concat_cmd", lambda segs, out: (captured.update(segs=segs) or (["ffmpeg","LISTFILE"], "")))
    w._h_concat(con, {"bundle_id": bid}, io.StringIO())
    assert captured["segs"][0].endswith("teaser.mp4")        # prepended first
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** In `_h_concat`, immediately **before** the `if bdir is not None:` branding block (worker.py:798, after `segs` is fully built from the chapter glob and before `wrap_with_branding`), insert the prepend so a branding-less bundle still gets the teaser first:
```python
    teaser_mp4 = REPO / "dist" / f"bundle_{bid}" / "teaser.mp4"
    row = con.execute("SELECT teaser_state FROM bundle WHERE id=?", (bid,)).fetchone()
    if row and row[0] == "approved" and teaser_mp4.exists():
        segs = [str(teaser_mp4)] + segs
```
(`wrap_with_branding` only appends the outro — bundles.py:89-101 — so prepend-then-wrap yields `[teaser, ch1…chN, outro]`.)
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

## Chunk 5: dashboard — plan button, review card, approve/decline

### Task 12: `/videos` teaser context + routes

**Files:** `studio/dashboard/app.py`, `tests/dashboard/test_videos_teaser.py` (Create)

- [ ] **Step 1: Failing test** (FastAPI TestClient, mirror existing dashboard tests):
```python
def test_plan_teaser_enqueues_job(...):
    client = TestClient(app)
    r = client.post("/bundles/%d/teaser/plan" % bid, follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser' AND bundle_id=?", (bid,)).fetchone()[0] == 1

def test_decline_sets_state(...):
    client.post("/bundles/%d/teaser/decline" % bid, follow_redirects=False)
    assert con.execute("SELECT teaser_state FROM bundle WHERE id=?", (bid,)).fetchone()[0] == "declined"
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** in `app.py`:
  - In `videos_page`, add to each bundle dict `b`: `teaser_state` (from the row) and `teaser_card = _teaser_card(out_dir)` where `_teaser_card` loads `dist/bundle_<id>/teaser/manifest.teaser.json` (reason/rewind_line/spoiler_boundary/panel_narration) + thumbnails from `teaser/scenes/` for the review card (mirror `_gallery` shape).
  - New routes (mirror `post_approve`/`post_bundle`):
    - `@app.post("/bundles/{bid}/teaser/plan")` → `jobs.enqueue(c,"plan_teaser",bundle_id=bid)` → 303 `/videos`.
    - `@app.post("/bundles/{bid}/teaser/approve")` → `gates.approve(c,"teaser",bundle_id=bid)`; `UPDATE bundle SET teaser_state='approved'`; 303.
    - `@app.post("/bundles/{bid}/teaser/decline")` → `UPDATE bundle SET teaser_state='declined'`; 303.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

### Task 13: `videos.html` button + review card + badge

**Files:** `studio/dashboard/templates/videos.html` (no new test — covered by Task 12 routes + a render smoke check)

- [ ] **Step 1:** Add a "Plan teaser" button (form POST `/bundles/{{b.id}}/teaser/plan`) in the action `<td>`, mirroring the existing concat/publish_meta forms. Add a teaser **state badge** column (none/planned/approved/declined) — bump the `<th>` count and the empty-state `colspan`.
- [ ] **Step 2:** When `b.teaser_state=='planned'` and `b.teaser_card` exists, render a review card (clone the `chapter.html` Groups gallery fragment: thumbnails from `/media/...teaser/scenes/...` + the per-panel narration + `reason`/`spoiler_boundary`) with **Approve** / **Decline** / **Re-plan** forms.
- [ ] **Step 3:** Smoke: `$V -m pytest tests/dashboard -q` stays green; manually verify the page renders (or a TestClient GET `/videos` returns 200 with the button text).
- [ ] **Step 4: Commit.**

### Task 14: Full-suite gate + docs

- [ ] **Step 1:** `$V -m pytest -q` — entire suite green.
- [ ] **Step 2:** Update `CLAUDE.md` "Current state / next work" with the teaser stage + the daemon-restart reminder (worker.py + dashboard changed → `launchctl kickstart -k` on deploy).
- [ ] **Step 3: Commit.**

---

## Deferred (not in this plan — named so scope is explicit)
- Flashback / story-memory inserts.
- Dashboard auto-batcher (suggest next range) + tail-handling policy.
- Future-teasing (windows outside the selected bundle).

## Open implementation notes (resolve during the task, don't guess)
- **manifest.scenes.json copy:** confirm each source chapter's `manifest.scenes.json` has an entry whose `out_file == <basename>` for every teaser scene; if a panel's scene isn't in its chapter's scenes manifest, drop it from the teaser window (log it) rather than emit a broken scenes manifest.
- **render_prep `--branding`:** use `none` for the teaser (it is the opening; no intro/outro wrap on the teaser segment itself).
- **timeline_planner `--groups` required:** the synthetic `manifest.groups.json` (Task 7) satisfies it; `--vision` is optional and omitted.
