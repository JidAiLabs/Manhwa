# Per-Panel Rolling Narration Implementation Plan

> **For agentic workers:** REQUIRED: Use subagent-driven-development (if subagents available) or executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-group narration + word-count "microbeat" splitting with one purpose-written narration line per surviving story panel, generated one-call-per-group with rolling continuity, and recognize in-world system/UI cards as a first-class `panel_kind`.

**Architecture:** The narration *unit* moves from the group to the panel; grouping stays as the continuity unit. `panel_understand.py` gains a `system` `panel_kind`. `gemini_narrative_pass.py` emits a `panel_narration[]` array per beat (one `{scene_file, line}` per panel) plus a joined `narration` string for back-compat. `script_expander.py` materializes one segment per panel (no microbeat split). `segment_id` (`g####_p##`) stays byte-identical. A deterministic repair-fill guarantees coverage so a panel can never be silently dropped.

**Tech Stack:** Python 3.12, pytest, the `.eval_venv` venv, local Ollama (Gemma) / Vertex Gemini backends. Spec: `docs/plans/specs/2026-06-19-per-panel-rolling-narration-design.md`.

---

## Conventions (read once)

- **Venv / test runner:** `V=.eval_venv/bin/python`. Full suite: `$V -m pytest -q`. One test: `$V -m pytest tests/test_x.py::test_y -v`.
- **Test imports:** tools are loaded via a **module-level singleton** (no `_load_*()` helpers exist — don't invent them). Follow `tests/test_story_group.py:9-14` exactly. For a new test file, put this at the top and then call `<obj>.func(...)`:
  ```python
  import importlib.util
  from pathlib import Path
  _SPEC = importlib.util.spec_from_file_location(
      "gemini_narrative_pass",
      Path(__file__).resolve().parent.parent / "tools" / "gemini_narrative_pass.py")
  gnp = importlib.util.module_from_spec(_SPEC)
  _SPEC.loader.exec_module(_SPEC and gnp)  # type: ignore[union-attr]
  ```
  The module objects used below are: `pu` (panel_understand), `sg` (story_group), `gnp` (gemini_narrative_pass), `npu` (narration_punchup), `se` (script_expander). Some target files have heavy imports (torch/cv2) — if `exec_module` is slow or import-heavy (e.g. `prep_qa`), prefer extending that tool's **existing** test file, which already loads it.
- **Subprocess tests:** `tests/test_b2_segment_id.py` runs `script_expander.py` as a **process** (builds input manifests on disk, runs the tool, reads the output) — it does NOT import the module. Any segment_id assertion there must follow that build-input→run→read-output pattern, not pass in-memory objects.
- **Branch:** work directly on `main`. This project deploys by `git push` → on the Mini `git fetch && git reset --hard origin/main`; the worker is currently **paused with Ch1 isolated**, so `main` is safe to iterate. Do **not** create a worktree (it can't reach the Mini's Ch1 validation loop). Commit after every passing step.
- **Manifest = API.** `segment_id` must stay byte-identical across `script_expander → tts → timeline_planner`. The joined `narration` string is retained so legacy readers don't break.
- **Order matters.** Implement chunks in order: a `system` panel must be classifiable (Ch1) and kept (Ch2) before per-panel narration (Ch3) can cover it.

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `tools/panel_understand.py` | Multimodal per-panel understanding | Add `system` to `panel_kind` enum + norm + prompt (§Ch1) |
| `tools/story_group.py` | Group panels; drop non-story | Keep `system` panels (one-line edit) (§Ch2) |
| `tools/gemini_narrative_pass.py` | Per-beat narration writer | Emit `panel_narration[]` + repair-fill (§Ch3) |
| `tools/narration_punchup.py` | Persona pass over narration | Operate per panel-line (§Ch4) |
| `tools/script_expander.py` | Materialize script segments | One segment per panel; retire live microbeat split (§Ch5) |
| `tools/timeline_planner.py` | Plan cuts from segments | Verify per-segment pick = identity (§Ch6) |
| `tools/prep_qa.py` | Pre-render QA | Per-panel coverage + system-shown assertion (§Ch6) |
| `tools/narration_heal.py` | Re-narrate flagged groups | Re-narrate group's `panel_narration` (§Ch6) |
| `studio/config.py`, `studio/pipeline.py` | Stage wiring | `narration_microbeats` switchable; per-panel default (§Ch6) |

---

## Chunk 1: Understanding — first-class `system` panel_kind

**Files:**
- Modify: `tools/panel_understand.py:43-44` (enum), `:87-89` (`_norm_panel_kind`), `:77-81` (prompt `story`/`system` boundary)
- Test: `tests/test_panel_understand.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_panel_understand.py` (use the file's existing module loader):

```python
def test_norm_panel_kind_accepts_system():
    assert pu._norm_panel_kind("system") == "system"
    assert pu._norm_panel_kind("SYSTEM") == "system"
    assert pu._norm_panel_kind("garbage") == "story"   # unknown -> never-drop side

def test_panel_schema_enumerates_system():
    enum = pu.PANEL_SCHEMA["properties"]["panel_kind"]["enum"]
    assert "system" in enum
    assert set(enum) == {"story", "chrome", "empty", "caption", "system"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.eval_venv/bin/python -m pytest tests/test_panel_understand.py::test_norm_panel_kind_accepts_system tests/test_panel_understand.py::test_panel_schema_enumerates_system -v`
Expected: FAIL (`system` not in enum; norm returns `story`).

- [ ] **Step 3: Implement**

`tools/panel_understand.py:43-44` — add `system` to the enum:
```python
        "panel_kind": {"type": "STRING",
                       "enum": ["story", "chrome", "empty", "caption", "system"]},
```
`tools/panel_understand.py:87-89` — accept it:
```python
def _norm_panel_kind(v: Any) -> str:
    v = str(v or "").strip().lower()
    return v if v in ("story", "chrome", "empty", "caption", "system") else "story"
```
`tools/panel_understand.py` prompt — replace the `'story'` definition block (the lines beginning `"    'story' = the STORY WORLD …"` through `"… When unsure, 'story'.\n"`) with a `system` bucket + a tightened `story`:
```python
    "    'system' = an IN-WORLD GAME / SYSTEM INTERFACE the CHARACTER perceives — "
    "a QUEST window, a STATUS / STAT / SKILL screen, a NOTIFICATION / ALARM / level-up "
    "toast, or a SYSTEM MESSAGE (e.g. 'QUEST DIRECTIONS', 'STATUS', 'NOTIFICATION — You "
    "have defeated a [Steel-Fanged Lycan]', '7TH GENERATION NANO MACHINE, STARTING "
    "ACTIVATION'). It can be ANY length, ANY case, ANY color/art style, and may be drawn "
    "OVER character art. These are PLOT and MUST be kept and shown.\n"
    "    'story' = the STORY WORLD — real scene art AND in-world device screens a "
    "character uses in-story (a reader app, chat, feed), a place/organization name card. "
    "A panel with real character art is 'story' even if a system window is drawn over it. "
    "When unsure between system/story (both are always kept), pick either; only an AUTHOR "
    "narrative caption is 'caption' and only platform furniture is 'chrome'.\n"
```
(Leave the `caption` and `chrome` definitions unchanged — the boundary is now explicit.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.eval_venv/bin/python -m pytest tests/test_panel_understand.py -v`
Expected: PASS (all, including pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add tools/panel_understand.py tests/test_panel_understand.py
git commit -m "feat(understand): first-class 'system' panel_kind for in-world UI cards"
```

---

## Chunk 2: `system` never dropped — across ALL panel_kind consumers

> **Scope expanded from the Chunk 1 code-quality review.** The drop decision isn't only in `story_group`; several modules branch on `panel_kind` and hardcode `("story","caption")` as the never-drop set. A `system` panel must be kept by every one of them. The highest-leverage fix is the **single chrome chokepoint** `scene_chrome.is_chrome_scene`, which `story_group`/`render_prep`/`prep_qa` all route through (per its own docstring at `tools/scene_chrome.py:90-95`).

**Files:**
- Modify: `tools/scene_chrome.py:98` (the chokepoint), `tools/story_group.py:393-395` (defense-in-depth), `tools/render_prep.py:1894` (bubble-drop exemption)
- Test: `tests/test_story_group.py`, `tests/test_scene_chrome.py` (or the file that tests `is_chrome_scene`), `tests/test_prep_qa.py`/render_prep tests
- [ ] **Step 0: Sweep first.** `grep -rn '("story", *"caption")\|panel_kind' tools/ | grep -i 'story.*caption'` and read every hit. Any site whose intent is "these kinds are kept / not chrome / exempt from drop" must include `"system"`. Confirmed sites below; add `system` to any others the sweep finds, with a one-line note in the commit body.

### Task 2a: scene_chrome chokepoint (primary)

- [ ] **Step 1 (test):** in the test file that loads `scene_chrome` (module obj per that file; create `tests/test_scene_chrome.py` with a spec loader if none exists), assert a `system` item is never chrome regardless of OCR/midtones:
```python
def test_system_panel_is_never_chrome():
    # sparse OCR + low midtone would trip the binary-card heuristic for a non-system kind
    assert sc.is_chrome_scene({"panel_kind": "system", "ocr_clean": ""}, midtone_frac=0.02) is False
    assert sc.is_chrome_scene({"panel_kind": "system", "ocr_clean": "QUEST DIRECTIONS"}, midtone_frac=0.5) is False
```
  (Match `is_chrome_scene`'s real signature — confirm whether `midtone_frac` is a kwarg/positional by reading `tools/scene_chrome.py`.)
- [ ] **Step 2:** run → FAIL (system falls through to OCR/midtone heuristic and can return True).
- [ ] **Step 3:** `tools/scene_chrome.py:98` — add `system` to the not-chrome fast-exit:
```python
    if kind in ("story", "caption", "system"):
        return False
```
  Update the docstring (`:90-95`) to say a `story`/`caption`/`system` verdict is never chrome.
- [ ] **Step 4:** run → PASS.

### Task 2b: story_group keep_by_understanding (defense-in-depth)

- [ ] **Step 1 (test):** add to `tests/test_story_group.py` (module obj `sg`):
```python
def test_system_panel_is_never_excluded():
    panels = [
        {"scene_file": "p01.jpg", "panel_kind": "story",  "description": "a man stands", "subjects": ["man"]},
        {"scene_file": "p02.jpg", "panel_kind": "system", "description": "QUEST DIRECTIONS window",
         "dialogue": "QUEST DIRECTIONS. NUMBER OF PLAYERS TO KILL: 1.", "subjects": []},
    ]
    assert "p02.jpg" not in sg.nonstory_files(panels)
    assert "p02.jpg" not in sg.effect_only_files(panels)
    assert "p02.jpg" not in sg.caption_files(panels)
```
- [ ] **Step 2:** run → these PASS already (the helpers don't catch `system`); the real gap is `keep_by_understanding` shielding it from `ocr_chrome`. Add the fix anyway for explicitness.
- [ ] **Step 3:** `tools/story_group.py:393-395`:
```python
        keep_by_understanding = {p.get("scene_file") for p in panels
                                 if str(p.get("panel_kind") or "").lower()
                                 in ("story", "caption", "system") and not p.get("error")}
```

### Task 2c: render_prep bubble-drop exemption

- [ ] **Step 1 (test):** assert a text-bearing `system` panel is exempt from `drop_bubble_dominated_cuts` (mirror the existing story/caption exemption test in the render_prep tests; if none, add a focused one).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** `tools/render_prep.py:1894`:
```python
                    if recoverable and (vit.get("panel_kind") in ("story", "caption", "system")
                            and str(vit.get("ocr_clean") or "").strip()):
                        exempt.add(f)
```
- [ ] **Step 4:** run → PASS.

- [ ] **Final: Run** `.eval_venv/bin/python -m pytest tests/test_story_group.py tests/test_scene_chrome.py tests/test_prep_qa.py -v` → PASS (all). Then **Commit**:
```bash
git add tools/scene_chrome.py tools/story_group.py tools/render_prep.py tests/
git commit -m "feat: never drop in-world 'system' panels across chrome/story_group/render_prep"
```

---

## Chunk 3: beated — per-panel narration (`panel_narration[]`)

This is the core. `gemini_narrative_pass.py` currently emits one `narration` string per beat (`beat_schema:1017-1068`, prompt at `:891`, prev threading at `:1132-1133`, fallback at `:1155-1172`). Change it to return one line per panel, aligned to `scene_files`, with a deterministic repair-fill, and keep `narration` as the space-joined string.

All new `gnp` unit tests go in a **new** `tests/test_panel_narration.py` with its own module-level spec loader (per Conventions) — NOT in `test_narrative_quality.py` (that file uses a `sys.path` style and targets `script_expander`).

### Task 3-pre: load the understanding manifest in the beated stage (prerequisite for the grounded pad)

The repair-fill pad needs each panel's understanding (`description`/`action`/`subjects`). `gemini_narrative_pass.main()` loads only `groups_m` and `vision_m` (`:865-866`); there is **no understanding in scope** and `source_understood` in the groups manifest is unreliable (older manifests write `None`). Add an explicit `--understood` arg + load + pipeline wiring.

**Files:** Modify `tools/gemini_narrative_pass.py` (arg parser near `:830-856`, `main()` near `:865`), `studio/pipeline.py` (`_stage_beated`, ~`:253-261`)
- [ ] **Step 1 (test):** in `tests/test_panel_narration.py`, assert the CLI parser accepts `--understood` and `main` tolerates its absence. Minimal: `assert "--understood" in open(gnp.__file__).read()` is too weak — instead test the parser: build the `argparse` via a thin `gnp.build_arg_parser()` (refactor the inline parser into a function) and `assert parser.parse_args([...,"--understood","x.json"]).understood == "x.json"`.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** refactor the inline `argparse` into `build_arg_parser()`, add `ap.add_argument("--understood", default="", help="manifest.panels.understood.json for per-panel pad grounding")`. In `main()` after `vision_m` load:
```python
    understood_m = load_json(args.understood) if args.understood and os.path.exists(args.understood) else {}
    u_by_file = {p.get("scene_file"): p for p in (understood_m.get("panels") or []) if p.get("scene_file")}
```
  In `studio/pipeline.py`, `_ep_paths` (`:105-122`) has **no `"understood"` key** — the path is a local literal in `_stage_grouped` (`:182`). Add the key once:
```python
        "understood": ep_dir / "manifest.panels.understood.json",
```
  then in `_stage_beated` add `"--understood", str(p["understood"])` to the `gemini_narrative_pass.py` arg list (and optionally replace the `_stage_grouped` local literal with `p["understood"]` to de-duplicate). Do NOT use `p["understood"]` before adding the key — it will `KeyError`.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(beats): pass --understood to beated stage for per-panel grounding`.

### Task 3a: the repair-fill helper (pure, fully unit-tested)

- [ ] **Step 1: Write the failing test** (new `tests/test_panel_narration.py`, loader pattern from `tests/test_story_group.py`):

```python
def test_align_pads_missing_panels_from_understanding():
    files = ["a.jpg", "b.jpg", "c.jpg"]
    model = [{"scene_file": "a.jpg", "line": "He draws the blade."},
             {"scene_file": "c.jpg", "line": "Silence falls."}]   # b missing
    u = {"b.jpg": {"description": "the beast lunges"}}            # understood schema: description/action/subjects
    out = gnp.align_panel_narration(files, model, u)
    assert [p["scene_file"] for p in out] == files               # order + coverage
    assert out[1]["line"] == "the beast lunges"                  # padded from understanding, never empty

def test_align_is_positional_when_model_omits_scene_file():
    files = ["a.jpg", "b.jpg"]
    model = [{"line": "First."}, {"line": "Second."}]            # no scene_file keys
    out = gnp.align_panel_narration(files, model, {})
    assert [p["line"] for p in out] == ["First.", "Second."]

def test_align_folds_overflow_into_last_panel_no_phantoms():
    files = ["a.jpg"]
    model = [{"scene_file": "a.jpg", "line": "One."}, {"scene_file": "zzz.jpg", "line": "Two."}]
    out = gnp.align_panel_narration(files, model, {})
    assert len(out) == 1 and out[0]["scene_file"] == "a.jpg"     # no phantom panel
    assert out[0]["line"] == "One. Two."                         # overflow folded, nothing lost

def test_align_invariant_length_matches_scene_files():
    files = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    out = gnp.align_panel_narration(files, [], {})               # model returned nothing
    assert len(out) == len(files)
    assert all(p["line"] for p in out)                           # every panel has a (fallback) line
```

- [ ] **Step 2: Run** the four tests → FAIL (`align_panel_narration` undefined).

- [ ] **Step 3: Implement** — add near the top of `tools/gemini_narrative_pass.py`:

```python
def align_panel_narration(scene_files, model_panels, understand_by_file=None):
    """Return exactly one {scene_file, line} per surviving scene_file, in order.

    Match the model's returned lines to panels by scene_file; fall back to
    positional fill for any panel the model didn't key; pad any still-missing
    panel with a grounded line from the understanding (description/action/
    subjects); fold overflow lines into the LAST panel so nothing is lost. Never
    invents a panel absent from scene_files. Guarantees len(out)==len(scene_files).
    """
    understand_by_file = understand_by_file or {}
    files = [f for f in (scene_files or []) if f]
    file_set = set(files)
    keyed: Dict[str, str] = {}
    leftover: List[str] = []
    for item in (model_panels or []):
        if not isinstance(item, dict):
            continue
        line = str(item.get("line") or item.get("narration") or "").strip()
        if not line:
            continue
        sf = str(item.get("scene_file") or "").strip()
        if sf in file_set and sf not in keyed:
            keyed[sf] = line
        else:
            leftover.append(line)
    for f in files:                       # positional fill for unkeyed panels
        if f not in keyed and leftover:
            keyed[f] = leftover.pop(0)
    for f in files:                       # grounded pad — never empty
        if f not in keyed:
            u = understand_by_file.get(f) or {}
            subj = ", ".join(str(s) for s in (u.get("subjects") or []) if s)
            keyed[f] = (str(u.get("description") or u.get("action") or "").strip()
                        or subj.strip() or "The moment holds.")
    out = [{"scene_file": f, "line": keyed[f]} for f in files]
    if leftover and out:                  # fold any remaining overflow into the last panel
        out[-1]["line"] = (out[-1]["line"] + " " + " ".join(leftover)).strip()
    return out
```

- [ ] **Step 4: Run** the four tests → PASS.

- [ ] **Step 5: Commit**
```bash
git add tools/gemini_narrative_pass.py tests/test_panel_narration.py
git commit -m "feat(beats): align_panel_narration repair-fill (coverage guaranteed)"
```

### Task 3b: schema + prompt + wiring

- [ ] **Step 1: Write the failing test** (append to `tests/test_panel_narration.py`):

```python
def test_beat_schema_requires_panel_narration():
    schema = gnp.build_beat_schema()          # extract the inline beat_schema into a builder
    props = schema["properties"]
    assert "panel_narration" in props
    assert props["panel_narration"]["type"] == "ARRAY"
    item = props["panel_narration"]["items"]["properties"]
    assert set(item) >= {"scene_file", "line"}
    assert "panel_narration" in schema["required"]
    assert "narration" in props          # joined string kept for back-compat
```

- [ ] **Step 2: Run** → FAIL (`build_beat_schema` undefined / no `panel_narration`).

- [ ] **Step 3: Implement**
  1. Refactor the inline `beat_schema` dict (`:1017-1068`) into a top-level `def build_beat_schema() -> dict:` returning the same dict, then call it where the inline literal was. Add the new property + requirement:
```python
        "panel_narration": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "scene_file": {"type": "STRING"},
                    "line": {"type": "STRING"},
                },
                "required": ["scene_file", "line"],
            },
        },
```
     Add `"panel_narration"` to the `required` list. Keep `"narration"` in properties and required (the model still returns a joined line; we overwrite it from the panels below to guarantee consistency).
  2. Narration prompt (`:891`, the `"WRITE ONE 'narration' line …"` block): replace with a per-panel instruction:
```python
        "For EACH file in scene_files, in order, WRITE ONE narration line in "
        "'panel_narration' as {scene_file, line}. Give EVERY panel its own line — "
        "a quick action panel gets a punchy phrase, a pivotal/quiet panel gets a "
        "fuller cinematic sentence; match length to what the panel shows. The lines "
        "must FLOW as one continuous story (continue from previous_narration), not "
        "isolated captions. Then set 'narration' to all the lines joined with a space.\n"
```
  3. In the per-group loop, after `_generate_beat_for_group` returns `beat`, normalize panels (handles both success and the parse-failure fallback). `u_by_file` was built in `main()` by **Task 3-pre** and is in scope:
```python
        surviving = [f for f in (beat.get("scene_files") or payload["scene_files"]) if f]
        beat["panel_narration"] = align_panel_narration(
            surviving, beat.get("panel_narration"), u_by_file)
        # spec §8 loud invariant: the repair-fill guarantees this; assert so a
        # future regression in align_panel_narration fails the beated stage
        # rather than emitting a misaligned manifest.
        assert len(beat["panel_narration"]) == len(surviving), (
            f"panel_narration/scene_files mismatch in group {gid}")
        beat["narration"] = " ".join(p["line"] for p in beat["panel_narration"]).strip()
```
     Do **not** set `narration_plain` here — the punchup stage owns that key (it stashes the grounded line into `narration_plain` in `merge`). The beated joined `narration` becomes punchup's plain base, which is correct.
  4. Fallback beat (`:1155-1172`): leave as-is — the normalization in (3) runs after it and fills `panel_narration` from `scene_files` + understanding, so a parse failure still yields one grounded line per panel.
  5. prev threading (`:1132-1133`): **unchanged** — it reads `b.get("narration")`, which we keep as the joined spine tail.

- [ ] **Step 4: Run** `.eval_venv/bin/python -m pytest tests/test_panel_narration.py tests/test_narrative_quality.py tests/test_register_narration.py tests/test_meta_garbage_narration.py -v` → PASS. (Register/meta-garbage tests guard that the joined `narration` still behaves.)

- [ ] **Step 5: Commit**
```bash
git add tools/gemini_narrative_pass.py tests/test_panel_narration.py
git commit -m "feat(beats): emit panel_narration[] (one line per panel) + joined narration"
```

---

## Chunk 4: punchup per panel-line

`narration_punchup.py` builds a payload `[{group_id, narration}]` per beat (`:200-217`) and writes the persona line back per beat. Move it to per-panel: persona each `panel_narration[i].line`, preserve a per-panel `line_plain`, and rejoin `narration`.

**Files:**
- Modify: `tools/narration_punchup.py` (`build_prompt:200-217`, the apply/merge step, the per-beat write)
- Test: `tests/test_narration_punchup.py` (extend)

- [ ] **Step 1: Write the failing test:**
```python
def test_punchup_payload_is_per_panel_line():
    beats = {"beats": [{"group_id": 1, "panel_narration": [
        {"scene_file": "a.jpg", "line": "He stands."},
        {"scene_file": "b.jpg", "line": "The system speaks."}]}]}
    payload = npu.build_panel_payload(beats)        # new per-panel payload builder
    assert payload == [
        {"group_id": 1, "panel_index": 0, "narration": "He stands."},
        {"group_id": 1, "panel_index": 1, "narration": "The system speaks."}]

def test_apply_punchup_preserves_alignment_and_plain():
    beat = {"group_id": 1, "panel_narration": [
        {"scene_file": "a.jpg", "line": "He stands."},
        {"scene_file": "b.jpg", "line": "The system speaks."}]}
    rewrites = {(1, 0): "He rises, blade ready.", (1, 1): "The System hums to life."}
    npu.apply_panel_punchup(beat, rewrites)
    pn = beat["panel_narration"]
    assert [p["line"] for p in pn] == ["He rises, blade ready.", "The System hums to life."]
    assert pn[0]["line_plain"] == "He stands."          # grounded line preserved
    assert beat["narration"] == "He rises, blade ready. The System hums to life."
```

- [ ] **Step 2: Run** → FAIL (`build_panel_payload`/`apply_panel_punchup` undefined).

- [ ] **Step 3: Implement** (model the per-panel apply on the existing `merge`, `tools/narration_punchup.py:302`)
  - `build_panel_payload(beats_obj)` — flatten beats→panel lines, carry `(group_id, panel_index)`. Send to the existing persona LLM call (same prompt/voice); update `build_prompt` to say "array of `{group_id, panel_index, narration}`, same length, rewrite each line in the persona, keep length proportional, never merge or drop lines."
  - **Preserve the grounding safety gate — it is `validate_line` (`:269`), not `validate_rewrites` (that does not exist).** `merge` (`:302`) today validates each beat's candidate with `validate_line(original, cand, cast_names, required=caption_words.get(gid), max_ratio=<by class>)` and **restores the plain line on failure** (the strictly-better guard — memory `closed-loop-regen-needs-safety-guard`). Per-class `max_ratio`: DRAMATIC 3.0 / COMIC 2.2 / else 1.5 (`:322-328`). `apply_panel_punchup` MUST call `validate_line` **per panel line** with the same arguments and the same per-class `max_ratio`, accept only if it passes, else keep that panel's plain line.
  - Mirror `merge`'s text hygiene per line: `cand.replace("*", "")` (md→TTS-safe, `:321`) and `strip_chrome_opener(...)` on both the accepted line and its plain (`:341-342`).
  - The per-panel plain key is **`line_plain`** (distinct from the beat-level `narration_plain`, which `merge` writes at `:320`/`:342` — do not conflate). `apply_panel_punchup` writes each accepted `line`, stashes the original under `line_plain`, rejoins `beat["narration"]`, and still sets the beat-level `narration_plain` (joined plain lines) so beat readers are unaffected.
  - Keep the COMIC/CINEMATIC `classify_beats` gate per beat (a beat's class supplies the `max_ratio` and still gates whether its panel lines may take persona).

- [ ] **Step 4: Run** `.eval_venv/bin/python -m pytest tests/test_narration_punchup.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add tools/narration_punchup.py tests/test_narration_punchup.py
git commit -m "feat(punchup): apply persona per panel-line, preserve alignment + plain"
```

---

## Chunk 5: scripted — one segment per panel

`_build_verbatim_section` (`tools/script_expander.py:790`; the real function name — there is no `_materialize_section_from_beats`) currently splits each beat's narration into microbeat parts and (with `--microbeats`) builds one shot per part via `_build_microbeat_shot`, assigning panels positionally with `_scene_for_microbeat`. Replace the live path: iterate `beat.panel_narration` → one paragraph + one shot per panel (`scene_files=[that panel]`). Keep the microbeat functions behind `--microbeats` for the A/B fallback.

**Contract guard:** the `segment_id` stamp (`:2271-2275`) requires `len(shots) == len(paras) == len(tts_v3)`. The per-panel path appends exactly one `paras` entry and one `shots` entry per panel in the same loop iteration, and `tts` is derived from `para_beats` — so all three stay equal. Keep it that way.

**Files:**
- Modify: `tools/script_expander.py` — the per-beat loop in `_build_verbatim_section` (`:820-890`); segment_id stamp untouched (`:2271-2275`)
- Test: `tests/test_verbatim_script.py` — both the module-import tests (`_build_verbatim_section`) AND the `script_expander.py` CLI subprocess test (for the segment_id contract, Step 1b); update existing `test_verbatim_section_is_valid_with_one_shot_per_beat` (`:206`). (`tests/test_b2_segment_id.py` is a timeline_planner test — run it as regression, but it is not where the expander segment_id assertion goes.)
- ⚠️ **Existing test will break:** `test_verbatim_section_is_valid_with_one_shot_per_beat` (`tests/test_verbatim_script.py:206`) asserts one shot per *beat*. Under per-panel it becomes one shot per *panel* — update that test (rename to `…one_shot_per_panel`) as part of this task.

- [ ] **Step 1: Write the failing test** (add to `tests/test_verbatim_script.py`; module object `se` already defined there). Use the REAL keyword-only signature:
```python
def test_one_segment_per_panel_aligned():
    beats = [{"group_id": 7, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"],
              "panel_narration": [
                  {"scene_file": "p1.jpg", "line": "He steps in."},
                  {"scene_file": "p2.jpg", "line": "The quest window flares."},
                  {"scene_file": "p3.jpg", "line": "Numbers tick down."}]}]
    payload = {"beats": [{"group_id": 7, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=beats, payload=payload, word_target=120,
        genre_mode="action", proper_case=None, wpm=170, microbeats=False)
    assert len(sec["shots"]) == 3                            # one shot per panel
    # match the real per-shot scene-file key after _normalize_shots (scene_files):
    assert [s.get("scene_files") for s in sec["shots"]] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert len(sec["script_paragraphs"]) == 3                # one paragraph per panel
    assert len(sec["tts_paragraphs_v3"]) == 3                # contract: paras==tts==shots
    assert any("quest window" in p.lower() for p in sec["script_paragraphs"])
```

- [ ] **Step 1b: segment_id contract (subprocess)** — ⚠️ NOT `tests/test_b2_segment_id.py` (that runs **`timeline_planner.py`** and asserts on `render.plan.json`'s `timeline[]`, not script_expander output). The correct host is the existing `script_expander.py` CLI subprocess test in `tests/test_verbatim_script.py` (`test_cli_verbatim_runs_without_openai_key`, ~`:443`). Add a sibling CLI test there: build an input beats manifest with a 1-group 3-panel `panel_narration`, run `script_expander.py` as a process (verbatim path, no `--microbeats`), read `manifest.script.json`, and assert the emitted `segment_id`s are exactly `["g0007_p00", "g0007_p01", "g0007_p02"]` (contiguous, one per panel). Follow the existing CLI test's subprocess/tmp-dir pattern; do NOT call the function in-memory here.

- [ ] **Step 2: Run** → FAIL (still one paragraph per beat / wrong shot count).

- [ ] **Step 3: Implement** — in the `for idx, b in enumerate(chunk)` loop, before the microbeat block, branch on `panel_narration`:
```python
        panel_lines = b.get("panel_narration") or []
        if panel_lines and not microbeats:
            items = [(str(p.get("line") or "").strip(),
                      [str(p.get("scene_file") or "").strip()])
                     for p in panel_lines if str(p.get("line") or "").strip()]
        else:
            # legacy A/B path: single paragraph, or word-count microbeat split
            panel_files = _selected_scene_files_for_microbeats(b, payload_beat)
            max_parts = max(1, len(panel_files))
            parts = (_split_recap_microbeats(text, max_words=microbeat_max_words,
                                             max_parts=max_parts) if microbeats else [text])
            items = [(p, _scene_for_microbeat(panel_files, i, len(parts)))
                     for i, p in enumerate(parts or ["The scene continues."])]
        for line, sfiles in items:
            t2, had = normalize_caps_for_tts(line, proper_case)
            paras.append(t2); shout_flags.append(had); para_beats.append(b)
            shots.append(_build_microbeat_shot(payload_beat, t2, sfiles, wpm=wpm))
```
Then make shot normalization unconditional (the per-panel path always builds `shots`):
```python
        shots = _normalize_shots(shots)
```
(Drop the `if microbeats: … else _build_default_shots_from_payload` branch from the live path; keep `_build_default_shots_from_payload` only as the legacy fallback when no `panel_narration` exists, for old manifests.)

- [ ] **Step 4: Run** `.eval_venv/bin/python -m pytest tests/test_script_coverage.py tests/test_b2_segment_id.py tests/test_verbatim_script.py tests/test_script_chrome_opener.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add tools/script_expander.py tests/test_script_coverage.py tests/test_b2_segment_id.py
git commit -m "feat(script): one segment per panel from panel_narration (microbeats behind flag)"
```

---

## Chunk 6: planner identity, QA coverage, heal, config

### Task 6a: timeline_planner — verify per-segment pick collapses to identity

With one panel per segment, `_pick_for_segment` (`:1351`) returns that panel. No code change expected; add a guard test and keep `inject_missing_protected`.

- [ ] Write a test (`tests/test_timeline_selection.py`): a plan built from one-panel segments yields exactly one cut per segment, each cut showing that segment's panel. Run; if green, no change. If a segment shows >1 or 0 panels, fix `_pick_for_segment` to return `scene_files[:1]`. Commit (test only, or test+fix).

### Task 6b: prep_qa — per-panel coverage + system-shown

There is already a `system_card_dropped` WARN emitted by the `_is_title_card` **OCR heuristic** in `story_flags` (`tools/prep_qa.py:~891`, which exempts `panel_kind=="caption"`). The new check is **authoritative** because it keys on the stamped `panel_kind=="system"` (no regex); the old heuristic stays as harmless belt-and-suspenders. Do not remove it in this task (removal is part of the post-validation cleanup in Ch7).

- [ ] Add `tests/test_prep_qa.py` cases (module object already defined in that file): (1) every kept `story`/`system` panel appears in exactly one cut → no `uncovered`/`montage` flag; (2) a `system` panel (by stamped `panel_kind`) missing from the cuts raises an ERROR flag (new `system_card_unshown`); (3) a true author `caption` folded into a neighbor is NOT flagged. Run → FAIL.
- [ ] Implement in `tools/prep_qa.py`: add `system_coverage_flags(beats_obj, plan)` that asserts each `panel_kind=="system"` file from the understanding appears in `iter_shown_cuts` (ERROR `system_card_unshown` if not). Wire it into the QA aggregation alongside the existing flags. Defer the verdict to the stamped `panel_kind` (no independent regex). Drop any microbeat-specific coverage assumptions. Run → PASS. Commit.

### Task 6c: narration_heal — re-narrate the group's panel_narration

- [ ] `tests/test_narration_heal.py`: a QA-flagged group is re-narrated and the rebuilt beat still has `len(panel_narration)==len(scene_files)` (every panel keeps a line; none dropped). Run → FAIL.
- [ ] Implement: the heal re-narrate path rebuilds `panel_narration` for the flagged group (reuse `align_panel_narration` so the invariant holds) and the joined `narration`. Run → PASS. Commit.

### Task 6d: config / pipeline — switchable flag, per-panel default

- [ ] `tests/test_cli.py` (or `tests/catalog`): assert the scripted stage runs per-panel by default (no `--microbeats` needed) and that setting `narration_microbeats=true` still routes the legacy flag. Run → FAIL.
- [ ] In `studio/config.py:56` keep `narration_microbeats` (default stays a switch; per-panel is the new default behaviour because `panel_narration` now exists). In `studio/pipeline.py:305-306`, only pass `--microbeats` when the flag is set (already true) — confirm the per-panel path is taken when it is off. Run → PASS. Commit.

- [ ] **Full-suite gate:** `.eval_venv/bin/python -m pytest -q` → all green (re-confirm count; CLAUDE.md's "170" is stale, last session showed ~762). Commit any test-list fixups.

---

## Chunk 7: Ch1 validation (operational — human gate, no unit tests)

> Per `confirm-upstream-before-expensive-downstream` + `qa-scan-before-rendering`: prove Ch1 before any batch. The adding of `system` is a contract change → understanding MUST be re-run (delete `understood.json`).

- [ ] **Deploy to the Mini:** `git push`, then on `jidailabs@10.88.0.1`: `cd <repo> && git fetch && git reset --hard origin/main`. (Worker stays paused.)
- [ ] **Re-run understanding** for Ch1 (series 2): delete `ongoing/<nano-slug>/Chapter_1/manifest.panels.understood.json`, then run the `grouped` stage so `panel_understand` re-labels panels (system cards now get `system`). Confirm in the log that p000114 and the other cards are `system`.
- [ ] **Run beated → scripted → planned** on Ch1 (per-panel default). Confirm: every kept panel has its own line; `len(panel_narration)==len(scene_files)` per beat; `segment_id`s are contiguous `g####_p##`.
- [ ] **Re-plan** and grep the cuts for p000114 ("7TH GENERATION NANO MACHINE…") and p000113 ("SKY CORPORATION") — both must appear with their text.
- [ ] **Produce the side-by-side artifact** vs the current group/microbeat output (pin the old baseline to a microbeats-**on** prepare for an apples-to-apples clip count): every panel explained, flowing, cinematic, cards shown, no repeats, no misalignment. Verify TTS clip count ≈ unchanged.
- [ ] **Human review gate (STOP):** present the side-by-side to the user. Do **not** proceed to batch until approved.
- [ ] **After approval:** reset ch1-20 (`/tmp/start_nano_20.py`), bootstrap the worker, run the 20-chapter validation + average, then the full 317-chapter run. Once per-panel is confirmed across the 20, remove the legacy microbeat functions + the `narration_microbeats` flag in a cleanup commit.

---

## Done-when (goal-backward checks)

- Every surviving story/system panel has exactly one narration line and one cut (prep_qa green; Ch1 visual check).
- `segment_id` (`g####_p##`) byte-identical contract preserved (`test_b2_segment_id.py` green).
- System cards of any shape (the 6 spec exemplars) are labeled `system` and shown (Ch1: p000113 + p000114 in cuts).
- No panel can be silently dropped (repair-fill invariant asserted; `test_panel_narration.py` green).
- Full `pytest` suite green; Ch1 side-by-side approved by the user before any batch.
