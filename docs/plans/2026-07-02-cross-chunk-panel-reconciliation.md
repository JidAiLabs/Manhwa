# Cross-Chunk Panel Seam Reconciliation — Implementation Plan (TDD)

**Date:** 2026-07-02
**Spec (source of truth):** `docs/plans/specs/2026-07-02-cross-chunk-panel-reconciliation-design.md` (commit `84bf1b4`) — read §1 (mechanism), §3 (detector rule), §3.1 (chains), §5 (reassembly / overlap-trim at the TRUE seam), §5.1 (`chunk_as_panel` exemption), §6 (acceptance), §7 (non-goals).
**Baseline suite:** `1198 passed, 1 skipped`.
**Execute with:** `subagent-driven-development` (each Task below is a self-contained TDD unit; dispatch one subagent per Task in order).

---

## Goal

A chunk seam can bisect one tall webtoon panel into two near-duplicate slices (bottom of chunk N + top of chunk N+1). YOLO, running per-chunk, detects each slice as its own panel → the same drawing is materialized as two scenes → two montage cuts → the `cross_dup` ERROR in `prep_qa`. This feature adds a **new scene-level step** that detects seam-bisected slices geometrically and re-assembles them into ONE panel, run **inside `_stage_scened`** (after `panels_to_scenes`, before `visioned`) so vision/understanding/narration only ever see the corrected set. Result: `cross_dup` → ~0, the same art never shown twice, **zero narration-pipeline changes**.

## Architecture

```
stitch → detect (per-chunk YOLO) → expand-to-gutters → scenes (per-chunk crops)
        → [RECONCILE cross-chunk seams]  ← NEW (tools/reconcile_seam_panels.py)
        → vision → understanding → grouping → beats → narration → script → tts → plan → render
```

Three touch-points:

1. **NEW `tools/reconcile_seam_panels.py`** — reads `manifest.scenes.json` + `manifest.stitch.json` (`adaptive.overlap_px`) + the chunk images (via each scene record's `chunk_path`), detects seam chains, re-crops each chain into one merged panel, rewrites `manifest.scenes.json` in place, deletes orphan slice JPGs, stamps `reconciled_seam: true`. Idempotent (a re-run finds no chains → no changes).
2. **`studio/pipeline.py` `_stage_scened`** — append one `_run_tool("reconcile_seam_panels.py", …)` call right after the `panels_to_scenes.py` call. **No new catalog status** → `STATUS_ORDER`, worker, dashboard untouched → **no daemon restart** (tool + `pipeline.py` subprocess are fresh on `git pull`).
3. **`tools/prep_qa.py`** — `image_flags` gains a `reconciled` kwarg; the `chunk_as_panel` height check (`h > 8000`, `:187`) is skipped for reconciled panels (they are tall BY DESIGN). `main()` builds the reconciled file-set from `manifest.scenes.json` (it already reads that file) and passes the flag through.

## Tech stack

Python 3.12, venv `.eval_venv` (PIL/Pillow, numpy). Tests: `pytest`. No new dependencies. `dhash64`/`hamming64` are reused from `tools/panels_to_scenes.py` (byte-identical to the hashes already stored in scene records).

---

## Conventions (READ before any Task)

- **Run tests:** `V=.eval_venv/bin/python; $V -m pytest -q`
  - Single file: `$V -m pytest -q tests/test_reconcile_seam_panels.py`
- **Worktree note (if executing in a git worktree):** `tests/test_ocr_chrome.py` subprocesses a **relative** `.eval_venv/bin/python` (`tests/test_ocr_chrome.py:28`). A fresh worktree has no `.eval_venv`. Symlink it in before running the suite:
  ```bash
  ln -s /Users/anka/repos/Manhwa/.eval_venv .eval_venv   # run from the worktree root
  ```
- **Edit plain files only.** Never edit `*-BAK.py`, `*XXX.py`, `*X.py` (frozen snapshots). Canonical = plain name.
- **`segment_id` rule:** `segment_id` (`g####_p##`) must stay byte-identical across `script_expander → tts → timeline_planner`. This feature runs **upstream of vision**, well before any `segment_id` is minted, so it cannot break alignment — but do not touch any code that mints or reads `segment_id`.
- **Manifest is the API.** `manifest.scenes.json` `scenes[]` records key downstream by `out_file` / `panel_id`; nothing hardcodes a panel count. When you rewrite a record, keep every existing field present (only change `box_px_xyxy`/`box_norm`/`w`/`h`/`dhash64`/`out_file` for the survivor and add the `reconciled_seam`/`merged_from` markers).
- **Commit after each Task** (frequent commits). Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- **TDD discipline:** write the failing test FIRST, run it, see it fail for the right reason, then implement, then see it pass. Never write impl before a red test.

---

## File structure

| File | Action |
|------|--------|
| `tools/reconcile_seam_panels.py` | **create** — detector + reassembly + CLI |
| `tests/test_reconcile_seam_panels.py` | **create** — unit tests (detector, chain, veto, reassembly, idempotence) |
| `studio/pipeline.py` | **edit** — 1 line block in `_stage_scened` (after `:166`) |
| `tests/test_pipeline.py` | **edit** — 1 test: reconcile is invoked between scenes and vision |
| `tools/prep_qa.py` | **edit** — `image_flags` `reconciled` kwarg + `main()` plumbing |
| `tests/test_prep_qa.py` | **edit** — 1 test: reconciled tall panel is exempt |

---

## Verified anchors (live code, 2026-07-02 — all confirmed present)

- `tools/panels_to_scenes.py`: `dhash64` `:269`, `hamming64` `:281`, per-chunk crop `chunk_im.crop(tuple(box_xyxy))` `:914`, naive `chunk_global_y0` sum `:782-792`, scene record `:1004-1035` (fields: `panel_id, chunk_file, chunk_path, chunk_w, chunk_h, chunk_global_y0, panel_index_in_chunk, recovered, part_index, box_px_xyxy, box_norm, out_file, out_path, w, h, blank_score, edge_density, trim, protected_spans_local, dhash64, split`). `chunk_path` is stored **absolute** (`:1008`).
- `tools/chunk_stitch_adaptive.py`: `--overlap-px` default `700`; manifest carries `adaptive.overlap_px`.
- `studio/pipeline.py`: `_stage_scened` `:155-166` (insertion point = right after `:166`), `_stage_visioned` `:169`, `_STAGE_TABLE` `:431-441`. `_run_tool(script_name, args_list)` `:44`.
- `studio/catalog/models.py`: `STATUS_ORDER` `:3-4` (`scened`→`visioned`) — **not modified**.
- `tools/prep_qa.py`: `image_flags` `:160` with the `h > 8000` `chunk_as_panel` ERROR `:187-197`; `image_flags` call site `:1673-1676`; `parent_scene()` `:85`; `main()` already opens `manifest.scenes.json` at `:1559-1568` (reads `recovered`) — reuse this loop for `reconciled_seam`; `cross_dup_flags` `:295`/`:308`, invoked `:1686`.

**Real seam geometry** (verified on `ongoing/the-tutorial-tower-of-the-advanced-player/Chapter_1/manifest.scenes.json` — used as concrete fixture values):
- chunk_0003: `chunk_h=13398`, `chunk_global_y0=29806`, bottom panel `box_px_xyxy=[0,2115,1200,13398]` → `y1=13398` **touches** the forced edge (gap 0).
- chunk_0004: `chunk_h=16026`, `chunk_global_y0=43204`, top panel `box_px_xyxy=[0,0,1200,825]` → `y0=0` **touches** the top edge.
- Contiguity: `A.gy0+A.y1 = 29806+13398 = 43204`; `B.gy0+B.y0 = 43204+0 = 43204`. Δ=0 → **SEAM**.
- Negative (gutter cut): chunk_0001 bottom `y1=13883`, `chunk_h=14457` → gap `574 > 24` → not a seam. chunk_0002 bottom `y1=15120`, `chunk_h=15349` → gap `229 > 24` → not a seam.

---

## Chunk 1 — the tool + its unit tests

Build the pure detector first (fast TDD), then the pure reassembly function, then the CLI wrapper.

### Task 1.1 — Detector: `find_seam_chains` (pure, unit-tested)

**RED.** Create `tests/test_reconcile_seam_panels.py`:

```python
"""
tests/test_reconcile_seam_panels.py

TDD for tools/reconcile_seam_panels.py — cross-chunk seam detection + reassembly.
The detector operates purely on scenes[] records (no image I/O). The reassembler
is unit-tested with small synthetic PIL images in tmp_path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "reconcile_seam_panels",
    Path(__file__).resolve().parent.parent / "tools" / "reconcile_seam_panels.py",
)
rsp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rsp)  # type: ignore[union-attr]


# ---- fixture builder --------------------------------------------------------

def _scene(panel_id, chunk_file, chunk_h, gy0, box, *, dhash=0, w=1200, h=None):
    """Minimal scene record carrying only the fields the detector reads."""
    x0, y0, x1, y1 = box
    return {
        "panel_id": panel_id,
        "chunk_file": chunk_file,
        "chunk_path": f"/fake/{chunk_file}",
        "chunk_h": chunk_h,
        "chunk_w": w,
        "chunk_global_y0": gy0,
        "box_px_xyxy": [x0, y0, x1, y1],
        "w": (x1 - x0) if w is None else w,
        "h": (y1 - y0) if h is None else h,
        "dhash64": dhash,
        "out_file": f"{panel_id}.jpg",
    }


def _ch1_like_scenes():
    """Real tutorial-tower ch1 geometry: chunk_0003→chunk_0004 is a true seam;
    chunk_0001→0002 and chunk_0002→0003 are clean gutter cuts (negatives)."""
    return [
        _scene("p01", "chunk_0001.jpg", 14457, 0,     [0, 12644, 1072, 13883]),  # gap 574
        _scene("p02", "chunk_0002.jpg", 15349, 14457, [0, 0, 1200, 1007]),
        _scene("p03", "chunk_0002.jpg", 15349, 14457, [309, 13352, 1134, 15120]),  # gap 229
        _scene("p04", "chunk_0003.jpg", 13398, 29806, [339, 10, 1159, 471]),
        _scene("p05", "chunk_0003.jpg", 13398, 29806, [0, 2115, 1200, 13398]),  # A: y1==chunk_h
        _scene("p06", "chunk_0004.jpg", 16026, 43204, [0, 0, 1200, 825]),        # B: y0==0
    ]


# ---- detector ---------------------------------------------------------------

def test_detects_the_true_seam_pair():
    chains = rsp.find_seam_chains(_ch1_like_scenes())
    # exactly one chain, the p05/p06 seam, in stitch order
    assert chains == [["p05", "p06"]]


def test_gutter_cut_pairs_are_not_merged():
    # remove the true seam so only the two gutter-cut adjacencies remain
    scenes = [s for s in _ch1_like_scenes() if s["panel_id"] not in ("p05", "p06")]
    assert rsp.find_seam_chains(scenes) == []


def test_high_dhash_pair_is_vetoed():
    scenes = _ch1_like_scenes()
    for s in scenes:
        if s["panel_id"] == "p05":
            s["dhash64"] = 0
        if s["panel_id"] == "p06":
            s["dhash64"] = (1 << 40) - 1  # popcount 40 > DHASH_VETO(20)
    assert rsp.find_seam_chains(scenes) == []


def test_three_chunk_chain_is_one_component():
    # a very tall panel across 3 chunks: the middle chunk's SOLE panel touches
    # BOTH edges (y0~0 AND y1~chunk_h) -> connected component of size 3.
    scenes = [
        _scene("a", "c1.jpg", 10000, 0,     [0, 3000, 1200, 10000]),  # bottom touches
        _scene("m", "c2.jpg", 10000, 10000, [0, 0, 1200, 10000]),     # touches BOTH
        _scene("b", "c3.jpg", 10000, 20000, [0, 0, 1200, 4000]),      # top touches
    ]
    assert rsp.find_seam_chains(scenes) == [["a", "m", "b"]]
```

Run it — expect failure (`ModuleNotFoundError` / `AttributeError: find_seam_chains`):
```bash
V=.eval_venv/bin/python; $V -m pytest -q tests/test_reconcile_seam_panels.py
```

**GREEN.** Create `tools/reconcile_seam_panels.py` (detector portion). Header + constants + `find_seam_chains`:

```python
#!/usr/bin/env python3
"""
reconcile_seam_panels.py

A chunk seam can bisect ONE tall panel into two near-duplicate slices (bottom of
chunk N + top of chunk N+1). Detection runs per-chunk, so each slice becomes its
own scene -> the same art shows twice -> the `cross_dup` QA ERROR.

This tool runs at the SCENE level (after panels_to_scenes, before vision). It:
  1. detects seam-bisected slice CHAINS geometrically (find_seam_chains),
  2. re-crops each chain into ONE merged panel from the chunk images
     (trimming the shared overlap band at the TRUE seam), and
  3. rewrites manifest.scenes.json + scenes/ in place, deleting orphan slices
     and stamping `reconciled_seam: true` on the survivor.

Idempotent: a re-run finds no chains and changes nothing.

See docs/plans/specs/2026-07-02-cross-chunk-panel-reconciliation-design.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Reuse the EXACT dhash/hamming the scene records were hashed with.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
for _p in (_TOOLS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from panels_to_scenes import dhash64, hamming64  # noqa: E402

# Tolerances — small ABSOLUTE pixel values (loose fractions re-introduce false
# merges; see spec §3.2, §8). Chunk pixels.
EDGE_TOL_PX = 24    # A.y1 within this of chunk_h[N]; B.y0 within this of 0
SEAM_TOL_PX = 48    # EDGE_TOL_PX * 2; contiguity slack in stacked global-y
DHASH_VETO = 20     # VETO-ONLY: reject a match whose slice dhashes are FARTHER
                    # apart than this. Never triggers a merge.


def _chunk_meta(scenes: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """chunk_file -> {chunk_h, gy0} (chunk_h/gy0 are constant within a chunk)."""
    meta: Dict[str, Dict[str, int]] = {}
    for s in scenes:
        cf = str(s.get("chunk_file") or "")
        if cf and cf not in meta:
            meta[cf] = {"chunk_h": int(s.get("chunk_h") or 0),
                        "gy0": int(s.get("chunk_global_y0") or 0)}
    return meta


def _y0(s: Dict[str, Any]) -> int:
    return int(s["box_px_xyxy"][1])


def _y1(s: Dict[str, Any]) -> int:
    return int(s["box_px_xyxy"][3])


def find_seam_chains(
    scenes: List[Dict[str, Any]],
    *,
    edge_tol: int = EDGE_TOL_PX,
    seam_tol: int = SEAM_TOL_PX,
    dhash_veto: int = DHASH_VETO,
) -> List[List[str]]:
    """Return seam-bisected panel CHAINS as lists of panel_id in stitch order.

    Only chains of length >= 2 are returned (a merge target). A chain is a
    connected component of pairwise seam links between adjacent chunks.

    A pair (A = bottommost panel of chunk N, B = topmost of chunk N+1) links iff:
      1. A touches the forced bottom edge:  chunk_h[N] - A.y1 <= edge_tol
      2. B touches the top edge:            B.y0 <= edge_tol
      3. contiguous stacked global-y:       |(A.gy0+A.y1) - (B.gy0+B.y0)| <= seam_tol
      4. NOT vetoed by a HIGH dhash distance: hamming(A,B) <= dhash_veto
    """
    meta = _chunk_meta(scenes)
    # chunks in stitch order = by naive global-y offset
    order = sorted(meta, key=lambda cf: meta[cf]["gy0"])
    by_chunk: Dict[str, List[Dict[str, Any]]] = {cf: [] for cf in order}
    for s in scenes:
        cf = str(s.get("chunk_file") or "")
        if cf in by_chunk:
            by_chunk[cf].append(s)

    # union-find over panel_ids
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    linked: set = set()
    for n in range(len(order) - 1):
        cf_n, cf_m = order[n], order[n + 1]
        rows_n, rows_m = by_chunk[cf_n], by_chunk[cf_m]
        if not rows_n or not rows_m:
            continue
        a = max(rows_n, key=_y1)          # bottommost of chunk N
        b = min(rows_m, key=_y0)          # topmost of chunk N+1
        if meta[cf_n]["chunk_h"] - _y1(a) > edge_tol:        # cond 1
            continue
        if _y0(b) > edge_tol:                                 # cond 2
            continue
        ga = meta[cf_n]["gy0"] + _y1(a)
        gb = meta[cf_m]["gy0"] + _y0(b)
        if abs(ga - gb) > seam_tol:                           # cond 3
            continue
        if hamming64(int(a["dhash64"]), int(b["dhash64"])) > dhash_veto:  # cond 4 veto
            continue
        union(a["panel_id"], b["panel_id"])
        linked.add(a["panel_id"])
        linked.add(b["panel_id"])

    # gather components, ordered by each member's stacked global-y
    def stack_y(pid: str) -> int:
        s = next(x for x in scenes if x["panel_id"] == pid)
        return meta[str(s["chunk_file"])]["gy0"] + _y0(s)

    comps: Dict[str, List[str]] = {}
    for pid in linked:
        comps.setdefault(find(pid), []).append(pid)
    chains = [sorted(members, key=stack_y) for members in comps.values()]
    chains.sort(key=lambda ch: stack_y(ch[0]))
    return [ch for ch in chains if len(ch) >= 2]


def main() -> int:  # pragma: no cover  (filled in Task 1.3)
    raise SystemExit("main() implemented in Task 1.3")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

Run tests — all 4 detector tests pass:
```bash
V=.eval_venv/bin/python; $V -m pytest -q tests/test_reconcile_seam_panels.py
# expect: 4 passed
```

**COMMIT:**
```bash
git add tools/reconcile_seam_panels.py tests/test_reconcile_seam_panels.py
git commit -m "feat(reconcile): seam-chain detector (pure, geometric) + tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.2 — Reassembly: `reassemble_slices` (pure, unit-tested)

**RED.** Append to `tests/test_reconcile_seam_panels.py`:

```python
# ---- reassembly -------------------------------------------------------------

def test_reassemble_trims_overlap_band_and_sums_height():
    OVERLAP = 30
    # A = 100px solid red. B = [30px green overlap band == A's tail] + 80px blue.
    a = Image.new("RGB", (40, 100), (255, 0, 0))
    b = Image.new("RGB", (40, 110), (0, 0, 255))
    for y in range(OVERLAP):                       # paint B's top band green
        for x in range(40):
            b.putpixel((x, y), (0, 255, 0))
    # B.y0 == 0 -> top_trim = OVERLAP - 0 = OVERLAP; A's top_trim = 0
    merged = rsp.reassemble_slices([a, b], [0, OVERLAP])
    assert merged.size == (40, 100 + 110 - OVERLAP)   # 180: A.h + B.h - overlap
    # the green overlap band must be gone (appears zero times)
    px = list(merged.getdata())
    assert (0, 255, 0) not in px
    # red on top, blue on the bottom row
    assert merged.getpixel((0, 0)) == (255, 0, 0)
    assert merged.getpixel((0, merged.height - 1)) == (0, 0, 255)


def test_reassemble_partial_edge_offset():
    # B.y0 = 5 (topmost box began 5px below the top edge, within EDGE_TOL) ->
    # top_trim = OVERLAP - B.y0 leaves NO sliver of the repeated band.
    OVERLAP = 30
    a = Image.new("RGB", (10, 50), (10, 10, 10))
    b = Image.new("RGB", (10, 60), (20, 20, 20))
    top_trim = OVERLAP - 5
    merged = rsp.reassemble_slices([a, b], [0, top_trim])
    assert merged.height == 50 + (60 - top_trim)
```

Run — fails (`AttributeError: reassemble_slices`).

**GREEN.** Add to `tools/reconcile_seam_panels.py` (above `main`):

```python
def reassemble_slices(slices: List[Image.Image], top_trims: List[int]) -> Image.Image:
    """Vertically stack *slices*, trimming top_trims[k] px off slice k's top.

    Slices share a source column (same width), aligned by construction. The trim
    on each interior seam removes the duplicated overlap band at the TRUE seam
    (top_trims[k] = OVERLAP_PX - B.y0 for the k-th slice; 0 for the first).
    """
    assert len(slices) == len(top_trims) and slices, "slices/top_trims mismatch"
    parts: List[Image.Image] = []
    for im, trim in zip(slices, top_trims):
        t = max(0, min(int(trim), im.height))
        parts.append(im.crop((0, t, im.width, im.height)) if t else im)
    width = max(p.width for p in parts)
    height = sum(p.height for p in parts)
    out = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for p in parts:
        out.paste(p.convert("RGB"), (0, y))
        y += p.height
    return out
```

Run — 6 passed total. **COMMIT:**
```bash
git add tools/reconcile_seam_panels.py tests/test_reconcile_seam_panels.py
git commit -m "feat(reconcile): overlap-trimming slice reassembly + tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.3 — CLI: re-crop from chunks, rewrite manifest, delete orphans, idempotent

**RED.** Append an end-to-end test that builds a tiny 2-chunk episode on disk and runs `reconcile_episode` (the CLI core, factored for testability):

```python
# ---- episode-level reconcile (image I/O in tmp_path) ------------------------

def _write_episode(tmp_path, overlap=30):
    """Two chunk images + a scenes manifest whose bottom-of-c1 / top-of-c2
    panels form a true seam. Returns (ep_dir, scenes_manifest_path)."""
    ep = tmp_path / "ep"
    (ep / "scenes").mkdir(parents=True)
    ch1 = Image.new("RGB", (40, 100), (200, 30, 30))   # chunk_h = 100
    ch2 = Image.new("RGB", (40, 100), (30, 30, 200))   # chunk_h = 100
    # make c2's top `overlap` band a copy of c1's bottom band (shared pixels)
    band = ch1.crop((0, 100 - overlap, 40, 100))
    ch2.paste(band, (0, 0))
    ch1.save(ep / "c1.jpg"); ch2.save(ep / "c2.jpg")

    def scene(pid, cf, gy0, box, chunk_path):
        x0, y0, x1, y1 = box
        crop = Image.open(chunk_path).crop((x0, y0, x1, y1))
        of = f"{pid}.jpg"
        crop.save(ep / "scenes" / of)
        return {"panel_id": pid, "chunk_file": cf, "chunk_path": str(chunk_path),
                "chunk_w": 40, "chunk_h": 100, "chunk_global_y0": gy0,
                "panel_index_in_chunk": 0, "recovered": False, "part_index": 0,
                "box_px_xyxy": [x0, y0, x1, y1],
                "box_norm": [y0 / 100, x0 / 40, y1 / 100, x1 / 40],
                "out_file": of, "out_path": str(ep / "scenes" / of),
                "w": x1 - x0, "h": y1 - y0, "blank_score": 0.0,
                "edge_density": 0.1, "trim": {"trimmed": False},
                "protected_spans_local": [], "dhash64": 0,
                "split": {"enabled": True}}

    scenes = [
        scene("p_top", "c1.jpg", 0,   [0, 5, 40, 40],  ep / "c1.jpg"),   # a normal panel
        scene("p_a",   "c1.jpg", 0,   [0, 40, 40, 100], ep / "c1.jpg"),  # A: y1==chunk_h
        scene("p_b",   "c2.jpg", 100, [0, 0, 40, 70],  ep / "c2.jpg"),   # B: y0==0
    ]
    sm = ep / "manifest.scenes.json"
    sm.write_text(json.dumps({"count_scenes": len(scenes), "scenes": scenes}))
    (ep / "manifest.stitch.json").write_text(
        json.dumps({"adaptive": {"overlap_px": overlap}}))
    return ep, sm


def test_reconcile_episode_merges_and_rewrites(tmp_path):
    import json
    ep, sm = _write_episode(tmp_path, overlap=30)
    n = rsp.reconcile_episode(str(sm), str(ep / "manifest.stitch.json"),
                              str(ep / "scenes"))
    assert n == 1  # one chain merged
    out = json.loads(sm.read_text())
    ids = [s["panel_id"] for s in out["scenes"]]
    assert "p_b" not in ids                     # orphan slice record dropped
    assert not (ep / "scenes" / "p_b.jpg").exists()   # orphan JPG deleted
    surv = next(s for s in out["scenes"] if s["panel_id"] == "p_a")
    assert surv["reconciled_seam"] is True
    assert surv["merged_from"] == ["p_a", "p_b"]
    # merged height = A(60) + B(70) - (overlap 30 - B.y0 0) = 100
    assert surv["h"] == 100
    assert Image.open(ep / "scenes" / surv["out_file"]).height == 100
    assert out["count_scenes"] == 2


def test_reconcile_episode_is_idempotent(tmp_path):
    import json
    ep, sm = _write_episode(tmp_path, overlap=30)
    assert rsp.reconcile_episode(str(sm), str(ep / "manifest.stitch.json"),
                                 str(ep / "scenes")) == 1
    # second pass: no seam pairs left -> 0 merges, manifest unchanged in shape
    assert rsp.reconcile_episode(str(sm), str(ep / "manifest.stitch.json"),
                                 str(ep / "scenes")) == 0
    assert json.loads(sm.read_text())["count_scenes"] == 2
```

Add `import json` at the top of the test module if not already present.

Run — fails (`AttributeError: reconcile_episode`).

**GREEN.** Replace the placeholder `main` in `tools/reconcile_seam_panels.py` with `reconcile_episode` + a real `main`:

```python
def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconcile_episode(scenes_manifest: str, stitch_manifest: str,
                      scenes_dir: str) -> int:
    """Detect + merge seam chains in place. Returns the number of chains merged."""
    manifest = _load(scenes_manifest)
    scenes: List[Dict[str, Any]] = manifest.get("scenes") or []
    if not scenes:
        return 0
    overlap_px = int(((_load(stitch_manifest).get("adaptive") or {})
                      .get("overlap_px")) or 700)

    chains = find_seam_chains(scenes)
    if not chains:
        return 0

    by_id = {s["panel_id"]: s for s in scenes}
    meta = _chunk_meta(scenes)
    drop_ids: set = set()

    for chain in chains:
        members = [by_id[pid] for pid in chain]
        head = members[0]                       # survivor = top slice (chunk N)
        x0 = min(int(m["box_px_xyxy"][0]) for m in members)
        x1 = max(int(m["box_px_xyxy"][2]) for m in members)

        slices: List[Image.Image] = []
        top_trims: List[int] = []
        for i, m in enumerate(members):
            with Image.open(m["chunk_path"]) as im:
                im = im.convert("RGB")
                if i == len(members) - 1:
                    y_lo, y_hi = int(m["box_px_xyxy"][1]), int(m["box_px_xyxy"][3])
                else:
                    # interior/head slices run down to the forced cut edge
                    y_lo, y_hi = int(m["box_px_xyxy"][1]), int(m["chunk_h"])
                slices.append(im.crop((x0, y_lo, x1, y_hi)))
            top_trims.append(0 if i == 0 else max(0, overlap_px - int(m["box_px_xyxy"][1])))

        merged = reassemble_slices(slices, top_trims)
        out_path = os.path.join(scenes_dir, head["out_file"])
        merged.save(out_path, "JPEG", quality=92, optimize=True)

        head["w"] = merged.width
        head["h"] = merged.height
        head["box_px_xyxy"] = [x0, int(head["box_px_xyxy"][1]),
                               x1, int(head["box_px_xyxy"][1]) + merged.height]
        head["box_norm"] = [
            head["box_px_xyxy"][1] / max(1, int(head["chunk_h"])),
            x0 / max(1, int(head["chunk_w"])),
            head["box_px_xyxy"][3] / max(1, int(head["chunk_h"])),
            x1 / max(1, int(head["chunk_w"])),
        ]
        head["dhash64"] = int(dhash64(merged))
        head["reconciled_seam"] = True
        head["merged_from"] = list(chain)
        if isinstance(head.get("split"), dict):
            head["split"]["enabled"] = False

        for m in members[1:]:
            drop_ids.add(m["panel_id"])
            slice_path = os.path.join(scenes_dir, m["out_file"])
            if os.path.exists(slice_path):
                os.remove(slice_path)

    manifest["scenes"] = [s for s in scenes if s["panel_id"] not in drop_ids]
    manifest["count_scenes"] = len(manifest["scenes"])
    stats = manifest.setdefault("stats", {})
    stats["reconciled_seams"] = int(stats.get("reconciled_seams", 0)) + len(chains)
    with open(scenes_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return len(chains)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes-manifest", required=True)
    ap.add_argument("--stitch-manifest", required=True)
    ap.add_argument("--scenes-dir", required=True)
    args = ap.parse_args()
    if not os.path.exists(args.scenes_manifest):
        print(f"[reconcile] no scenes manifest at {args.scenes_manifest} — skip")
        return 0
    n = reconcile_episode(args.scenes_manifest, args.stitch_manifest, args.scenes_dir)
    print(f"[ok] reconcile_seam_panels: merged {n} seam chain(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run the full new-test file — all pass:
```bash
V=.eval_venv/bin/python; $V -m pytest -q tests/test_reconcile_seam_panels.py
# expect: 8 passed
```

**GATE (full suite):**
```bash
V=.eval_venv/bin/python; $V -m pytest -q
# expect: 1206 passed, 1 skipped   (1198 + 8 new)
```

**COMMIT:**
```bash
git add tools/reconcile_seam_panels.py tests/test_reconcile_seam_panels.py
git commit -m "feat(reconcile): episode reconcile (re-crop, rewrite, delete orphans, idempotent) + CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Chunk 2 — pipeline wiring + prep_qa exemption

### Task 2.1 — Wire `reconcile_seam_panels.py` into `_stage_scened`

**RED.** Add a test to `tests/test_pipeline.py`. First register the new script in `_tool_stub`'s `SCRIPT_TO_MARKER` so the stub touches the (already-existing) scenes marker harmlessly — add this entry after the `panels_to_scenes.py` line:

```python
        "reconcile_seam_panels.py":   "manifest.scenes.json",
```

Then add a test (place it near `TestRunChapterFullProgress`):

```python
def test_reconcile_runs_between_scened_and_visioned(self, tmp_path, monkeypatch):
    """reconcile_seam_panels.py is invoked AFTER panels_to_scenes.py and BEFORE
    vision_extract.py (scene-level seam merge, upstream of vision)."""
    import studio.pipeline as pipeline_mod

    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    ep_dir_ref = [ep_dir]
    tool_stub = _tool_stub(ep_dir_ref)
    detect_stub = _detect_stub(ep_dir_ref)
    monkeypatch.setattr(pipeline_mod, "_run_tool", tool_stub)
    monkeypatch.setattr("studio.detect.yolo_panels.detect_panels", detect_stub)
    _block_cred_gated_stages(monkeypatch, pipeline_mod)

    con = connect(tmp_path / "test.db")
    chapter = _make_chapter(con, ep_dir, status="downloaded")
    pipeline_mod.run_chapter(con, chapter, _make_cfg(tmp_path), now_fn=_now)

    calls = tool_stub.calls
    assert "reconcile_seam_panels.py" in calls
    assert (calls.index("panels_to_scenes.py")
            < calls.index("reconcile_seam_panels.py")
            < calls.index("vision_extract.py"))
```

Run — fails (`reconcile_seam_panels.py` not in `calls`):
```bash
V=.eval_venv/bin/python; $V -m pytest -q tests/test_pipeline.py::TestRunChapterFullProgress::test_reconcile_runs_between_scened_and_visioned
```

**GREEN.** In `studio/pipeline.py` `_stage_scened`, append the reconcile call after the `panels_to_scenes.py` `_run_tool(...)` (after line `:166`, still inside `_stage_scened`):

```python
    # SEAM RECONCILE (scene-level, upstream of vision): merge panels a chunk cut
    # bisected into two near-duplicate slices, so the same drawing is never shown
    # twice. In-place rewrite of manifest.scenes.json + scenes/. No new status.
    _run_tool("reconcile_seam_panels.py",
              ["--scenes-manifest", str(p["scenes_manifest"]),
               "--stitch-manifest", str(p["stitch"]),
               "--scenes-dir", str(p["scenes"])])
```

Run the new pipeline test + the two existing progression tests:
```bash
V=.eval_venv/bin/python; $V -m pytest -q tests/test_pipeline.py
# expect: all pass (existing + 1 new)
```

**COMMIT:**
```bash
git add studio/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): run seam reconcile in _stage_scened, after scenes, before vision

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2.2 — `prep_qa` `chunk_as_panel` exemption for reconciled panels

**RED.** Add to `tests/test_prep_qa.py` (next to `test_chunk_as_panel_blocks_a_whole_chunk`):

```python
def test_reconciled_tall_panel_is_exempt_from_chunk_as_panel():
    # a correctly reassembled seam panel is tall BY DESIGN (spec §5.1) -> the
    # reconciled marker exempts it from the h>8000 chunk_as_panel gate.
    flags = pq.image_flags("p000007.jpg", _art(9000, 800), [], doc=True,
                           reconciled=True, dims_entry={"w": 800, "h": 9000})
    assert not any(f["code"] == "chunk_as_panel" for f in flags)


def test_non_reconciled_tall_panel_still_blocks():
    # negative control: same tall crop with NO marker is still a BLOCKING ERROR.
    flags = pq.image_flags("p000008.jpg", _art(9000, 800), [], doc=True,
                           dims_entry={"w": 800, "h": 9000})
    assert any(f["code"] == "chunk_as_panel" and f["severity"] == "ERROR"
               for f in flags)
```

Run — the first fails (`image_flags() got an unexpected keyword argument 'reconciled'`):
```bash
V=.eval_venv/bin/python; $V -m pytest -q tests/test_prep_qa.py -k reconciled
```

**GREEN — two edits in `tools/prep_qa.py`:**

(a) `image_flags` signature (`:160`) — add the kwarg (keyword-only, after `min_art_score`/`vitem`):
```python
    vitem: Optional[Dict[str, Any]] = None,
    reconciled: bool = False,
) -> List[Dict[str, Any]]:
```
and guard the height check (`:187`):
```python
    if h > 8000 and not reconciled:
```
Add a one-line comment above it noting: a `reconciled_seam` panel is tall by design (spec §5.1) and is exempt; every non-reconciled panel is still gated.

(b) `main()` plumbing. In the existing scenes-manifest read loop (`:1559-1568`), also collect reconciled `out_file`s. Change that block to:
```python
    reconciled_files: set = set()
    sp_ = os.path.join(ep, "manifest.scenes.json")
    if os.path.exists(sp_):
        try:
            with open(sp_, "r", encoding="utf-8") as f:
                for sc in json.load(f).get("scenes") or []:
                    of = str(sc.get("out_file") or "")
                    if sc.get("recovered"):
                        vitems.setdefault(of, {})["recovered"] = True
                    if sc.get("reconciled_seam"):
                        reconciled_files.add(of)
        except Exception:
            pass
```
Then at the `image_flags(...)` call site (`:1673-1676`) pass the flag (match on the parent scene, since shown files may be split2 parts `_a`/`_b`):
```python
        flags.extend(image_flags(
            fname, img, boxes, doc=doc, dims_entry=d if d else None,
            sys=sys_panel, segment_id=seg_by_file[fname],
            vitem=vitems.get(parent_scene(fname)) or vitems.get(fname),
            reconciled=(parent_scene(fname) in reconciled_files
                        or fname in reconciled_files)))
```

Run — the two new tests pass:
```bash
V=.eval_venv/bin/python; $V -m pytest -q tests/test_prep_qa.py -k "reconciled or chunk_as_panel"
```

**GATE (full suite):**
```bash
V=.eval_venv/bin/python; $V -m pytest -q
# expect: 1209 passed, 1 skipped   (1206 + 1 pipeline + 2 prep_qa)
```

**COMMIT:**
```bash
git add tools/prep_qa.py tests/test_prep_qa.py
git commit -m "fix(prep_qa): exempt reconciled_seam panels from chunk_as_panel height gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Done — gate

**Unit acceptance (spec §6.1-3) — all green:**
- `find_seam_chains` finds the true seam pair on the ch1-geometry fixture and rejects both gutter-cut adjacencies (§6.1) and the high-dhash veto case (§3 cond 4).
- 3-chunk chain → one connected component of size 3 (§6.2).
- `reassemble_slices` trims the overlap band → height `A.h + B.h − (OVERLAP_PX − B.y0)`, no duplicated band (§6.3).
- `reconcile_episode` rewrites the manifest, deletes orphan slice JPGs, stamps `reconciled_seam`/`merged_from`, and is idempotent.
- `prep_qa` exempts a reconciled tall panel; a non-reconciled tall panel still ERRORs.

**Suite gate (must show before claiming done):**
```bash
V=.eval_venv/bin/python; $V -m pytest -q
# REQUIRED: 1209 passed, 1 skipped   (baseline 1198 + 11 new)
```
(If the pytest count differs from 1209, reconcile the delta before proceeding — do not hand-wave it.)

**MANUAL acceptance (spec §6.4 — deploy-time, in the away-run; NOT part of this plan's code):**
Reprocess a chapter with known seams from `detected` (delete `manifest.scenes.json` + `scenes/` and re-run scened → visioned → … so the fix + downstream re-materialize; spec §8 "Existing rendered chapters"). Then regenerate the plan + run `prep_qa` and confirm:
1. Panel count drops by *(total seam slices − reconciled panels)* — a 2-chunk seam is −1, a 3-chunk chain is −2 (§6.4). For Nano ch1's 6 seam pairs, ~`112 → ~106`.
2. `cross_dup` → ~0.
3. **No new `chunk_as_panel`** flag on the reconciled tall panels (the exemption works).
4. **No other new flags** (`montage_degenerate`, `blank_crop`, …).
5. Spot-check merged crops visually: one contiguous panel, no repeated overlap band, no seam sliver/gap.

**Non-goals reminder (spec §7):** do not touch stitch heuristics / `max_chunk_height` / overflow cap; do not touch any narration-side code (understanding, grouping, beats, punchup, script, TTS, plan); do not repurpose this to dedup legitimately-distinct panels (those never touch a forced chunk edge — leave them to `panels_to_scenes --dedupe` + `cross_dup`).

**Deploy note:** the changeset is `tools/reconcile_seam_panels.py` (new), `tools/prep_qa.py`, `studio/pipeline.py` — all tools / `pipeline.py` subprocesses → **fresh on `git pull`, no daemon restart** (`launchctl kickstart -k` only needed for `studio/worker.py` / `studio/dashboard/**`, which this does not touch). No new catalog status; `STATUS_ORDER` unchanged.
```
