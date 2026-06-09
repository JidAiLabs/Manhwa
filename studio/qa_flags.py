"""
studio/qa_flags.py

Automated QA "confidence instrument": consumes the manifest dicts the pipeline
already produces and emits per-scene / per-group flags plus a summary scorecard
scored against the SP2 (Scene & Bubble Quality) acceptance criteria.

This exists because the visual QA report (``studio/qa.py``) is eyeball-only — a
human had to personally spot every duplicate, text bubble, choppy run and
OCR-echo across dozens of cards. These functions turn the known defect classes
into computed signals so "is the QA confident enough?" becomes a number.

All functions are pure (no image I/O). Signals are drawn from fields that
already exist in the manifests:

  - near-duplicates  → ``manifest.scenes.json`` ``dhash64`` (perceptual hash;
                       catches CROSS-chunk dups the within-chunk geometric
                       dedupe cannot see)
  - text-dominated   → ``manifest.vision.json`` ``targets`` text_block bboxes
                       (the legacy ``text_coverage`` field is miscalibrated and
                       reads ~0 even for visible bubbles)
  - OCR-echo         → narration paragraph vs the scene's ``ocr_clean`` (content
                       rule #2: never repeat on-page text verbatim)
  - short-on-screen  → script ``shots[].duration_s`` / number of pictures
                       (the "min 3-4s per picture" rule)
  - scene-set drift  → scene_files referenced by groups vs scenes on disk/manifest

Public API
----------
    hamming64(a, b) -> int
    near_duplicate_pairs(scenes, *, max_hamming=8) -> list[dict]
    text_block_area_frac(vision_item) -> float
    longest_common_run(a, b, *, min_words=4) -> str
    seconds_per_scene(shot) -> float
    compute_flags(*, scenes, vision_items, groups, script,
                  source_page_count, min_sec_per_pic=3.5,
                  dup_hamming=8, text_frac=0.20,
                  density_per_page_max=3.0) -> dict
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Primitive signals
# ---------------------------------------------------------------------------

def hamming64(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit perceptual hashes."""
    return bin(int(a) ^ int(b)).count("1")


def near_duplicate_pairs(
    scenes: list[dict[str, Any]], *, max_hamming: int = 8
) -> list[dict[str, Any]]:
    """Return near-duplicate scene pairs by ``dhash64`` Hamming distance.

    Scenes lacking a ``dhash64`` are skipped. Compares ALL pairs (including
    cross-chunk), which is what catches duplicates the geometric within-chunk
    dedupe misses. Each result: ``{"a": file, "b": file, "hamming": int}``.
    """
    hashed = [
        (s.get("out_file") or s.get("scene_file"), s.get("dhash64"))
        for s in scenes
        if s.get("dhash64") is not None
    ]
    pairs: list[dict[str, Any]] = []
    for i in range(len(hashed)):
        fi, hi = hashed[i]
        for j in range(i + 1, len(hashed)):
            fj, hj = hashed[j]
            d = hamming64(hi, hj)
            if d <= max_hamming:
                pairs.append({"a": fi, "b": fj, "hamming": d})
    pairs.sort(key=lambda p: p["hamming"])
    return pairs


def text_block_area_frac(vision_item: dict[str, Any]) -> float:
    """Fraction of the frame covered by OCR ``text_block`` targets.

    Sums the (possibly overlapping) text_block bbox areas, clamped to 1.0. This
    is a *better* text-dominance proxy than the legacy ``text_coverage`` field,
    which reads near-zero even for clearly text-heavy bubbles. bbox is
    normalized ``[y0, x0, y1, x1]``.
    """
    total = 0.0
    for t in vision_item.get("targets") or []:
        if t.get("type") != "text_block":
            continue
        bbox = t.get("bbox") or []
        if len(bbox) != 4:
            continue
        y0, x0, y1, x1 = bbox
        total += max(0.0, y1 - y0) * max(0.0, x1 - x0)
    return min(1.0, total)


_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def longest_common_run(a: str, b: str, *, min_words: int = 4) -> str:
    """Longest contiguous run of words shared between *a* and *b*.

    Case- and punctuation-insensitive. Returns the matched run (joined from
    *a*'s words) if it is at least *min_words* long, else ``""``. Used to detect
    narration that repeats on-page OCR text verbatim (content rule #2).
    """
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return ""
    # classic DP longest-common-substring over token lists
    best_len = 0
    best_end_a = 0
    prev = [0] * (len(wb) + 1)
    for i in range(1, len(wa) + 1):
        cur = [0] * (len(wb) + 1)
        ai = wa[i - 1]
        for j in range(1, len(wb) + 1):
            if ai == wb[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len = cur[j]
                    best_end_a = i
        prev = cur
    if best_len < min_words:
        return ""
    return " ".join(wa[best_end_a - best_len:best_end_a])


def seconds_per_scene(shot: dict[str, Any]) -> float:
    """Predicted on-screen seconds per picture = duration_s / #scene_files."""
    files = shot.get("scene_files") or []
    if not files:
        return 0.0
    return float(shot.get("duration_s") or 0.0) / len(files)


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def _add(flags: dict[Any, list], key: Any, flag: dict[str, Any]) -> None:
    flags.setdefault(key, []).append(flag)


def compute_flags(
    *,
    scenes: dict[str, Any],
    vision_items: dict[str, Any],
    groups: dict[str, Any],
    script: dict[str, Any] | None,
    source_page_count: int,
    beats: dict[str, Any] | None = None,
    min_sec_per_pic: float = 3.5,
    dup_hamming: int = 8,
    text_frac: float = 0.20,
    density_per_page_max: float = 3.0,
) -> dict[str, Any]:
    """Compute per-scene / per-group QA flags and a summary scorecard.

    Parameters mirror the manifest dicts (already-parsed JSON). Returns::

        {
          "scorecard":   {... counts + pass/fail booleans ...},
          "scene_flags": {scene_file: [{"kind","detail"}, ...]},
          "group_flags": {group_id:   [{"kind","detail"}, ...]},
        }

    Scoring thresholds are parameters so the SP2 fixes can tighten them.
    """
    scene_list: list[dict] = scenes.get("scenes") or []
    vlist: list[dict] = vision_items.get("items") or []
    shots: list[dict] = groups.get("shots") or []

    scene_flags: dict[str, list] = {}
    group_flags: dict[Any, list] = {}

    # --- vision index ---------------------------------------------------
    vis_by_file: dict[str, dict] = {}
    for it in vlist:
        sf = it.get("scene_file")
        if sf:
            vis_by_file[str(sf)] = it

    # --- 1. near-duplicates (cross-chunk via dhash64) -------------------
    dup_pairs = near_duplicate_pairs(scene_list, max_hamming=dup_hamming)
    for p in dup_pairs:
        detail = f"≈ {p['b']} (hamming {p['hamming']})"
        _add(scene_flags, p["a"], {"kind": "near_duplicate", "detail": detail})
        _add(scene_flags, p["b"],
             {"kind": "near_duplicate", "detail": f"≈ {p['a']} (hamming {p['hamming']})"})

    # --- 2. text-dominated bubbles --------------------------------------
    text_dominated = 0
    for it in vlist:
        sf = str(it.get("scene_file") or "")
        frac = text_block_area_frac(it)
        if frac >= text_frac or it.get("text_only") is True:
            text_dominated += 1
            _add(scene_flags, sf,
                 {"kind": "text_dominated",
                  "detail": f"text area {frac:.2f}" + (" (text_only)" if it.get("text_only") else "")})

    # --- script index: group_id -> [{"text", "duration_s", "scene_files"}] ---
    narr_by_gid: dict[int, list[dict]] = {}
    if script:
        for section in script.get("sections") or []:
            paras = section.get("script_paragraphs") or []
            srefs = section.get("shots") or []
            n = min(len(paras), len(srefs))
            for i in range(n):
                ref = srefs[i] or {}
                gid = int(ref.get("group_id") or 0)
                if not gid:
                    continue
                para = paras[i]
                text = para if isinstance(para, str) else str((para or {}).get("text") or "")
                narr_by_gid.setdefault(gid, []).append({
                    "text": text,
                    "duration_s": float(ref.get("duration_s") or 0.0),
                    "scene_files": ref.get("scene_files") or [],
                })

    # --- 3. shown vs dropped — the REAL rendered montage --------------------
    # The video does NOT show every panel: timeline_planner keeps
    # floor(group_seconds / min_cut_sec) panels per group, dropping 'redundant'
    # ones first. So measure what actually renders (shown), what gets cut
    # (dropped), and flag ONLY panels that are still under min_sec after cutting
    # (a genuinely too-short shot) — not the script's pre-trim over-references.
    role_by_file: dict[str, str] = {}
    for beat in (beats or {}).get("beats") or []:
        for ent in beat.get("scene_selection") or []:
            role_by_file[str(ent.get("scene_file") or "")] = str(ent.get("role") or "keep")

    group_dur: dict[int, float] = {}
    for gid, entries in narr_by_gid.items():
        group_dur[gid] = sum(float(e.get("duration_s") or 0.0) for e in entries)

    shown_panels = 0
    dropped_panels = 0
    shown_under_min = 0
    panels_in_groups = 0
    for shot in shots:
        gid = int(shot.get("shot_id") or shot.get("group_id") or 0)
        sfiles = [str(x) for x in (shot.get("scene_files") or [])]
        if not sfiles:
            continue
        panels_in_groups += len(sfiles)
        dur = group_dur.get(gid, 0.0)
        kmax = max(1, int(dur // min_sec_per_pic)) if dur > 0 else len(sfiles)
        keepers = [sf for sf in sfiles if role_by_file.get(sf, "keep") == "keep"]
        shown = (keepers or sfiles)[:kmax]
        shown_set = set(shown)
        shown_panels += len(shown)
        dropped_panels += len(sfiles) - len(shown)
        for sf in sfiles:
            if sf not in shown_set:
                _add(scene_flags, sf, {"kind": "dropped",
                                       "detail": "cut from the video (over time budget / redundant)"})
        # genuinely too short: even after cutting, each shown panel is < min_sec
        if dur > 0 and shown and (dur / len(shown)) < min_sec_per_pic:
            shown_under_min += 1
            secs = dur / len(shown)
            for sf in shown:
                _add(scene_flags, sf, {"kind": "short_on_screen",
                                       "detail": f"{secs:.1f}s shown (shot too short for {min_sec_per_pic:g}s)"})

    # --- 4. OCR-echo (narration repeats on-page text verbatim) ----------
    ocr_echo = 0
    for gid, entries in narr_by_gid.items():
        # gather OCR for this group's scenes
        gshot = next((s for s in shots if int(s.get("shot_id") or s.get("group_id") or 0) == gid), None)
        gfiles = (gshot.get("scene_files") if gshot else None) or []
        ocr_blob = " ".join(
            (vis_by_file.get(str(sf), {}).get("ocr_clean") or "") for sf in gfiles
        )
        if not ocr_blob.strip():
            continue
        for ent in entries:
            run = longest_common_run(ent["text"], ocr_blob, min_words=4)
            if run:
                ocr_echo += 1
                _add(group_flags, gid,
                     {"kind": "ocr_echo", "detail": f'narration echoes OCR: "{run}"'})

    # --- 5. missing narration -------------------------------------------
    missing_narration = 0
    group_ids = [int(s.get("shot_id") or s.get("group_id") or 0) for s in shots]
    for gid in group_ids:
        if gid and gid not in narr_by_gid:
            missing_narration += 1
            _add(group_flags, gid, {"kind": "no_narration", "detail": "group has no narration paragraph"})

    # --- 6b. redundant panels marked by the Gemini scene-selection pass ------
    redundant_marked = 0
    for beat in (beats or {}).get("beats") or []:
        for ent in beat.get("scene_selection") or []:
            if str(ent.get("role") or "keep") == "redundant":
                redundant_marked += 1
                _add(scene_flags, str(ent.get("scene_file") or ""),
                     {"kind": "redundant",
                      "detail": str(ent.get("reason") or "marked redundant by selector")})

    # --- 6. scene-set drift (groups reference files not in scenes set) --
    on_disk = {str(s.get("out_file") or s.get("scene_file") or "") for s in scene_list}
    referenced = {str(sf) for s in shots for sf in (s.get("scene_files") or [])}
    drift_missing = sorted(referenced - on_disk)
    scene_set_drift = bool(drift_missing)

    # --- scorecard ------------------------------------------------------
    total_scenes = len(scene_list)
    scenes_per_page = (total_scenes / source_page_count) if source_page_count else 0.0

    scorecard = {
        "total_scenes": total_scenes,
        "source_pages": source_page_count,
        "scenes_per_page": round(scenes_per_page, 2),
        "groups": len(shots),
        "near_dup_pairs": len(dup_pairs),
        "text_dominated": text_dominated,
        "shown_panels": shown_panels,
        "dropped_panels": dropped_panels,
        "shown_under_min": shown_under_min,
        "ocr_echo": ocr_echo,
        "missing_narration_groups": missing_narration,
        "redundant_marked": redundant_marked,
        "scene_set_drift": scene_set_drift,
        "drift_missing_files": drift_missing,
        # pass/fail vs SP2 acceptance
        "density_ok": scenes_per_page <= density_per_page_max,
        "dup_ok": len(dup_pairs) == 0,
        "echo_ok": ocr_echo == 0,
        "pacing_ok": shown_under_min == 0,
        "narration_ok": missing_narration == 0,
        "sync_ok": not scene_set_drift,
    }
    scorecard["all_ok"] = all(
        scorecard[k] for k in ("density_ok", "dup_ok", "echo_ok", "pacing_ok", "narration_ok", "sync_ok")
    )

    return {
        "scorecard": scorecard,
        "scene_flags": scene_flags,
        "group_flags": group_flags,
        "near_dup_pairs": dup_pairs,
    }
