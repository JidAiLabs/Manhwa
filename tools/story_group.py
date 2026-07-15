#!/usr/bin/env python3
"""story_group.py — Pass 2 of the understanding-first pipeline.

Group panels by UNDERSTANDING, not by gutters. Reads the per-panel descriptions
(Pass 1, panel_understand) in reading order and segments them into STORY CONTEXT
SPANS: a span = a run of CONSECUTIVE panels that form one moment or idea. New
span at a scene/location change, a time jump, a flashback start/end, or a topic
shift; near-identical consecutive panels merge into one context span. Each span
is tagged segment (present|flashback|dream) + arc_label — so flashbacks are
native.

This REPLACES the position/threshold grouping in scene_group_builder.py. Output
is a byte-compatible manifest.groups.json: top-level `shots` with per-shot
`shot_id` (contiguous int) + `scene_files`, plus extra `segment`/`arc_label`
tags that downstream ignores safely (timeline can carry the flashback tag).

Coverage is an invariant: every non-chrome panel lands in exactly one shot
(repair_to_shots reconstructs a consecutive partition from the model's intent).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from gemini_narrative_pass import (                                   # noqa: E402
    load_json, _call_model_with_backoff)
from manifest_io import write_manifest                                # noqa: E402

_SEGMENTS = ("present", "flashback", "dream")

# Cap PANELS PER BEAT (not narration length). A single beat of 29/35 panels
# overflows the grouping/narration model (gemma4:26b) and parse-fails, which then
# collapses the whole group to one fallback shot. Keeping beats <= this many
# panels lets the model emit valid JSON per beat. Each panel still gets its own
# content-scaled narration downstream — this only bounds how many panels share
# one context span/shot. Operators can pass --max-beat-len 0 to disable.
DEFAULT_MAX_BEAT_LEN = 6

# GROUP_SCHEMA asks for INDEX RANGES, not filename echoes. The old schema made
# beats.scene_files an array of strings and forced the model to type out every
# panel filename per beat -- on a ~96-panel chapter (~30 beats) that is ~30
# separate filename arrays, and gemma4:26b died mid-enumeration on EVERY
# production attempt (raw captures cut off mid scene_files-array; one attempt
# came back near-empty). build_grouping_payload already numbers every panel
# (`n`, 0-based, stable reading order) -- a beat is by definition a CONSECUTIVE
# run in that same order, so it can be expressed as {from_index, to_index}
# (inclusive) plus the existing short tags, instead of echoing names back.
# Budget: a beat is now ~2 integers + 2 short strings, so a realistic
# ~30-beat/~96-panel chapter response is well under 1000 tokens (MEASURED:
# ~1.9KB for a 24-beat/96-panel synthetic response, see
# test_realistic_96_panel_range_response_is_well_under_output_budget) --
# trivial next to the ~4-5KB the old scheme couldn't even finish generating on
# the same chapter. expand_index_ranges() converts the model's ranges back
# into the classic {scene_files: [...]} shape immediately after parsing, so
# repair_to_shots and everything downstream never sees a range -- the
# shots/groups output stays byte-shape-identical to before this schema change.
# The beat shape is defined ONCE and shared by reference between the
# whole-chapter schema and the chunk schema below, so the two can never drift.
_BEATS_ITEMS_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "from_index": {"type": "INTEGER"},
        "to_index": {"type": "INTEGER"},
        "segment": {"type": "STRING", "enum": list(_SEGMENTS)},
        "arc_label": {"type": "STRING"},
        "why": {"type": "STRING"},
    },
    "required": ["from_index", "to_index"],
}

GROUP_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "chapter": {"type": "OBJECT", "properties": {
            "logline": {"type": "STRING", "minLength": 12},
            "premise": {"type": "STRING", "minLength": 12},
        }, "required": ["logline", "premise"]},
        "beats": {"type": "ARRAY", "items": _BEATS_ITEMS_SCHEMA},
    },
    "required": ["chapter", "beats"],
}

# CHUNK calls (the chunked fallback below) see a contiguous SLICE of the
# chapter, never the whole thing — they cannot honestly write a whole-chapter
# logline/premise; the dedicated spine call covers that. The old chunk path
# reused GROUP_SCHEMA anyway, so constrained decoding FORCED the model to
# fabricate a 12+-char spine per chunk that chunk_call_fn then discarded —
# wasted decode plus an invitation to invent chapter-level claims from partial
# evidence. Beats only; no chapter key at all.
BEATS_ONLY_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {"beats": {"type": "ARRAY", "items": _BEATS_ITEMS_SCHEMA}},
    "required": ["beats"],
}

# Shared instruction core: everything both the whole-chapter call and a chunk
# call need. SYSTEM (whole chapter) appends the spine ask; SYSTEM_CHUNK (a
# slice) appends the beats-only line instead — mirroring the schema split
# above, so the prompt and the schema always agree about whether a 'chapter'
# object is wanted.
_SYSTEM_CORE = (
    "You are a manhwa recap editor. You get a numbered sequence of panel "
    "descriptions from ONE chapter, in reading order. Segment them into STORY "
    "CONTEXT SPANS for the recap.\n"
    "A context span = a run of CONSECUTIVE panels that form one moment, idea, "
    "exchange, action burst, or slow reveal. Start a NEW span at: a scene or "
    "location change, a time jump, a FLASHBACK start OR end, or a clear topic "
    "shift. Group near-identical consecutive panels (e.g. a multi-panel action "
    "or a slow reveal) into ONE span.\n"
    "ORDER + FLOW — the beats must read as ONE story in reading order:\n"
    "  - A caption / monologue panel that INTRODUCES the moment right after it "
    "(e.g. 'ON THE DAY I FINISHED THE WEB NOVEL…' immediately before that event) "
    "belongs in the SAME beat as the art it introduces. Never strand an intro "
    "caption as its own separate beat sitting before the moment it sets up.\n"
    "  - Keep a flashback or dream as a CONTIGUOUS block. Do NOT bounce "
    "present→flashback→present→flashback: only change 'segment' at a real "
    "time-shift, and change it back only when the story truly returns to now.\n"
    "Each input panel has an integer index `n` (0-based, in reading order). For "
    "each span return: from_index and to_index (the INCLUSIVE `n` range it "
    "covers — a span is always a CONSECUTIVE run, so from_index <= to_index, "
    "and the next span's from_index must be greater than this span's to_index: "
    "spans stay in ascending order and never overlap), segment (present | "
    "flashback | dream — MARK flashbacks and dreams), arc_label (a 2-4 word "
    "label for the scene). Cover EVERY panel exactly once, in order. Do not "
    "target a fixed number of spans or a fixed panel count. The downstream "
    "script/timeline renders panel-level cues; these spans are only story "
    "context and must follow the source's natural rhythm.\n"
)

SYSTEM = _SYSTEM_CORE + (
    "ALSO return 'chapter': a "
    "LOGLINE (one vivid sentence — what this chapter is about, its arc), and a "
    "PREMISE (1-2 sentences: the situation + the stakes), "
    "synthesized from the WHOLE sequence. This is the through-line the narrator "
    "uses to connect the beats — base it ONLY on what the panels actually show. "
    "Do not fuse separate facts into a new cause, origin, inheritance, identity, "
    "or timeline. If the chapter separately mentions a bloodline and later shows "
    "a power/technology, do not claim the power comes from that bloodline unless "
    "the panels explicitly say so. Prefer a simple accurate sequence over a clever "
    "but unsupported causal connection. "
    "chapter.logline and chapter.premise are REQUIRED and MUST NEVER be blank."
)

# Chunk-call variant (pairs with BEATS_ONLY_SCHEMA): tells the model the truth
# about what it is looking at, and asks for nothing it cannot honestly give.
SYSTEM_CHUNK = _SYSTEM_CORE + (
    "You are seeing a CONTIGUOUS SLICE of the chapter, not the whole chapter: "
    "output beats only. Do NOT write any chapter summary, logline or premise — "
    "the chapter spine is synthesized separately from all slices."
)


# CHUNKED FALLBACK spine call (see fit_grouping_payload/group_chapter below): a
# chapter too large for one grouping call is split into contiguous chunks, each
# grouped independently -- no single call ever sees the whole panel sequence, so
# none of them can honestly synthesize a WHOLE-CHAPTER logline/premise. Instead
# ONE extra, tiny call gets just the ordered arc_label of every beat (across all
# chunks) plus a gist of the first/last panel -- enough to write the spine
# without ever re-sending the full sequence that overflowed in the first place.
CHAPTER_SPINE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "chapter": {"type": "OBJECT", "properties": {
            "logline": {"type": "STRING", "minLength": 12},
            "premise": {"type": "STRING", "minLength": 12},
        }, "required": ["logline", "premise"]},
    },
    "required": ["chapter"],
}

SYSTEM_SPINE = (
    "You are a manhwa recap editor. This chapter was too large to group in one "
    "pass, so it was split into chunks and grouped separately. You get the "
    "ordered arc_label of every story beat across the WHOLE chapter (in "
    "reading order), plus a short description of the very first and very last "
    "panel. From this alone, write the chapter 'spine': a LOGLINE (one vivid "
    "sentence -- what this chapter is about, its arc) and a PREMISE (1-2 "
    "sentences: the situation + the stakes). Base it ONLY on the arc labels "
    "and first/last panel given -- do not invent details, and do not fuse "
    "separate facts into a new cause, origin, inheritance, identity, or "
    "timeline. chapter.logline and chapter.premise are REQUIRED and MUST "
    "NEVER be blank."
)


def _norm_segment(s: Any) -> str:
    s = str(s or "").strip().lower()
    return s if s in _SEGMENTS else "present"


def _chapter_spine_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(str(value.get("logline") or "").strip()
                and str(value.get("premise") or "").strip())


def _chapter_spine_issue(value: Any, *, parsed: Any = None, raw: Any = None) -> str:
    """Return why the spine is unsafe, or empty string when it is usable.

    When the response never parsed into a dict at all (raw non-empty), say so
    explicitly instead of the misleading "logline/premise is blank" — there was
    no chapter object to inspect because generation was almost certainly
    truncated mid-JSON by the context/output budget (ORV ep1/2: raw responses
    captured cut ~4-5KB in, mid-string). See build_grouping_payload's gist
    shrink + call_fn's shrink-retry, which target this directly. When raw
    ALSO came back empty (no text at all, not even a truncated fragment), say
    that too instead of the same misleading blank-spine text — a true empty
    response and a parsed-but-blank chapter are different failures."""
    raw_str = str(raw or "")
    if parsed is None and raw_str.strip():
        return ("response failed to parse (likely truncated mid-generation — "
                 f"context/output budget); raw length={len(raw_str)}")
    if parsed is None:
        return "model returned an empty response"
    if not _chapter_spine_complete(value):
        return "chapter logline/premise is blank"
    return ""


def _dump_story_group_raw(ep_dir: str, *, raw_response: Any, parsed_obj: Any,
                          model: str) -> str:
    """Atomically dump the grouper's LAST raw model response beside the
    manifests when spine validation rejects it, and return the path (the
    raise names it). Diagnosability fix: a grouped_failed used to leave
    NOTHING to inspect — the raw text was discarded inside call_fn. Plain
    tmp+rename (not write_manifest): this is a debug capture, not a manifest
    with declared inputs."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(ep_dir, f".story_group_raw-{ts}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"raw_response": raw_response, "parsed_obj": parsed_obj,
                   "model": model, "ts": ts},
                  f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)
    return path


# THE VERIFIED EFFECTIVE CONTEXT CEILING for gemma4:26b in this fleet.
# Ground-truthed 2026-07-07 (read-only `ollama show gemma4:26b` on both
# machines): the Air's Modelfile pins `PARAMETER num_ctx 8192` (model-level
# cap); the Mini's Modelfile has NO num_ctx parameter, so the request rules
# there — but 16384 requests are the production-verified SWA-cache-thrash
# wedge (~32min stalls, see gemini_narrative_pass's num_ctx comment), and the
# ORV ep1 grouping truncated silently at ~10.5K tokens when this path
# requested 12288+. 8192 is also exactly what per-beat narration runs at
# (STUDIO_BEATS_NUM_CTX default). Requesting above it wedges or silently
# clamps; requesting below it starves the _PROMPT_TOKEN_BUDGET-sized prompt.
_GROUP_NUM_CTX = 8192


def _normalized_group_num_ctx(value: Any) -> int:
    """Pin the grouping call to the VERIFIED effective ceiling (8192 — see
    _GROUP_NUM_CTX). The old behavior (max(12288, requested), default 16384)
    asked for context the fleet never actually grants: above the Air's
    model-level cap, into the Mini's SWA-thrash wedge zone, and past the point
    where ollama silently truncates the prompt. Any requested value — higher,
    lower, or unparseable — normalizes to the one size the payload budget
    below is calibrated against (*value* is accepted only so the CLI flag
    stays wire-compatible)."""
    del value
    return _GROUP_NUM_CTX


def _gist(text: Any, limit: int = 160) -> str:
    """Word-boundary truncation to a short GIST, ellipsis-marked when cut. A rich
    chapter's full-prose panel descriptions made build_grouping_payload's output
    large enough to starve the model's context/output budget and truncate
    generation mid-JSON (ORV ep1/2: raw responses cut ~4-5KB in). Grouping only
    needs enough prose to tell panels apart — subjects/panel_kind carry the
    concrete signal — so a gist is enough; the beats/narration writer keeps the
    untruncated description elsewhere."""
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0].rstrip()
    return (cut or s[:limit].rstrip()) + "…"


def build_grouping_payload(panels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure: the numbered, ordered description sequence the grouper reasons
    over. `description` is shrunk to a GIST (not full prose) FOR THIS PAYLOAD
    ONLY — grouping needs subjects/panel_kind/gist to tell panels apart, not
    prose; the beats writer elsewhere keeps the full description. panel_kind/
    subjects/intensity stay intact (already short + categorical).
    EMPTY action/setting/dialogue are OMITTED, not sent as "": an absent key
    reads the same to the model, and the always-emitted empty keys alone cost
    ~44 chars/panel — enough to push a ~96-panel chapter over the prompt
    budget (see fit_grouping_payload) purely on JSON scaffolding."""
    out = []
    for i, p in enumerate(panels):
        row: Dict[str, Any] = {
            "n": i,
            "scene_file": p.get("scene_file"),
            "description": _gist(p.get("description"), 160),
            "panel_kind": p.get("panel_kind") or "",
            "subjects": list(p.get("subjects") or []),
            "intensity": p.get("intensity") or "",
        }
        for key, cap in (("action", 160), ("setting", 80), ("dialogue", 160)):
            val = (p.get(key) or "")[:cap]
            if val:
                row[key] = val
        out.append(row)
    return {"panels": out}


def _shrink_payload(payload: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Defensive-retry helper: re-gist an already-built payload's descriptions to
    a HARDER cap. Used only when a response failed to parse at all (near-
    certainly truncated mid-generation) — a smaller prompt leaves the model more
    context/output headroom to finish valid JSON on the next attempt."""
    return {**payload, "panels": [
        {**p, "description": _gist(p.get("description"), limit)}
        for p in (payload.get("panels") or [])]}


# BUDGET-AWARE PAYLOAD (context/output overflow root cause): a rich chapter's
# 160-char-gisted payload can STILL overflow the model's effective context --
# unlike an explicit context-exceed error (see _bumped_num_ctx in
# gemini_narrative_pass.py), an oversized whole-chapter prompt gets silently
# truncated by ollama, so the model sees mangled/cut-off input and echoes a
# bare '{"' forever, on every retry including the 80-char shrink-retry (ORV
# Episode 1, ~96 panels, verified in production). The fix has to happen BEFORE
# the call: measure the serialized prompt (payload + SYSTEM + schema) and
# shrink it to fit a budget with real headroom under the smallest context this
# ever runs at, rather than reacting after truncation already happened.
#
# ~5500 tokens (the original target) turned out to be too tight once measured
# against this schema's fixed per-panel shape: scene_file + keys + panel_kind
# + intensity alone cost ~55-60 chars/panel even with description/subjects
# EMPTY, which already eats most of a 5500-token budget by ~96 panels, leaving
# no room for any real description. Recalibrated to 7000 (chars/3.5 estimate)
# so a ~96-panel chapter (the verified ORV ep1 scale) stays a single call with
# real content, while still forcing a 200+-panel chapter (the backlog scale)
# into the chunked fallback below.
#
# 7000 is sized against the VERIFIED 8192 ceiling (_GROUP_NUM_CTX — Air
# Modelfile `num_ctx 8192`, Mini wedges/truncates above it; checked 2026-07-07
# via `ollama show gemma4:26b` on both machines): 8192 minus ~1000 tokens of
# output headroom (a realistic range-beats response measured ~550 tokens),
# with the remainder absorbing the chars/3.5 estimate error. Future
# calibration signal: ollama returns the ACTUAL prompt size as
# `prompt_eval_count` (already surfaced as usage["input"] by
# _call_model_with_backoff) — compare it against _prompt_token_estimate on
# real chapters before ever moving this number; do not guess.
_CHARS_PER_TOKEN = 3.5
_PROMPT_TOKEN_BUDGET = 7000
# Shrink ladder, in order: 160 (default/full) -> 100 -> 60 -> (drop subjects).
_GIST_SHRINK_STEPS = (160, 100, 60)
# SYSTEM + GROUP_SCHEMA are sent on EVERY whole-chapter grouping call
# regardless of payload size -- measured once so fit_grouping_payload can
# budget the PAYLOAD's own share of _PROMPT_TOKEN_BUDGET. Chunk calls send the
# strictly SMALLER SYSTEM_CHUNK + BEATS_ONLY_SCHEMA pair, so budgeting against
# the larger pair stays conservative for them.
_FIXED_OVERHEAD_CHARS = len(SYSTEM) + len(json.dumps(GROUP_SCHEMA))


def _prompt_token_estimate(payload: Dict[str, Any]) -> int:
    """Cheap chars/3.5 estimate of the WHOLE prompt (payload + SYSTEM + schema)
    this payload would produce -- not a real tokenizer, just enough signal to
    size the payload before calling, so we shrink/chunk BEFORE ollama silently
    truncates an oversized prompt."""
    payload_chars = len(json.dumps(payload, ensure_ascii=False))
    return int((payload_chars + _FIXED_OVERHEAD_CHARS) / _CHARS_PER_TOKEN)


def fit_grouping_payload(panels: List[Dict[str, Any]]
                         ) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the grouping payload for *panels*, shrinking it until the
    estimated whole prompt (payload+SYSTEM+schema) fits _PROMPT_TOKEN_BUDGET.
    Scales, in order: gist 160->100->60 chars, then drops `subjects` (keeping
    panel_kind+intensity, the categorical signal `effect_only_files` etc. rely
    on). A small/normal chapter never pays this cost -- shrinking only engages
    once the estimate actually exceeds budget. Returns (payload, meta); meta
    feeds the required `[story_group] payload ...` log line and always
    reflects what is ACTUALLY in `payload` (even when the budget is never
    reached -- see group_chapter/`_group_chunked`)."""
    payload = build_grouping_payload(panels)          # gist=160 (default/full)
    gist_used = _GIST_SHRINK_STEPS[0]
    tok = _prompt_token_estimate(payload)
    for step in _GIST_SHRINK_STEPS[1:]:
        if tok <= _PROMPT_TOKEN_BUDGET:
            break
        payload = _shrink_payload(payload, step)
        gist_used = step
        tok = _prompt_token_estimate(payload)
    subjects_on = True
    if tok > _PROMPT_TOKEN_BUDGET:
        subjects_on = False
        payload = {**payload, "panels": [
            {**p, "subjects": []} for p in payload["panels"]]}
        tok = _prompt_token_estimate(payload)
    meta = {
        "panels": len(panels),
        "bytes": len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        "tokens": tok,
        "gist": gist_used,
        "subjects": subjects_on,
    }
    return payload, meta


def log_grouping_payload(meta: Dict[str, Any]) -> None:
    """The one required log line, emitted for every real grouping call (the
    single whole-chapter call, or once per leaf chunk)."""
    print(f"[story_group] payload {meta['panels']}p {meta['bytes']}B "
          f"(~{meta['tokens']}tok) gist={meta['gist']} "
          f"subjects={'on' if meta['subjects'] else 'off'}")


_SPINE_GIST_LEN = 160


def _spine_payload(shots: List[Dict[str, Any]], panels: List[Dict[str, Any]]
                   ) -> Dict[str, Any]:
    """Small payload for the CHUNKED path's one extra spine call. Cheap by
    construction: it NEVER re-sends the panel sequence (that is exactly what
    overflowed the budget) -- the ordered arc_label of every beat produced by
    the chunks, plus a gist of the very first and very last panel, is enough
    context to write a whole-chapter logline/premise."""
    return {
        "arc_labels": [s.get("arc_label") or "" for s in shots],
        "first_panel": _gist(panels[0].get("description"), _SPINE_GIST_LEN)
                       if panels else "",
        "last_panel": _gist(panels[-1].get("description"), _SPINE_GIST_LEN)
                      if panels else "",
    }


def repair_to_shots(scene_order: List[str], model_beats: List[Dict[str, Any]],
                    *, max_beat_len: int = 0) -> List[Dict[str, Any]]:
    """Pure + robust: reconstruct a CONSECUTIVE partition of scene_order from the
    model's grouping intent — guarantees coverage (every panel in exactly one
    shot, in order) no matter how the model mis-orders/omits. A new shot starts
    when the model's beat changes. ``max_beat_len`` is an explicit safety
    override only; <=0 means no forced split. Unassigned panels continue the
    current beat (never dropped)."""
    try:
        limit = int(max_beat_len or 0)
    except (TypeError, ValueError):
        limit = 0
    assign: Dict[str, tuple] = {}
    for bi, b in enumerate(model_beats or []):
        seg, arc = _norm_segment(b.get("segment")), str(b.get("arc_label") or "").strip()
        for sf in (b.get("scene_files") or []):
            assign.setdefault(str(sf), (bi, seg, arc))

    shots: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for sf in scene_order:
        info = assign.get(sf)
        if info is not None:
            bi, seg, arc = info
        elif cur is not None:                       # unassigned → continue beat
            bi, seg, arc = cur["_bi"], cur["segment"], cur["arc_label"]
        else:
            bi, seg, arc = -1, "present", ""
        if (cur is None or bi != cur["_bi"]
                or (limit > 0 and len(cur["scene_files"]) >= limit)):
            cur = {"_bi": bi, "scene_files": [], "segment": seg, "arc_label": arc}
            shots.append(cur)
        cur["scene_files"].append(sf)

    return [{"shot_id": i, "scene_files": s["scene_files"],
             "segment": s["segment"], "arc_label": s["arc_label"]}
            for i, s in enumerate(shots, 1)]


def expand_index_ranges(beats: Any, scene_order: List[str]
                        ) -> tuple[List[Dict[str, Any]], str]:
    """Validate + expand the model's GROUP_SCHEMA {from_index,to_index} beats
    into the classic {scene_files, segment, arc_label, why} shape
    repair_to_shots has always consumed. `scene_order` MUST be the same
    ordered scene_file list build_grouping_payload numbered as `n` for this
    call, so from_index/to_index are positions into it.

    Returns (expanded_beats, issue): issue is "" when every range is a
    well-formed, in-bounds, ascending, non-overlapping integer pair, and
    expanded_beats then holds one beat dict per input beat, in order.
    When issue is non-empty (naming the first bad beat, for the REPAIR
    re-ask), expanded_beats is [] — caller must not use a partial expansion.

    Ranges need NOT cover every panel — that is by design, not a defect:
    gaps are the model's prerogative exactly like a gap in the old
    scene_files lists always was, and repair_to_shots's coverage invariant
    folds any skipped panel into the beat before it, so nothing is ever
    lost. What MUST be rejected before repair_to_shots ever sees it is a
    range that would silently mis-partition the story: an out-of-bounds or
    inverted pair, or one that overlaps/precedes the previous beat
    (repair_to_shots's first-write-wins would quietly pick a winner instead
    of surfacing the model's mistake)."""
    n = len(scene_order)
    if not isinstance(beats, list) or not beats:
        return [], "beats is missing or empty"
    expanded: List[Dict[str, Any]] = []
    prev_to = -1
    for i, b in enumerate(beats):
        if not isinstance(b, dict):
            return [], f"beat {i} is not an object"
        fi, ti = b.get("from_index"), b.get("to_index")
        if not isinstance(fi, int) or isinstance(fi, bool):
            return [], f"beat {i} from_index ({fi!r}) is not an integer"
        if not isinstance(ti, int) or isinstance(ti, bool):
            return [], f"beat {i} to_index ({ti!r}) is not an integer"
        if fi > ti:
            return [], f"beat {i} from_index {fi} > to_index {ti}"
        if fi < 0 or ti >= n:
            return [], f"beat {i} range [{fi},{ti}] out of bounds for {n} panels"
        if fi <= prev_to:
            return [], (f"beat {i} range [{fi},{ti}] overlaps or precedes the "
                        f"previous beat (ended at index {prev_to})")
        prev_to = ti
        expanded.append({
            "scene_files": scene_order[fi:ti + 1],
            "segment": b.get("segment"),
            "arc_label": b.get("arc_label"),
            "why": b.get("why"),
        })
    return expanded, ""


def _shots_and_chapter_from_parsed(panels: List[Dict[str, Any]], parsed: Any,
                                   max_beat_len: int
                                   ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Shared tail of group_panels/_group_chunked: turn a call_fn response
    (already-expanded {scene_files,...} beats on success) into (shots,
    chapter) via the coverage-guaranteeing repair_to_shots."""
    pd = parsed if isinstance(parsed, dict) else {}
    shots = repair_to_shots([p.get("scene_file") for p in panels],
                            pd.get("beats") or [], max_beat_len=max_beat_len)
    chapter = pd.get("chapter") if isinstance(pd.get("chapter"), dict) else {}
    return shots, chapter


def group_panels(panels: List[Dict[str, Any]], call_fn: Callable[..., Any],
                 *, max_beat_len: int = 0
                 ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group the (story-only, ordered) panels into story context spans and capture the
    chapter spine (logline/premise). `call_fn(payload) -> parsed dict|None` is
    injected (real model, or stub in tests). Returns (shots, chapter). Single
    whole-chapter call ONLY -- see group_chapter for the budget-aware entry
    point that falls back to chunking when panels won't fit one call."""
    if not panels:
        return [], {}
    parsed = call_fn(build_grouping_payload(panels))
    return _shots_and_chapter_from_parsed(panels, parsed, max_beat_len)


def _validated_beats_or_raise(parsed: Any, raw: Any, scene_order: List[str],
                              out_path: str, model: str, *, require_spine: bool,
                              check_ranges: bool = True, prompt_desc: str = ""
                              ) -> List[Dict[str, Any]]:
    """Shared post-call gate for BOTH the whole-chapter call and each chunk of
    the chunked fallback: call_fn already retried once internally (shrink or
    REPAIR) -- if the result is STILL an unsafe spine or has invalid/unexpanded
    beat ranges, dump the raw response beside the manifests and fail loudly.
    Without this, repair_to_shots would silently collapse the panels into one
    degenerate shot (beats with no 'scene_files' key look "unassigned" to it)
    and write that out instead. `require_spine=False` skips the spine check
    for a chunk-local call, which never carries a meaningful whole-chapter
    spine; `check_ranges=False` skips the range check for the chunked path's
    final spine-only call, whose response never carries `beats` at all.
    Returns the EXPANDED beats ({scene_files,...} shape) when safe -- []
    when check_ranges=False, since that caller has no use for them."""
    spine_issue = ""
    if require_spine:
        chapter = (parsed or {}).get("chapter") if isinstance(parsed, dict) else None
        spine_issue = _chapter_spine_issue(chapter, parsed=parsed, raw=raw)
    range_issue = ""
    expanded: List[Dict[str, Any]] = []
    if check_ranges and not spine_issue:
        if isinstance(parsed, dict):
            expanded, range_issue = expand_index_ranges(parsed.get("beats"), scene_order)
        elif parsed is None:
            # a chunk call that never parsed AT ALL must fail loudly too --
            # falling through with expanded=[] would let repair_to_shots
            # silently collapse the whole chunk into one degenerate shot.
            # Reuse the truthful truncated-vs-empty wording.
            range_issue = _chapter_spine_issue(None, parsed=None, raw=raw)
        else:
            range_issue = f"response is not a JSON object ({type(parsed).__name__})"
    if not spine_issue and not range_issue:
        return expanded
    dump = _dump_story_group_raw(
        os.path.dirname(os.path.abspath(out_path)),
        raw_response=raw, parsed_obj=parsed, model=model)
    sent = f"; sent prompt {prompt_desc}" if prompt_desc else ""
    if spine_issue:
        raise SystemExit(
            "Grouping model returned an unsafe chapter story spine: "
            + spine_issue + sent + f"; raw response captured at {dump}"
            "; refusing to continue")
    raise SystemExit(
        "Grouping model returned unsafe beat index ranges: "
        + range_issue + sent + f"; raw response captured at {dump}"
        "; refusing to continue")


def _merge_seam(left: List[Dict[str, Any]], right: List[Dict[str, Any]],
                *, max_beat_len: int = 0) -> List[Dict[str, Any]]:
    """Join two adjacent chunks' already-repaired shot lists in reading order.
    When the chunk split fell mid-beat (the last shot of the left chunk and
    the first shot of the right chunk share the same non-blank arc_label AND
    segment -- each chunk's own grouping call independently described the SAME
    ongoing beat), fold them into one shot instead of leaving a spurious cut
    where the story never actually breaks. This is deliberately narrow: only
    the exact chunk seam is ever considered, never general same-arc_label
    merging across the whole chapter. When *max_beat_len* is set, a fold whose
    COMBINED panel count would exceed it is skipped -- each chunk's own
    repair_to_shots already enforced the cap, and the merge runs AFTER repair,
    so nothing downstream would ever re-split an over-cap seam beat (the exact
    overflow the cap exists to prevent: gemma parse-fails on giant beats)."""
    if not left:
        return list(right)
    if not right:
        return list(left)
    a, b = left[-1], right[0]
    arc_a = str(a.get("arc_label") or "").strip().lower()
    arc_b = str(b.get("arc_label") or "").strip().lower()
    if arc_a and arc_a == arc_b and a.get("segment") == b.get("segment"):
        merged_files = list(a["scene_files"]) + list(b["scene_files"])
        if max_beat_len and len(merged_files) > max_beat_len:
            print(f"[story_group] seam-merge skipped: folded beat would span "
                  f"{len(merged_files)} panels > max_beat_len={max_beat_len} "
                  f"(arc_label={a.get('arc_label')!r})")
            return left + right
        print(f"[story_group] seam-merge: chunk boundary folded "
              f"(arc_label={a.get('arc_label')!r}, segment={a.get('segment')!r})")
        merged = {**a, "scene_files": merged_files}
        return left[:-1] + [merged] + right[1:]
    return left + right


def _group_chunked(panels: List[Dict[str, Any]], chunk_call_fn: Callable[..., Any],
                   max_beat_len: int, *, out_path: str, model: str
                   ) -> List[Dict[str, Any]]:
    """CHUNKED GROUPING FALLBACK (scale answer for 200+-panel chapters): keeps
    halving the ordered panel list into contiguous, order-preserving pieces
    until each slice's OWN fit_grouping_payload estimate is back under budget,
    groups that slice with one call (indices LOCAL to the slice -- call_fn's
    existing expand_index_ranges already re-bases them onto the slice's own
    scene_file order, so no separate re-basing step is needed), then merges
    sibling slices back together in order, folding a same-arc_label seam (see
    _merge_seam). `chunk_call_fn(payload) -> (parsed, raw)`: raw travels WITH
    parsed in the return value (not via the shared last_call side-channel a
    sibling chunk's call would otherwise overwrite before a failing chunk's
    evidence gets dumped)."""
    payload, meta = fit_grouping_payload(panels)
    if meta["tokens"] <= _PROMPT_TOKEN_BUDGET or len(panels) <= 1:
        log_grouping_payload(meta)
        scene_order = [p.get("scene_file") for p in panels]
        parsed, raw = chunk_call_fn(payload)
        expanded_beats = _validated_beats_or_raise(
            parsed, raw, scene_order, out_path, model, require_spine=False,
            prompt_desc=f"{meta['bytes']}B (~{meta['tokens']}tok)")
        return repair_to_shots(scene_order, expanded_beats, max_beat_len=max_beat_len)
    mid = len(panels) // 2
    left = _group_chunked(panels[:mid], chunk_call_fn, max_beat_len,
                          out_path=out_path, model=model)
    right = _group_chunked(panels[mid:], chunk_call_fn, max_beat_len,
                           out_path=out_path, model=model)
    return _merge_seam(left, right, max_beat_len=max_beat_len)


def group_chapter(panels: List[Dict[str, Any]], call_fn: Callable[..., Any],
                  chunk_call_fn: Callable[..., Any],
                  spine_call_fn: Optional[Callable[..., Any]] = None, *,
                  max_beat_len: int = 0, out_path: str = "", model: str = ""
                  ) -> tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    """Budget-aware entry point (main() calls this instead of group_panels
    directly): fit_grouping_payload decides whether the whole chapter still
    fits one call. When it does not, even at maximum shrink, falls back to
    _group_chunked plus one extra small spine call (see _spine_payload).
    Returns (shots, chapter, chunked) -- `chunked` tells the caller that the
    post-call range re-validation (which assumes a single whole-chapter beats
    response) does not apply, since each chunk already validated its own."""
    if not panels:
        return [], {}, False
    payload, meta = fit_grouping_payload(panels)
    if meta["tokens"] <= _PROMPT_TOKEN_BUDGET:
        log_grouping_payload(meta)
        # send the FITTED payload (possibly shrunk to make budget) — calling
        # group_panels here would rebuild it at full gist and undo the fit.
        parsed = call_fn(payload)
        shots, chapter = _shots_and_chapter_from_parsed(panels, parsed, max_beat_len)
        return shots, chapter, False
    print(f"[story_group] payload over budget even at minimum shrink "
          f"(~{meta['tokens']}tok > {_PROMPT_TOKEN_BUDGET}); chunked fallback "
          f"engaged for {len(panels)} panels")
    shots = _group_chunked(panels, chunk_call_fn, max_beat_len,
                           out_path=out_path, model=model)
    # each chunk's repair_to_shots numbered its own shots from 1 — renumber the
    # merged chapter so shot_id stays contiguous and unique across chunks.
    for i, s in enumerate(shots, 1):
        s["shot_id"] = i
    chapter: Dict[str, Any] = {}
    if spine_call_fn is not None:
        parsed = spine_call_fn(_spine_payload(shots, panels))
        pd = parsed if isinstance(parsed, dict) else {}
        chapter = pd.get("chapter") if isinstance(pd.get("chapter"), dict) else {}
    return shots, chapter, True


def nonstory_files(panels: List[Dict[str, Any]]) -> set:
    """scene_files the UNDERSTANDING (Pass 1) marked non-story — panel_kind
    'chrome'/'empty', or a parse failure. The multimodal pass already SAW these
    aren't story (a logo, an end-card, a blank/empty-bubble frame), so we trust
    it and drop them here instead of re-deriving chrome from brittle OCR regex.
    These never become beats."""
    out = set()
    for p in panels:
        sf = p.get("scene_file")
        if not sf:
            continue
        kind = str(p.get("panel_kind") or "").strip().lower()
        if kind in ("chrome", "empty") or p.get("error"):
            out.add(sf)
    return out


# Words that signal a panel is ONLY a visual effect (no actor, object, or place).
# Real combat panels are FULL of these too — so an effect word ALONE never drops a
# panel; it must ALSO name nothing concrete (below). Substring match is fine here.
_EFFECT_CUES = (
    "fragment", "sliver", "streak", "blob", "smear", "blur", "speed line",
    "speed-line", "motion line", "glow", "flash", "spark", "ember", "gradient",
    "texture", "static", "swirl", "haze", "shape", "light", "energy", "abstract",
    "splatter", "distort", "glitch", "ripple", "shimmer", "beam",
)
# Concrete nouns: a person, body part, creature, object, or place. If the
# description names ANY of these (as a WHOLE word — 'background' must not match
# 'ground'), the panel depicts a real scene and is KEPT. This is the discriminator
# that stops a real character/establishing shot from being dropped as an effect.
_CONCRETE_NOUNS = frozenset((
    "man", "woman", "boy", "girl", "child", "baby", "person", "people", "figure",
    "character", "guy", "men", "women", "crowd", "soldier", "warrior", "king",
    "queen", "men", "kid", "stranger", "individual", "individuals",
    "face", "eye", "eyes", "hand", "hands", "arm", "arms", "leg", "legs", "body",
    "head", "hair", "finger", "fingers", "mouth", "back", "shoulder", "fist",
    "foot", "feet", "teeth", "skin",
    "beast", "monster", "creature", "animal", "dog", "wolf", "snake", "dragon",
    "horn", "horns", "claw", "claws", "wing", "wings", "tail", "fang", "foliage",
    "sword", "blade", "knife", "weapon", "gun", "phone", "smartphone", "book",
    "screen", "door", "car", "table", "chair", "machine", "machinery", "structure",
    "structures", "building", "buildings", "wall", "window", "armor", "coat",
    "suit", "mask", "sign", "card", "letter", "bottle", "cup", "device", "robe",
    "portal", "gate", "rift", "throne", "banner", "flag", "vehicle", "ship",
    "city", "street", "road", "room", "hall", "house", "forest", "mountain",
    "sky", "field", "sea", "ocean", "river", "bridge", "train", "subway", "school",
    "classroom", "office", "castle", "village", "town", "landscape", "horizon",
    "ground", "floor", "ceiling", "tree", "trees", "cliff", "cave", "desk", "desks",
    "tower", "gate", "garden", "alley", "rooftop", "platform", "hallway", "stairs",
    # establishing / atmosphere / aftermath nouns: a wide or scenery panel naming
    # any of these is a REAL scene (battlefield, ruins, a smoke-covered field under
    # a dim glow), not a pure-effect sliver — so it must KEEP. Additive only: a pure
    # flash/spark/energy-beam panel names NONE of these, so it still drops.
    "silhouette", "silhouettes", "battlefield", "wreckage", "ruins", "ruin",
    "rubble", "debris", "corpse", "corpses", "skull", "skeleton", "bones",
    "statue", "smoke", "fire", "flame", "flames", "explosion", "blood", "fog",
    "mist", "cloud", "clouds", "dust", "ash", "crater", "snow", "rain", "water",
    "grass", "rock", "rocks", "stone", "barrier", "tent", "skyline", "cityscape",
    "sun", "moon", "star", "stars", "grave",
))


def effect_only_files(panels: List[Dict[str, Any]]) -> set:
    """scene_files the understanding shows as a PURE visual EFFECT — a story-kind
    panel that names NO concrete subject (only shapes / light / streaks /
    fragments) and carries no dialogue. These are transition/impact slivers, not a
    scene: the narrator must not describe them and the montage must not show them.

    The 'names nothing concrete' test is the discriminator. gemma's `subjects`
    field is unreliable (it leaves it empty on clear character close-ups), and a
    real combat panel is full of effect words (sparks, flash, embers) — so neither
    subjects==[] NOR an effect word can drop a panel alone. We drop ONLY when the
    panel is story-kind, has no listed subject, no dialogue, AND its description
    names nothing real while carrying an effect cue. Calibrated on live gemma
    output across ORV/Nano/IE: drops the ORV glowing-red sliver (p000008) and
    keeps every man/face/beast/phone/machinery panel."""
    out = set()
    for p in panels:
        sf = p.get("scene_file")
        if not sf or p.get("error"):
            continue
        if str(p.get("panel_kind") or "").strip().lower() != "story":
            continue                                  # only reclassify 'story'
        if p.get("subjects"):
            continue                                  # model named a subject
        if str(p.get("dialogue") or "").strip():
            continue                                  # carries story text (card)
        desc = str(p.get("description") or "").strip().lower()
        if not desc:
            out.add(sf)                               # story-kind, nothing at all
            continue
        words = set(re.findall(r"[a-z]+", desc))
        if words & _CONCRETE_NOUNS:
            continue                                  # names something real -> KEEP
        if any(cue in desc for cue in _EFFECT_CUES):
            out.add(sf)                               # only effects, nothing real
    return out


def caption_files(panels: List[Dict[str, Any]]) -> set:
    """scene_files the understanding marked 'caption' — text-only monologue/
    transition cards (e.g. a black card 'BACK THEN, I HAD NO IDEA.'). Their WORDS
    belong in the narration, but the bare text image is not a standalone scene."""
    return {p.get("scene_file") for p in panels
            if str(p.get("panel_kind") or "").strip().lower() == "caption"
            and p.get("scene_file")}


def merge_caption_solos(shots: List[Dict[str, Any]], caption_set: set
                        ) -> List[Dict[str, Any]]:
    """A beat made of ONLY caption panels has no art to show — its bare text-on-
    plain image would be the entire shot. Fold it into the adjacent beat of the
    SAME segment so the caption's words ride that beat's narration and the bubble
    is never a standalone shown shot. PREFER the previous beat (a caption usually
    closes the moment before it); a LEADING caption with no same-segment beat
    before it folds FORWARD into the next same-segment beat (an intro caption
    belongs WITH the moment it sets up — never stranded before it). A caption with
    no same-segment neighbour on either side stays as-is. Renumbers shot_id."""
    cap = set(caption_set or [])

    def all_caption(s: Dict[str, Any]) -> bool:
        return bool(s["scene_files"]) and all(f in cap for f in s["scene_files"])

    # pass 1: fold backward into the previous same-segment beat (preferred).
    back: List[Dict[str, Any]] = []
    for s in shots:
        if all_caption(s) and back and back[-1]["segment"] == s["segment"]:
            back[-1]["scene_files"].extend(s["scene_files"])     # weave into prev beat
        else:
            back.append({**s, "scene_files": list(s["scene_files"])})

    # pass 2: a caption-only beat that survived (no previous same-segment beat —
    # it leads its segment) folds FORWARD into the next same-segment beat.
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(back):
        s = back[i]
        if (all_caption(s) and i + 1 < len(back)
                and back[i + 1]["segment"] == s["segment"]):
            nxt = back[i + 1]
            nxt["scene_files"] = list(s["scene_files"]) + list(nxt["scene_files"])
            out.append(nxt)
            i += 2
        else:
            out.append(s)
            i += 1

    # pass 3 (last resort, CROSS-segment): a caption-only beat must never
    # survive standalone — it has no art, so downstream it becomes a narrated
    # timeline item with NOTHING on screen (prep_qa empty_item; ORV Ep0 g0003
    # "BACK THEN," was segment-isolated and shipped 3 empty items). Segment
    # purity yields to the on-screen invariant: fold into the previous beat,
    # else forward into the next.
    final: List[Dict[str, Any]] = []
    for j, s in enumerate(out):
        if all_caption(s):
            if final:
                final[-1]["scene_files"].extend(s["scene_files"])
                continue
            if j + 1 < len(out):
                out[j + 1]["scene_files"] = (list(s["scene_files"])
                                             + list(out[j + 1]["scene_files"]))
                continue
        final.append(s)
    for i, s in enumerate(final, 1):
        s["shot_id"] = i
    return final


_INTENSITY_RANK = {"calm": 0, "tense": 1, "intense": 2, "explosive": 3}


def annotate_intensity(shots: List[Dict[str, Any]],
                       panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tag each shot with PACE = the STRONGEST intensity among its panels — the
    narrator writes punchy/fast for intense|explosive beats and fuller/slower for
    calm|tense. Peak (not mean) so one explosive panel keeps the beat urgent."""
    rev = {v: k for k, v in _INTENSITY_RANK.items()}
    intens = {p.get("scene_file"): str(p.get("intensity") or "calm").lower()
              for p in panels}
    for s in shots:
        ranks = [_INTENSITY_RANK.get(intens.get(f, "calm"), 0)
                 for f in s["scene_files"]]
        s["intensity"] = rev[max(ranks)] if ranks else "calm"
    return shots


def _midtone(item: Dict[str, Any]) -> Optional[float]:
    from scene_chrome import needs_image_stats
    if (not needs_image_stats(str(item.get("ocr_clean") or ""))
            or not item.get("scene_path")):
        return None
    try:
        from PIL import Image
        import numpy as np
        im = np.asarray(Image.open(item["scene_path"]).convert("L"))
        return float(((im > 60) & (im < 200)).mean())
    except Exception:
        return None


def chrome_files(vision_items: List[Dict[str, Any]], series_title: str) -> set:
    from scene_chrome import is_chrome_scene
    out = set()
    for it in vision_items:
        sf = it.get("scene_file")
        if sf and is_chrome_scene(it, series_title=series_title or None,
                                  midtone_frac=_midtone(it)):
            out.add(sf)
    return out


# in-world system-card vocabulary — the signal that a flat card the LLM mislabeled
# 'chrome' is actually a kept story beat (a status/quest/skill/notification window),
# NOT a chapter-number or scanlator credits card (which carry none of this and stay
# dropped). Gated so the chrome-rescue can't re-introduce title cards into the video.
_SYSTEM_CARD_RE = re.compile(
    r"\b(system|status|quest|notification|skill|activation|level|dungeon|"
    r"guild|alert|alarm|hp|mp|exp|stat)\b", re.I)


def title_card_files(vision_items: List[Dict[str, Any]]) -> set:
    """Story title/system cards — 'SYSTEM ACTIVATION.', an age/time card, an RPG
    status window — that the QA layer treats as UNDROPPABLE story beats.
    Detected with prep_qa's `_is_title_card` flat-frame heuristic so story_group and
    prep_qa agree. A flat card the LLM mislabels 'chrome' is rescued ONLY when its
    OCR carries in-world system vocab (`_SYSTEM_CARD_RE`); chapter-number/credits
    chrome has none and stays dropped."""
    try:
        from prep_qa import _is_title_card
        from PIL import Image
        import numpy as np
    except Exception:
        return set()
    out = set()
    for it in vision_items:
        sf = it.get("scene_file")
        ocr = str(it.get("ocr_clean") or "")
        if not sf or not (1 <= len(ocr.split()) <= 10) or it.get("text_only"):
            continue
        # chrome-stamped: only rescue a genuine in-world SYSTEM card (vocab gate),
        # then bypass _is_title_card's chrome short-circuit so it isn't lost.
        ignore_chrome = False
        if str(it.get("panel_kind") or "").lower() == "chrome":
            if not _SYSTEM_CARD_RE.search(ocr):
                continue
            ignore_chrome = True
        vit = it
        if "flat_frac" not in it and it.get("scene_path"):
            try:
                g = np.asarray(Image.open(it["scene_path"]).convert("L"), dtype=float)
                vit = {**it, "flat_frac": float(((g > 235) | (g < 25)).mean())}
            except Exception:
                continue
        if _is_title_card(ocr, vit, ignore_chrome=ignore_chrome):
            out.add(sf)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--understood", required=True,
                    help="manifest.panels.understood.json (Pass 1 output)")
    ap.add_argument("--vision-manifest", required=True,
                    help="for chrome exclusion + scene paths")
    ap.add_argument("--out", required=True, help="manifest.groups.json")
    ap.add_argument("--story-out", default="",
                    help="manifest.story.json (chapter spine); default: beside --out")
    ap.add_argument("--series-title", default="", help="chrome BAN (cover/title)")
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="ollama")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="")
    ap.add_argument("--max-beat-len", type=int, default=DEFAULT_MAX_BEAT_LEN,
                    help="cap PANELS per context span so each beat stays small "
                         "enough for the model to emit valid JSON (default "
                         f"{DEFAULT_MAX_BEAT_LEN}); 0 = no cap (trust semantic "
                         "boundaries). Caps panels-per-beat only, never narration "
                         "length.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = deterministic beat boundaries + segment tags")
    ap.add_argument(
        "--num-ctx", type=int,
        default=int(os.environ.get("STUDIO_GROUP_NUM_CTX", "8192")),
        help="Ollama context for the grouping call. Normalized to the "
             "verified 8192 effective ceiling regardless (see "
             "_normalized_group_num_ctx) — the payload budget guarantees the "
             "prompt fits it; higher requests wedge/clamp, lower ones starve "
             "the budgeted prompt")
    ap.add_argument("--keep-chrome", action="store_true")
    args = ap.parse_args()

    understood = load_json(args.understood)
    vision = load_json(args.vision_manifest)
    panels = [p for p in (understood.get("panels") or []) if p.get("scene_file")]

    vmap = {it.get("scene_file"): it for it in (vision.get("items") or [])}
    if args.keep_chrome:
        excluded: set = set()
    else:
        # ROOT filter: trust the multimodal understanding (panel_kind) to drop
        # chrome/empty/parse-failed panels; keep the OCR-regex chrome detector as
        # belt-and-suspenders for anything the understanding missed.
        understood_nonstory = nonstory_files(panels)
        # PURE-EFFECT panels the model still labelled 'story' (a glowing sliver, an
        # impact streak — names nothing concrete): drop them HERE, before narration,
        # so the narrator never writes a line for a panel the montage can't show. The
        # alternative (dropping at render, AFTER the line is written) guaranteed the
        # narration↔image mismatch the user kept hitting.
        effect_only = effect_only_files(panels)
        # the understanding is AUTHORITATIVE: never let the brittle OCR-regex drop a
        # panel it classified as real story/caption content (that silently lost 2 ORV
        # story panels — the regex vetoed a 'story' verdict on garbled OCR).
        keep_by_understanding = {p.get("scene_file") for p in panels
                                 if str(p.get("panel_kind") or "").lower()
                                 in ("story", "caption", "system") and not p.get("error")}
        ocr_chrome = chrome_files(list(vmap.values()),
                                  args.series_title) - keep_by_understanding
        # NEVER drop a story title/system card (age/time/status/org card) even if the
        # LLM mislabelled a flat info-card as chrome — same detector prep_qa uses, so
        # this can never trip the 'system_card_dropped' QA error.
        cards = title_card_files(list(vmap.values()))
        excluded = (understood_nonstory | ocr_chrome | effect_only) - cards
        if understood_nonstory:
            print(f"[nonstory] understanding dropped {len(understood_nonstory)}: "
                  f"{sorted(understood_nonstory)}")
        if effect_only - cards:
            print(f"[effect] dropped {len(effect_only - cards)} pure-effect panel(s) "
                  f"(no concrete subject): {sorted(effect_only - cards)}")
        protected = cards & (understood_nonstory | ocr_chrome)
        if protected:
            print(f"[protect] kept {len(protected)} story system/title card(s): "
                  f"{sorted(protected)}")
        if ocr_chrome - cards:
            print(f"[chrome] OCR-regex added (non-story only): {sorted(ocr_chrome - cards)}")
    story = [p for p in panels if p.get("scene_file") not in excluded]

    client = None
    model = args.ollama_model
    if args.backend == "ollama":
        # _call_model reads STUDIO_BEATS_NUM_CTX. Grouping runs at the SAME
        # verified 8192 ceiling as per-beat narration (_GROUP_NUM_CTX):
        # fit_grouping_payload guarantees the prompt fits under it, so this
        # call never needs — and the fleet never grants — more.
        os.environ["STUDIO_BEATS_NUM_CTX"] = str(
            _normalized_group_num_ctx(args.num_ctx))
    if args.backend == "vertex":
        from google import genai
        if not args.project or not args.location:
            raise SystemExit("--project/--location required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)
        model = args.model

    # the LAST attempt's raw/parsed response, kept for the rejection dump —
    # without this a spine rejection discards the very evidence it needs.
    last_call: Dict[str, Any] = {"raw": None, "parsed": None}

    def call_fn(payload: Dict[str, Any], *, require_spine: bool = True):
        parsed = None
        raw = ""
        issue = ""
        range_issue = ""
        # SAME ordered scene_file list build_grouping_payload numbered as `n`
        # for this payload — from_index/to_index are positions into it. Shrink
        # retries only touch `description`, never scene_file order, so this
        # stays valid across attempts.
        scene_order = [p.get("scene_file") for p in (payload.get("panels") or [])]
        n_panels = len(scene_order)
        # A chunk call (require_spine=False) sees a SLICE: beats-only schema +
        # the chunk-variant instruction, so constrained decoding is never
        # forced to fabricate a whole-chapter spine that gets discarded. The
        # whole-chapter single-call path keeps GROUP_SCHEMA/SYSTEM unchanged.
        base_instruction = SYSTEM if require_spine else SYSTEM_CHUNK
        schema = GROUP_SCHEMA if require_spine else BEATS_ONLY_SCHEMA
        for attempt in range(2):
            instruction = base_instruction
            cur_payload = payload
            if attempt and parsed is None:
                # DEFENSIVE RETRY: the previous attempt failed to parse AT ALL
                # (near-certainly truncated mid-generation, not a bad spine) —
                # ONE harder shrink (80-char descriptions) frees context/output
                # headroom instead of repeating the same oversized prompt.
                cur_payload = _shrink_payload(payload, 80)
                print(f"[story_group] parse failure (raw len={len(raw)}); "
                      "shrink-retry engaged (descriptions capped to 80 chars)")
            elif attempt and range_issue:
                instruction += (
                    "\n\nREPAIR: the previous beats used invalid index ranges: "
                    + issue + f". Each beat needs integer from_index/to_index "
                    f"with 0 <= from_index <= to_index <= {n_panels - 1}; beats "
                    "must stay in ascending order and never overlap. Return "
                    "the complete grouping again.")
            elif attempt and require_spine:
                instruction += (
                    "\n\nREPAIR: the previous chapter story spine was invalid: "
                    + issue + ". Return the complete grouping again with a "
                    "specific, source-grounded logline and premise. Do not "
                    "use placeholders or fuse separate facts into a new cause.")
            # measure the prompt actually SENT this attempt (payload may have
            # been shrunk) so a parse failure can report it (requirement: the
            # error must carry the measured prompt size, the overflow signal).
            sent_bytes = len(json.dumps(cur_payload, ensure_ascii=False)
                             .encode("utf-8"))
            sent_tok = _prompt_token_estimate(cur_payload)
            parsed, raw, _usage = _call_model_with_backoff(
                client=client, model=model, system_instruction=instruction,
                user_payload=cur_payload, image_paths=[], response_schema=schema,
                max_output_tokens=3000, temperature=args.temperature,
                backoff_max=60.0, backend=args.backend)
            last_call["raw"], last_call["parsed"] = raw, parsed
            last_call["prompt_desc"] = f"{sent_bytes}B (~{sent_tok}tok)"
            # require_spine=False (a chunk of the CHUNKED FALLBACK): this slice
            # never carries a meaningful whole-chapter spine, so skip the check
            # entirely instead of REPAIR-reasking a chunk for something it
            # cannot honestly answer (see group_chapter's dedicated spine call).
            issue = ""
            if require_spine:
                issue = _chapter_spine_issue(
                    (parsed or {}).get("chapter") if isinstance(parsed, dict) else None,
                    parsed=parsed, raw=raw)
            expanded_beats: List[Dict[str, Any]] = []
            range_issue = ""
            if isinstance(parsed, dict) and not issue:
                # spine is safe (or not required); now validate + expand the
                # index ranges into the classic scene_files shape BEFORE
                # returning, so group_panels/repair_to_shots never need to
                # know ranges exist.
                expanded_beats, range_issue = expand_index_ranges(
                    parsed.get("beats"), scene_order)
                issue = range_issue
            if isinstance(parsed, dict) and not issue:
                return {**parsed, "beats": expanded_beats}
        return parsed

    def chunk_call_fn(payload: Dict[str, Any]) -> tuple[Any, str]:
        """_group_chunked's injected callable for one chunk: run call_fn
        (shrink/REPAIR retries happen inside it, spine not required) and hand
        back the UNEXPANDED last-attempt snapshot -- matching last_call, and
        what _validated_beats_or_raise expects -- so a chunk that still fails
        after its own retries can be gated + dumped exactly like the
        whole-chapter path. call_fn's own (possibly-expanded) return value is
        not used here; _group_chunked re-expands after the gate passes."""
        call_fn(payload, require_spine=False)
        return last_call["parsed"], last_call["raw"]

    def spine_call_fn(payload: Dict[str, Any]):
        """The CHUNKED FALLBACK's one extra small call (see _spine_payload):
        same shrink-free REPAIR-retry shape as call_fn, scoped to just the
        chapter spine (CHAPTER_SPINE_SCHEMA has no `beats` at all)."""
        parsed = None
        raw = ""
        issue = ""
        for attempt in range(2):
            instruction = SYSTEM_SPINE
            if attempt and parsed is None:
                print(f"[story_group] spine parse failure (raw len={len(raw)}); retrying")
            elif attempt:
                instruction += (
                    "\n\nREPAIR: the previous chapter spine was invalid: " + issue
                    + ". Return logline and premise again, grounded only in "
                    "the arc labels and first/last panel given.")
            parsed, raw, _usage = _call_model_with_backoff(
                client=client, model=model, system_instruction=instruction,
                user_payload=payload, image_paths=[], response_schema=CHAPTER_SPINE_SCHEMA,
                max_output_tokens=400, temperature=args.temperature,
                backoff_max=60.0, backend=args.backend)
            last_call["raw"], last_call["parsed"] = raw, parsed
            last_call["prompt_desc"] = (
                f"{len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))}B "
                f"(~{_prompt_token_estimate(payload)}tok, spine call)")
            chapter = (parsed or {}).get("chapter") if isinstance(parsed, dict) else None
            issue = _chapter_spine_issue(chapter, parsed=parsed, raw=raw)
            if isinstance(parsed, dict) and not issue:
                return parsed
        return parsed

    # The reference-channel contract has no magic group count. These spans are
    # context for narration continuity only; panel/cue rows drive the script and
    # render timeline downstream. --max-beat-len caps PANELS PER BEAT (default
    # DEFAULT_MAX_BEAT_LEN) so an over-long beat never overflows the model and
    # parse-fails; it never caps narration length. Pass 0 to disable the cap.
    mbl = int(args.max_beat_len or 0)
    # Budget-aware entry: group_chapter measures the fitted payload first and
    # only chunks (with the extra small spine call) when even the minimum
    # shrink cannot fit one whole-chapter call.
    shots, chapter, chunked = group_chapter(
        story, call_fn, chunk_call_fn, spine_call_fn,
        max_beat_len=mbl, out_path=args.out, model=model)
    logline = str((chapter or {}).get("logline") or "").strip()
    premise = str((chapter or {}).get("premise") or "").strip()
    spine_issue = _chapter_spine_issue(
        chapter, parsed=last_call["parsed"], raw=last_call["raw"])
    range_issue = ""
    if not spine_issue and not chunked and isinstance(last_call["parsed"], dict):
        # call_fn already retried a bad range once (its own REPAIR re-ask); if
        # it still gave up, last_call["parsed"]["beats"] still holds the
        # UNEXPANDED from_index/to_index beats (call_fn returns a NEW dict on
        # success, it never mutates this one), so this re-checks exactly what
        # call_fn saw on its last attempt. Without this gate, group_panels's
        # repair_to_shots would already have silently collapsed every panel
        # into one giant shot (it saw beats with no "scene_files" key and
        # treated all of them as unassigned) and we'd write that out instead
        # of failing loudly. Skipped when CHUNKED: each chunk already gated its
        # own ranges inside _group_chunked, and last_call now holds the SPINE
        # call's response, which carries no beats at all.
        _, range_issue = expand_index_ranges(
            last_call["parsed"].get("beats"),
            [p.get("scene_file") for p in story])
    if spine_issue:
        if "failed to parse" in spine_issue and last_call.get("prompt_desc"):
            # requirement: a parse failure (the silent-truncation signature)
            # must report the measured prompt size that was sent.
            spine_issue += f"; sent prompt {last_call['prompt_desc']}"
        dump = _dump_story_group_raw(
            os.path.dirname(os.path.abspath(args.out)),
            raw_response=last_call["raw"], parsed_obj=last_call["parsed"],
            model=model)
        raise SystemExit(
            "Grouping model returned an unsafe chapter story spine: "
            + spine_issue + f"; raw response captured at {dump}"
            "; refusing to continue")
    if range_issue:
        dump = _dump_story_group_raw(
            os.path.dirname(os.path.abspath(args.out)),
            raw_response=last_call["raw"], parsed_obj=last_call["parsed"],
            model=model)
        raise SystemExit(
            "Grouping model returned unsafe beat index ranges: "
            + range_issue + f"; raw response captured at {dump}"
            "; refusing to continue")
    # caption-only beats fold into their neighbour so the text rides real art
    shots = merge_caption_solos(shots, caption_files(story))
    shots = annotate_intensity(shots, panels)   # per-shot PACE = peak intensity
    out = {
        "source_understood": os.path.abspath(args.understood),
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "chrome_excluded": sorted(excluded),
        "grouping": {"method": "understanding_first_context_spans_v2",
                     "max_beat_len": mbl,
                     "forced_split": bool(mbl > 0)},
        "summary": {"num_scenes": len(story), "num_shots": len(shots),
                    "flashback_shots": sum(1 for s in shots
                                           if s["segment"] != "present")},
        "shots": shots,
    }
    write_manifest(args.out, out, inputs=(args.understood,), tool="story_group")

    # Chapter STORY SPINE (logline + premise + ordered arc) — the through-line the
    # narrator uses so beats connect into one story instead of isolated captions.
    story_out = args.story_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "manifest.story.json")
    spine = {
        "source_groups": os.path.abspath(args.out),
        "logline": logline,
        "premise": premise,
        "arc": [{"group_id": s["shot_id"], "arc_label": s["arc_label"],
                 "segment": s["segment"]} for s in shots],
    }
    write_manifest(story_out, spine, inputs=(args.understood,), tool="story_group")
    print(f"[ok] wrote={args.out} scenes={len(story)} shots={len(shots)} "
          f"(story-grouped) excluded={len(excluded)} | spine={story_out} "
          f"logline={'y' if spine['logline'] else 'n'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
