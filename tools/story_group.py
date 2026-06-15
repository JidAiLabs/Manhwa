#!/usr/bin/env python3
"""story_group.py — Pass 2 of the understanding-first pipeline.

Group panels by UNDERSTANDING, not by gutters. Reads the per-panel descriptions
(Pass 1, panel_understand) in reading order and segments them into STORY BEATS:
a beat = a run of CONSECUTIVE panels that form one moment/shot. New beat at a
scene/location change, a time jump, a flashback start/end, or a topic shift;
near-identical consecutive panels merge into one montage beat. Each beat is
tagged segment (present|flashback|dream) + arc_label — so flashbacks are native.

This REPLACES the position/threshold grouping in scene_group_builder.py. Output
is a byte-compatible manifest.groups.json: top-level `shots` with per-shot
`shot_id` (contiguous int) + `scene_files`, plus extra `segment`/`arc_label`
tags that downstream ignores safely (timeline can carry the flashback tag).

Coverage is an invariant: every non-chrome panel lands in exactly one shot
(repair_to_shots reconstructs a consecutive partition from the model's intent).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from gemini_narrative_pass import (                                   # noqa: E402
    load_json, dump_json, _call_model_with_backoff)

_SEGMENTS = ("present", "flashback", "dream")

GROUP_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "chapter": {"type": "OBJECT", "properties": {
            "logline": {"type": "STRING"},
            "premise": {"type": "STRING"},
        }},
        "beats": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "scene_files": {"type": "ARRAY", "items": {"type": "STRING"}},
                "segment": {"type": "STRING", "enum": list(_SEGMENTS)},
                "arc_label": {"type": "STRING"},
                "why": {"type": "STRING"},
            },
            "required": ["scene_files"],
        }},
    },
    "required": ["beats"],
}

SYSTEM = (
    "You are a manhwa recap editor. You get a numbered sequence of panel "
    "descriptions from ONE chapter, in reading order. Segment them into STORY "
    "BEATS for the recap.\n"
    "A beat = a run of CONSECUTIVE panels that form one moment/shot. Start a NEW "
    "beat at: a scene or location change, a time jump, a FLASHBACK start OR end, "
    "or a clear topic shift. Group near-identical consecutive panels (e.g. a "
    "multi-panel action or a slow reveal) into ONE beat.\n"
    "ORDER + FLOW — the beats must read as ONE story in reading order:\n"
    "  - A caption / monologue panel that INTRODUCES the moment right after it "
    "(e.g. 'ON THE DAY I FINISHED THE WEB NOVEL…' immediately before that event) "
    "belongs in the SAME beat as the art it introduces. Never strand an intro "
    "caption as its own separate beat sitting before the moment it sets up.\n"
    "  - Keep a flashback or dream as a CONTIGUOUS block. Do NOT bounce "
    "present→flashback→present→flashback: only change 'segment' at a real "
    "time-shift, and change it back only when the story truly returns to now.\n"
    "For each beat return: scene_files (its consecutive panels, in order), "
    "segment (present | flashback | dream — MARK flashbacks and dreams), "
    "arc_label (a 2-4 word label for the scene). Cover EVERY panel exactly once, "
    "in order. Prefer MORE, tighter beats over a few large ones — each beat "
    "becomes one narration line, so it should be one clear moment.\n"
    "ALSO return 'chapter': a LOGLINE (one vivid sentence — what this chapter is "
    "about, its arc) and a PREMISE (1-2 sentences: the situation + the stakes), "
    "synthesized from the WHOLE sequence. This is the through-line the narrator "
    "uses to connect the beats — base it ONLY on what the panels actually show."
)


def _norm_segment(s: Any) -> str:
    s = str(s or "").strip().lower()
    return s if s in _SEGMENTS else "present"


def build_grouping_payload(panels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure: the numbered, ordered description sequence the grouper reasons over."""
    return {"panels": [{
        "n": i,
        "scene_file": p.get("scene_file"),
        "description": (p.get("description") or "")[:300],
        "action": (p.get("action") or "")[:160],
        "setting": (p.get("setting") or "")[:80],
        "dialogue": (p.get("dialogue") or "")[:160],
        "intensity": p.get("intensity") or "",
    } for i, p in enumerate(panels)]}


def repair_to_shots(scene_order: List[str], model_beats: List[Dict[str, Any]],
                    *, max_beat_len: int = 4) -> List[Dict[str, Any]]:
    """Pure + robust: reconstruct a CONSECUTIVE partition of scene_order from the
    model's grouping intent — guarantees coverage (every panel in exactly one
    shot, in order) no matter how the model mis-orders/omits. A new shot starts
    when the model's beat changes OR the run hits max_beat_len. Unassigned panels
    continue the current beat (never dropped)."""
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
                or len(cur["scene_files"]) >= max_beat_len):
            cur = {"_bi": bi, "scene_files": [], "segment": seg, "arc_label": arc}
            shots.append(cur)
        cur["scene_files"].append(sf)

    return [{"shot_id": i, "scene_files": s["scene_files"],
             "segment": s["segment"], "arc_label": s["arc_label"]}
            for i, s in enumerate(shots, 1)]


def group_panels(panels: List[Dict[str, Any]], call_fn: Callable[..., Any],
                 *, max_beat_len: int = 4
                 ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group the (story-only, ordered) panels into story shots AND capture the
    chapter spine (logline/premise). `call_fn(payload) -> parsed dict|None` is
    injected (real model, or stub in tests). Returns (shots, chapter)."""
    if not panels:
        return [], {}
    parsed = call_fn(build_grouping_payload(panels))
    pd = parsed if isinstance(parsed, dict) else {}
    beats = pd.get("beats")
    chapter = pd.get("chapter") if isinstance(pd.get("chapter"), dict) else {}
    shots = repair_to_shots([p.get("scene_file") for p in panels],
                            beats or [], max_beat_len=max_beat_len)
    return shots, chapter


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
    SAME segment (prefer the previous one) so the caption's words ride that beat's
    narration and the text image gets deduped. A closing caption with no same-
    segment neighbour stays as-is. Renumbers shot_id to stay contiguous."""
    cap = set(caption_set or [])

    def all_caption(s: Dict[str, Any]) -> bool:
        return bool(s["scene_files"]) and all(f in cap for f in s["scene_files"])

    out: List[Dict[str, Any]] = []
    for s in shots:
        if all_caption(s) and out and out[-1]["segment"] == s["segment"]:
            out[-1]["scene_files"].extend(s["scene_files"])     # weave into prev beat
        else:
            out.append({**s, "scene_files": list(s["scene_files"])})
    for i, s in enumerate(out, 1):
        s["shot_id"] = i
    return out


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


def title_card_files(vision_items: List[Dict[str, Any]]) -> set:
    """Story title/system cards — 'SKY CORPORATION.', 'LIN ZICHEN - AGE: 5 MONTHS',
    an RPG status window — that the QA layer treats as UNDROPPABLE story beats.
    Detected with prep_qa's EXACT `_is_title_card` heuristic (same flat-frame test)
    so story_group and prep_qa always agree: a card kept here can never be flagged
    'system_card_dropped'. These are protected from chrome/empty exclusion even when
    the LLM mislabels a flat info-card as chrome."""
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
        vit = it
        if "flat_frac" not in it and it.get("scene_path"):
            try:
                g = np.asarray(Image.open(it["scene_path"]).convert("L"), dtype=float)
                vit = {**it, "flat_frac": float(((g > 235) | (g < 25)).mean())}
            except Exception:
                continue
        if _is_title_card(ocr, vit):
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
    ap.add_argument("--max-beat-len", type=int, default=0,
                    help="cap panels per beat; 0 = AUTO-scale to chapter size "
                         "(~16-beat target) so small chapters get tight beats and "
                         "big ones don't explode (ORV 34p->cap 2, a 116p chapter->7)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = deterministic beat boundaries + segment tags")
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
        # the understanding is AUTHORITATIVE: never let the brittle OCR-regex drop a
        # panel it classified as real story/caption content (that silently lost 2 ORV
        # story panels — the regex vetoed a 'story' verdict on garbled OCR).
        keep_by_understanding = {p.get("scene_file") for p in panels
                                 if str(p.get("panel_kind") or "").lower()
                                 in ("story", "caption") and not p.get("error")}
        ocr_chrome = chrome_files(list(vmap.values()),
                                  args.series_title) - keep_by_understanding
        # NEVER drop a story title/system card (age/time/status/org card) even if the
        # LLM mislabelled a flat info-card as chrome — same detector prep_qa uses, so
        # this can never trip the 'system_card_dropped' QA error.
        cards = title_card_files(list(vmap.values()))
        excluded = (understood_nonstory | ocr_chrome) - cards
        if understood_nonstory:
            print(f"[nonstory] understanding dropped {len(understood_nonstory)}: "
                  f"{sorted(understood_nonstory)}")
        protected = cards & (understood_nonstory | ocr_chrome)
        if protected:
            print(f"[protect] kept {len(protected)} story system/title card(s): "
                  f"{sorted(protected)}")
        if ocr_chrome - cards:
            print(f"[chrome] OCR-regex added (non-story only): {sorted(ocr_chrome - cards)}")
    story = [p for p in panels if p.get("scene_file") not in excluded]

    client = None
    model = args.ollama_model
    if args.backend == "vertex":
        from google import genai
        if not args.project or not args.location:
            raise SystemExit("--project/--location required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)
        model = args.model

    def call_fn(payload: Dict[str, Any]):
        parsed, _raw, _usage = _call_model_with_backoff(
            client=client, model=model, system_instruction=SYSTEM,
            user_payload=payload, image_paths=[], response_schema=GROUP_SCHEMA,
            max_output_tokens=2400, temperature=args.temperature,
            backoff_max=60.0, backend=args.backend)
        return parsed

    # AUTO-scale the per-beat cap to chapter size (target ~16 beats): a fixed cap
    # of 2 made ORV's 34 panels a good 13 beats but a 116-panel chapter 64
    # (over-fragmented — captions split off -> fragment_dangle, narration overload).
    mbl = args.max_beat_len or max(2, round(len(story) / 16))
    shots, chapter = group_panels(story, call_fn, max_beat_len=mbl)
    # caption-only beats fold into their neighbour so the text rides real art
    shots = merge_caption_solos(shots, caption_files(story))
    shots = annotate_intensity(shots, panels)   # per-shot PACE = peak intensity
    out = {
        "source_understood": os.path.abspath(args.understood),
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "chrome_excluded": sorted(excluded),
        "grouping": {"method": "understanding_first_v1",
                     "max_beat_len": args.max_beat_len},
        "summary": {"num_scenes": len(story), "num_shots": len(shots),
                    "flashback_shots": sum(1 for s in shots
                                           if s["segment"] != "present")},
        "shots": shots,
    }
    dump_json(args.out, out)

    # Chapter STORY SPINE (logline + premise + ordered arc) — the through-line the
    # narrator uses so beats connect into one story instead of isolated captions.
    story_out = args.story_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "manifest.story.json")
    spine = {
        "source_groups": os.path.abspath(args.out),
        "logline": str((chapter or {}).get("logline") or "").strip(),
        "premise": str((chapter or {}).get("premise") or "").strip(),
        "arc": [{"group_id": s["shot_id"], "arc_label": s["arc_label"],
                 "segment": s["segment"]} for s in shots],
    }
    dump_json(story_out, spine)
    print(f"[ok] wrote={args.out} scenes={len(story)} shots={len(shots)} "
          f"(story-grouped) excluded={len(excluded)} | spine={story_out} "
          f"logline={'y' if spine['logline'] else 'n'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
