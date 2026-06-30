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
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from gemini_narrative_pass import (                                   # noqa: E402
    load_json, dump_json, _call_model_with_backoff)

_SEGMENTS = ("present", "flashback", "dream")

# Cap PANELS PER BEAT (not narration length). A single beat of 29/35 panels
# overflows the grouping/narration model (gemma4:26b) and parse-fails, which then
# collapses the whole group to one fallback shot. Keeping beats <= this many
# panels lets the model emit valid JSON per beat. Each panel still gets its own
# content-scaled narration downstream — this only bounds how many panels share
# one context span/shot. Operators can pass --max-beat-len 0 to disable.
DEFAULT_MAX_BEAT_LEN = 8

GROUP_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "chapter": {"type": "OBJECT", "properties": {
            "logline": {"type": "STRING", "minLength": 12},
            "premise": {"type": "STRING", "minLength": 12},
        }, "required": ["logline", "premise"]},
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
    "required": ["chapter", "beats"],
}

SYSTEM = (
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
    "For each span return: scene_files (its consecutive panels, in order), "
    "segment (present | flashback | dream — MARK flashbacks and dreams), "
    "arc_label (a 2-4 word label for the scene). Cover EVERY panel exactly once, "
    "in order. Do not target a fixed number of spans or a fixed panel count. The "
    "downstream script/timeline renders panel-level cues; these spans are only "
    "story context and must follow the source's natural rhythm.\n"
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


def _norm_segment(s: Any) -> str:
    s = str(s or "").strip().lower()
    return s if s in _SEGMENTS else "present"


def _chapter_spine_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(str(value.get("logline") or "").strip()
                and str(value.get("premise") or "").strip())


def _chapter_spine_issue(value: Any, payload: Dict[str, Any]) -> str:
    """Return why the spine is unsafe, or empty string when it is usable."""
    if not _chapter_spine_complete(value):
        return "chapter logline/premise is blank"
    return ""


def _normalized_group_num_ctx(value: Any) -> int:
    """The whole-chapter grouping prompt needs room for input plus JSON output."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 16384
    return max(12288, n)


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


def group_panels(panels: List[Dict[str, Any]], call_fn: Callable[..., Any],
                 *, max_beat_len: int = 0
                 ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group the (story-only, ordered) panels into story context spans and capture the
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
        default=int(os.environ.get("STUDIO_GROUP_NUM_CTX", "16384")),
        help="Ollama context for the one whole-chapter grouping call; kept "
             "separate from the 8K per-beat narration context")
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
        # _call_model reads STUDIO_BEATS_NUM_CTX. Override it only inside this
        # one-call grouping subprocess: the whole chapter payload can exceed 8K,
        # while per-beat narration intentionally stays at 8K to avoid SWA thrash.
        os.environ["STUDIO_BEATS_NUM_CTX"] = str(
            _normalized_group_num_ctx(args.num_ctx))
    if args.backend == "vertex":
        from google import genai
        if not args.project or not args.location:
            raise SystemExit("--project/--location required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)
        model = args.model

    def call_fn(payload: Dict[str, Any]):
        parsed = None
        issue = ""
        for attempt in range(2):
            instruction = SYSTEM
            if attempt:
                instruction += (
                    "\n\nREPAIR: the previous chapter story spine was invalid: "
                    + issue + ". Return the complete grouping again with a "
                    "specific, source-grounded logline and premise. Do not "
                    "use placeholders or fuse separate facts into a new cause.")
            parsed, _raw, _usage = _call_model_with_backoff(
                client=client, model=model, system_instruction=instruction,
                user_payload=payload, image_paths=[], response_schema=GROUP_SCHEMA,
                max_output_tokens=3000, temperature=args.temperature,
                backoff_max=60.0, backend=args.backend)
            issue = _chapter_spine_issue(
                (parsed or {}).get("chapter") if isinstance(parsed, dict) else None,
                payload)
            if isinstance(parsed, dict) and not issue:
                return parsed
        return parsed

    # The reference-channel contract has no magic group count. These spans are
    # context for narration continuity only; panel/cue rows drive the script and
    # render timeline downstream. --max-beat-len caps PANELS PER BEAT (default
    # DEFAULT_MAX_BEAT_LEN) so an over-long beat never overflows the model and
    # parse-fails; it never caps narration length. Pass 0 to disable the cap.
    mbl = int(args.max_beat_len or 0)
    shots, chapter = group_panels(story, call_fn, max_beat_len=mbl)
    logline = str((chapter or {}).get("logline") or "").strip()
    premise = str((chapter or {}).get("premise") or "").strip()
    spine_issue = _chapter_spine_issue(chapter, build_grouping_payload(story))
    if spine_issue:
        raise SystemExit(
            "Grouping model returned an unsafe chapter story spine: "
            + spine_issue + "; refusing to continue")
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
    dump_json(args.out, out)

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
    dump_json(story_out, spine)
    print(f"[ok] wrote={args.out} scenes={len(story)} shots={len(shots)} "
          f"(story-grouped) excluded={len(excluded)} | spine={story_out} "
          f"logline={'y' if spine['logline'] else 'n'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
