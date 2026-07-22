#!/usr/bin/env python3
"""panel_understand.py — Pass 1 of the understanding-first pipeline.

Describe EVERY panel (multimodal): what is literally happening, who is in it,
the dialogue, the setting, the intensity. One record per panel = **full
coverage by construction** — nothing can be merged or dropped before it has been
understood. This output feeds the story-grouper (Pass 2, which segments the
sequence into story-sized beats + flashback boundaries) and the per-beat
narrator (Pass 3).

It reuses the battle-tested multimodal call from gemini_narrative_pass
(`_call_model_with_backoff`: ollama/Gemma or Vertex, schema-constrained, 429-safe).

Out: manifest.panels.understood.json = {panels:[{scene_file, description,
subjects[], action, dialogue, setting, intensity}]}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

_TD = os.path.dirname(os.path.abspath(__file__))
# repo root too: `from studio...` must work even when spawned as a bare
# script without PYTHONPATH (the worker does; pipeline._run_tool sets it) —
# same bootstrap prep_qa.py uses.
for _p in (_TD, os.path.dirname(_TD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gemini_narrative_pass import (                                   # noqa: E402
    load_json, _call_model_with_backoff, _model_safe_image)
from manifest_io import write_manifest, input_sha                     # noqa: E402

# Gemini-style schema (UPPERCASE enums) — _call_model converts it for Ollama.
PANEL_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "description": {"type": "STRING"},
        "subjects": {"type": "ARRAY", "items": {"type": "STRING"}},
        "action": {"type": "STRING"},
        "dialogue": {"type": "STRING"},
        "setting": {"type": "STRING"},
        "intensity": {"type": "STRING",
                      "enum": ["calm", "tense", "intense", "explosive"]},
        "panel_kind": {"type": "STRING",
                       "enum": ["story", "chrome", "empty", "caption", "system"]},
        # eyes wave: structured action-intensity fields. NOT in `required` so
        # every existing consumer of pu_v1-shaped records keeps parsing.
        "strikes_or_weapons": {"type": "STRING",
                               "enum": ["none", "visible", "in_use"]},
        "sfx_text": {"type": "STRING"},
        # pu_v4: the analyst's own confidence — NOT in `required`, same
        # back-compat posture as the pu_v2 fields above.
        "uncertain": {"type": "BOOLEAN"},
        # pu_v5: structured agent->target attribution. The free-text `action`
        # carries no direction ("a strike lands"), which let a visually
        # ambiguous fight panel ship an INVERTED who-struck-whom that no
        # downstream guard could check. NOT in `required` (back-compat).
        "actions": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {"actor": {"type": "STRING"},
                           "verb": {"type": "STRING"},
                           "target": {"type": "STRING"}},
            "required": ["actor", "verb", "target"]}},
    },
    "required": ["description", "action", "intensity", "panel_kind"],
}

SYSTEM = (
    "You are a manhwa recap analyst. You see ONE webtoon panel image plus its "
    "OCR text. Describe what is LITERALLY happening in this panel — specific and "
    "vivid, but strictly faithful to what is shown (never invent characters or "
    "events). Return JSON:\n"
    "  description: 1-2 concrete sentences of the action/scene in this panel.\n"
    "  subjects: the characters / creatures / key objects visible. For each\n"
    "PERSON, include their distinguishing look AS DRAWN — clothing/outfit and\n"
    "its color, hair, any mask/hood/armor or notable accessory (e.g. 'a young\n"
    "man in a light robe with a blue sash', 'a masked figure in a dark hooded\n"
    "cloak') — never just 'a man' or 'a figure'. If the SAME person appears in "
    "several sub-frames/insets of this ONE panel, list that person ONCE — never "
    "one subjects entry per sub-frame.\n"
    "  action: the single key event or beat of this panel.\n"
    "  actions: each distinct action performed IN this panel (0-3 entries) as "
    "{actor, verb, target}. actor and target MUST each copy the EXACT wording "
    "of one `subjects` entry; target is '' when the action has no target; "
    "write 'unclear' when you genuinely cannot tell WHO acts or WHO is hit. "
    "verb is a short present-tense phrase ('strikes', 'draws a blade at', "
    "'collapses'). WHO does WHAT to WHOM is the single most important fact of "
    "an action panel — if the striker or the struck is ambiguous, say "
    "'unclear', NEVER guess.\n"
    "  Evidence discipline: describe marks and stains at the certainty the art "
    "gives — an ambiguous dark or reddish stain is 'a dark stain' / 'stained', "
    "NEVER 'blood' unless a wound, dripping, or spatter makes it unambiguous; "
    "never upgrade dirt, shadow, or paint to gore, and never upgrade an "
    "ambiguous shape into a specific object or an attack.\n"
    "  uncertain: set true when you genuinely cannot identify a subject — then "
    "describe it hedged ('an unclear pale shape') and do NOT assign it an "
    "action or intent; false otherwise.\n"
    "  dialogue: any spoken line or caption, copied VERBATIM from the OCR; '' if "
    "none. Do not paraphrase dialogue.\n"
    "  setting: where/what the scene is (a train, a city street, a flashback "
    "screen, etc.).\n"
    "  intensity: calm | tense | intense | explosive. RESERVE 'intense' and "
    "'explosive' for genuine PEAKS — a real clash, a shocking reveal, mortal "
    "danger. Grade routine action, travel, dialogue, and ordinary reactions "
    "(e.g. a stumble or a fall) as 'calm' or 'tense'. Most panels are calm/tense.\n"
    "  strikes_or_weapons: 'none' (no weapon or strike in this panel), "
    "'visible' (a weapon is drawn/present but not being used), 'in_use' (a "
    "strike, stab, blow, or crash is being DELIVERED in this panel).\n"
    "  sfx_text: any painted sound-effect lettering on the art, transcribed "
    "if you can read it; '' if none.\n"
    "  panel_kind: classify this panel for the recap —\n"
    "    'chrome' = PUBLICATION/PLATFORM furniture wrapping THIS release, never the "
    "story world: this series' COVER, an EPISODE/CHAPTER-NUMBER card, the creator/site/"
    "publisher LOGO or watermark (e.g. a '…toon.com' end-card), a 'thanks for reading / "
    "subscribe / follow / join our Discord' promo, or a credits page. Chrome is the "
    "WEBSITE / APP / RELEASE that HOSTS the comic — NOT the characters or their world. "
    "A phone/screen/device a CHARACTER is using IN-STORY (their app, a novel they read, "
    "a chat, a game UI) is NOT chrome — that is the story world; classify it 'story'.\n"
    "    'empty' = NO content: a blank or near-blank frame, a plain gradient / "
    "speed-line / texture transition with no subject, or speech bubbles with NO "
    "readable text.\n"
    "    'caption' = TEXT WITHOUT A SCENE: either the story's narrative VOICE as "
    "text on a plain card (an author monologue or scene-setting / transition line, "
    "e.g. a plain card carrying a retrospective or scene-setting line in the "
    "narrator's own voice), OR a lone speech / shout / "
    "dialogue bubble (or any text) floating on a PLAIN / BLANK / WHITE / EMPTY "
    "background with NO drawn scene, character, or object behind it (e.g. 'a single "
    "white speech bubble against a plain white background'). Its words go in "
    "'dialogue'; it is not a picture. A panel with REAL ART (a character, a place, "
    "an object) AND a bubble/caption is 'story', not 'caption'.\n"
    "    'system' = an IN-WORLD GAME / SYSTEM INTERFACE the CHARACTER perceives — "
    "a QUEST window, a STATUS / STAT / SKILL screen, a NOTIFICATION / ALARM / level-up "
    "toast, or a SYSTEM MESSAGE — for instance a quest or objective window, a status or "
    "stat readout, a level-up toast, or a message announcing that something has been "
    "acquired, defeated, unlocked, or activated. "
    "It can be ANY length, ANY case, ANY color/art style, and may be drawn "
    "OVER character art. These are PLOT and MUST be kept and shown.\n"
    "    'story' = the STORY WORLD — real scene art AND in-world device screens a "
    "character uses in-story (a reader app, chat, feed), a place/organization name card. "
    "A panel with real character art is 'story' even if a system window is drawn over it. "
    "When unsure between system/story (both are always kept), pick either; only an AUTHOR "
    "narrative caption is 'caption' and only platform furniture is 'chrome'.\n"
    "The 'previous_panels' field is context for continuity only — describe THIS "
    "panel, not the previous ones. 'nearby_dialogue', when present, is what "
    "characters SAY in the surrounding panels — use it ONLY to settle who does "
    "what to whom here; never describe those other panels."
)

# Bump this whenever SYSTEM/PANEL_SCHEMA change materially — it is stamped onto
# every record and gates --resume acceptance (see understand_panels), so a
# prompt change no longer requires manually deleting understood.json.
# pu_v2: impact-SFX fusion (strikes_or_weapons + sfx_text fields, the
# detector-driven impact notice) — invalidates ALL pu_v1 records, INTENDED:
# chapters re-understand under the impact-aware prompt.
# pu_v3: subjects must carry each person's distinguishing LOOK (outfit color,
# hair, mask/hood, accessories) — the raw material cast_identity.py resolves
# panel figures from deterministically (round-2 identity misattribution fix).
# Invalidates ALL pu_v2 records, INTENDED: chapters re-understand so figure
# resolution has appearance evidence to match against manifest.cast.json.
# pu_v4: evidence discipline (ambiguous stain != blood; never upgrade an
# ambiguous shape to an object/attack), same-person-across-sub-frames listed
# ONCE, and the `uncertain` flag (+ forced-choice re-ask) so the writer and
# the grounding judge know when the analyst itself hedged. Invalidates ALL
# pu_v3 records, INTENDED (2026-07-16 root-cause wave: dirt->blood,
# limb-or-object->attack, 3-sub-shots->3 people).
# pu_v5: structured `actions` = [{actor, verb, target}] (subjects-verbatim or
# 'unclear', never a guess) + forced-choice re-ask on an in_use strike with an
# unclear actor/target. Root-cause for the nano ch1 inverted kill: the
# free-text `action` carried no direction, so a wrong who-struck-whom guess
# was unstructured and uncheckable. Invalidates ALL pu_v4 records, INTENDED
# (2026-07-20 story-state wave).
# pu_v6: DIRECTION VERIFICATION. pu_v5 shipped and the kill was STILL
# inverted — because the model never hedged; it stated the wrong direction
# confidently, so the 'unclear' re-ask never fired. Every two-person in_use
# strike now gets one image re-ask with the NEIGHBOURING PANELS' DIALOGUE
# attached (the evidence that actually settles it, already on disk).
# Invalidates ALL pu_v5 records, INTENDED.
# pu_v7: series-derived examples stripped from SYSTEM — the system-card and
# caption illustrations quoted lines from specific works, which both biases
# the classifier toward one series' vocabulary and puts source text in our
# prompt. Replaced with descriptions of the pattern. Examples steer
# classification, so this is a material prompt change.
PROMPT_VERSION = "pu_v7"

# --- extreme-tall strips: windowed understanding -----------------------------
# A cover/credits strip (ORV Ep0: 800x7540) downscaled to model resolution is
# unreadable — gemma described the top art, never saw the series title, said
# 'story', and the title shipped on screen. Strips past these gates are
# understood in ~1600px windows instead. Records carry a DISTINCT version
# suffix so only tall strips re-run on upgrade — bumping PROMPT_VERSION itself
# would invalidate every cached panel fleet-wide.
TALL_WINDOWS_VERSION = PROMPT_VERSION + "+tw2"   # tw2: rescue respects windows
_TALL_MIN_H_PX = 4000
_TALL_MIN_RATIO = 6.0
_TALL_WIN_PX = 1600
_TALL_WIN_OVERLAP = 200

# _model_safe_image (downscale an over-tall panel so the vision encoder can't
# OOM) lives in gemini_narrative_pass — the SHARED model-call module — so it
# covers BOTH the understanding call_fn wrapper below AND the beats pass. The
# wrapper here is belt-and-suspenders (its own re-ask sites) + owns temp cleanup;
# gemini_narrative_pass._call_model also downscales, so a temp it gets is a no-op.


def _tall_dims(scene_path: Optional[str]) -> Optional[Tuple[int, int]]:
    """(w, h) when the scene is an extreme-tall strip gemma cannot read at
    model scale (cover/credits blocks, chunk-as-panel leftovers); else None.
    Real tall art (vertical falls, ~3-4k px, ratio < 6) stays single-pass."""
    if not scene_path or not os.path.exists(scene_path):
        return None
    try:
        from PIL import Image
        with Image.open(scene_path) as im:
            w, h = im.size
    except Exception:
        return None
    if h >= _TALL_MIN_H_PX and h >= _TALL_MIN_RATIO * max(1, w):
        return (int(w), int(h))
    return None


def _expected_version(it: Dict[str, Any]) -> str:
    return (TALL_WINDOWS_VERSION if _tall_dims(it.get("scene_path"))
            else PROMPT_VERSION)


def understand_tall_strip(
    it: Dict[str, Any],
    ctx: List[str],
    call_fn: Callable[..., Any],
    dims: Tuple[int, int],
    *,
    impact_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Describe an extreme-tall strip window-by-window; merge to ONE record.

    ANY chrome window makes the WHOLE strip publication chrome: covers carry
    the series title ON the art, so keeping the "art part" would still show
    the cover. A strip with zero chrome windows stays one pannable panel with
    the window fields merged. Returns (parsed_or_None, windows_meta) —
    parsed=None when every window failed to parse (resume re-runs it).
    """
    # ponytail: whole-strip verdict; per-window keep/crop only if a mixed
    # story+credits strip ever shows up in QA.
    import tempfile
    from PIL import Image
    w, h = dims
    spans: List[Tuple[int, int]] = []
    y = 0
    while y < h:
        y1 = min(h, y + _TALL_WIN_PX)
        spans.append((y, y1))
        if y1 >= h:
            break
        y += _TALL_WIN_PX - _TALL_WIN_OVERLAP

    parsed_windows: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []
    with Image.open(it.get("scene_path")) as im:
        rgb = im.convert("RGB")
        for wy0, wy1 in spans:
            regs = [r for r in (impact_regions or [])
                    if (r.get("bbox") or [0, 0, 0, 0])[1] < wy1
                    and ((r.get("bbox") or [0, 0, 0, 0])[1]
                         + (r.get("bbox") or [0, 0, 0, 0])[3]) > wy0]
            win = rgb.crop((0, wy0, w, wy1))
            tmp = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg",
                                                 delete=False) as tf:
                    tmp = tf.name
                win.save(tmp, "JPEG", quality=90)
                parsed = call_fn(build_payload(it, ctx, impact_regions=regs),
                                 tmp)
            finally:
                if tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            pw = parsed if isinstance(parsed, dict) else {}
            parsed_windows.append(pw)
            meta.append({"y0": wy0, "y1": wy1,
                         "panel_kind": _norm_panel_kind(pw.get("panel_kind")),
                         "desc": str(pw.get("description") or "")[:80]})

    if not any(pw for pw in parsed_windows):
        return None, meta

    kinds = [m["panel_kind"] for m in meta]
    if any(k == "chrome" for k in kinds):
        kind = "chrome"
    else:
        kind = next((k for k in ("story", "system", "caption")
                     if k in kinds), "empty")
    subjects: List[str] = []
    seen: set = set()
    for pw in parsed_windows:
        for s in (pw.get("subjects") or []):
            s = str(s)
            if s and s not in seen:
                seen.add(s)
                subjects.append(s)
    order = {"calm": 0, "tense": 1, "intense": 2, "explosive": 3}
    intensities = [str(pw.get("intensity") or "").lower()
                   for pw in parsed_windows]
    intensities = [v for v in intensities if v in order]
    merged = {
        "description": " / ".join(
            str(pw.get("description") or "").strip()
            for pw in parsed_windows
            if str(pw.get("description") or "").strip())[:600],
        "subjects": subjects[:12],
        "action": next((str(pw.get("action") or "").strip()
                        for pw in parsed_windows
                        if str(pw.get("action") or "").strip()), ""),
        "dialogue": " ".join(str(pw.get("dialogue") or "").strip()
                             for pw in parsed_windows
                             if str(pw.get("dialogue") or "").strip())[:400],
        "setting": next((str(pw.get("setting") or "").strip()
                         for pw in parsed_windows
                         if str(pw.get("setting") or "").strip()), ""),
        "intensity": (max(intensities, key=lambda v: order[v])
                      if intensities else ""),
        "panel_kind": kind,
        "strikes_or_weapons": next(
            (str(pw.get("strikes_or_weapons") or "").strip().lower()
             for pw in parsed_windows
             if str(pw.get("strikes_or_weapons") or "").strip().lower()
             not in ("", "none")), "none"),
        "sfx_text": " ".join(str(pw.get("sfx_text") or "").strip()
                             for pw in parsed_windows
                             if str(pw.get("sfx_text") or "").strip())[:200],
        # pu_v4: any-window OR — one hedged window makes the strip uncertain
        "uncertain": any(bool(pw.get("uncertain")) for pw in parsed_windows),
        # pu_v5: concatenate windows' structured actions (reading order)
        "actions": [a for pw in parsed_windows
                    for a in (pw.get("actions") or [])
                    if isinstance(a, dict)][:6],
    }
    return merged, meta


def _norm_panel_kind(v: Any) -> str:
    v = str(v or "").strip().lower()
    return v if v in ("story", "chrome", "empty", "caption", "system") else "story"


# --- bubble/text-on-plain reclassification (the recurring "husk" root) --------
# A panel that is ONLY a speech/shout/caption bubble or a line of text on a plain/
# blank/white/empty background — with NO drawn scene — is a CAPTION: its words ride
# the narration and the bubble is never shown. The model labels this 'story' (or
# 'system') non-deterministically, which protects an EMPTY-bubble husk on screen
# (Nano ch1 p000020). The model is the describer; a deterministic rule on its own
# description/subjects is the guarantee. A real IN-WORLD system/stat/HUD/status
# window is a STORY VISUAL and must NEVER be swept up by this rule.

# a flat, featureless backdrop with no scene art ("plain white background",
# "blank background", "solid black background", "empty background").
_PLAIN_BG_RE = re.compile(
    r"\b(?:plain|blank|empty|solid|featureless|white|black|gr[ae]y|grey)\s+"
    r"(?:white\s+|black\s+|gr[ae]y\s+|grey\s+|colou?red\s+)?backgrounds?\b",
    re.IGNORECASE)
# the panel is ABOUT a bubble / balloon / caption-card / bare line of text.
_BUBBLE_OR_TEXT_RE = re.compile(
    r"\b(?:speech|shout|dialogue|thought)\s+(?:bubble|balloon)s?\b|"
    r"\b(?:bubbles?|balloons?)\b|\bcaptions?\b|"
    r"\b(?:line|box|card|panel)\s+of\s+text\b|"
    r"\b(?:text|words?)\b",
    re.IGNORECASE)
# an in-world game/system interface — a STORY VISUAL; its presence vetoes the rule.
_SYSTEM_WINDOW_RE = re.compile(
    r"\b(?:system|status|stat|stats|quest|hud|window|screen|interface|menu|"
    r"notification|alert|alarm|skill|level(?:\s*up)?|exp|hp|mp|inventory|"
    r"dungeon|guild|health\s*bar|progress\s*bar|map)\b",
    re.IGNORECASE)
# a subject that merely names the bubble/text itself (so subjects "empty or only
# describing the bubble/text" passes); anything else is a real drawn subject.
_BUBBLE_TEXT_SUBJECT_RE = re.compile(
    r"^(?:a\s+|an\s+|the\s+|some\s+)?(?:single\s+|lone\s+|plain\s+|white\s+|"
    r"black\s+|empty\s+)*(?:speech\s+|shout\s+|dialogue\s+|thought\s+)?"
    r"(?:bubbles?|balloons?|captions?|texts?|words?|letters?|"
    r"text\s+box(?:es)?|backgrounds?)\s*$",
    re.IGNORECASE)


def _is_caption_bubble_on_plain(description: Any, subjects: Any) -> bool:
    """True when the understanding describes ONLY a speech/shout/caption bubble or
    bare text on a plain/blank/white/empty background, with NO real drawn scene.

    Agnostic — keyed entirely on the model's own description/subjects, no manhwa
    specifics. Vetoed when the description names an in-world system/stat/HUD/status
    window (a story visual) or when subjects name a real drawn subject (a person,
    creature, place, object) rather than the bubble/text itself."""
    desc = str(description or "").strip()
    if not desc:
        return False
    # never demote an in-world system/stat/HUD window — it is a kept story visual
    if _SYSTEM_WINDOW_RE.search(desc):
        return False
    # subjects must be empty OR only describe the bubble/text itself
    subs = [str(s).strip() for s in (subjects or []) if str(s).strip()]
    if subs and not all(_BUBBLE_TEXT_SUBJECT_RE.match(s) for s in subs):
        return False
    # the description must read as a bubble/text panel AND name a plain backdrop
    return bool(_BUBBLE_OR_TEXT_RE.search(desc) and _PLAIN_BG_RE.search(desc))


def build_payload(panel: Dict[str, Any], prev_descs: List[str],
                  impact_regions: Optional[List[Dict[str, Any]]] = None,
                  nearby_dialogue: Optional[List[str]] = None
                  ) -> Dict[str, Any]:
    """Pure: the per-panel model input (OCR + cheap vision signals + rolling
    context for continuity). Image is attached separately by the caller.

    `impact_regions` (from impact_lettering.detect_impact_lettering) appends
    ONE context block when painted impact-SFX lettering was detected: OCR
    captures ZERO stylized SFX, so without this the model under-reads a stab
    panel as a calm one. The wording maps the signal to a GENERIC physical
    impact only — the calm-landscape control proved this shape does not
    hallucinate violence onto quiet panels (the detector never fires there)."""
    v = panel.get("vision") or {}
    labels = [x.get("desc") for x in (v.get("labels") or []) if x.get("desc")]
    objects = [x.get("name") for x in (v.get("objects") or []) if x.get("name")]
    payload = {
        "scene_file": panel.get("scene_file"),
        "ocr": (panel.get("ocr_clean") or "")[:900],
        "labels": labels[:12],
        "objects": objects[:12],
        "previous_panels": [d for d in prev_descs[-2:] if d],
    }
    if impact_regions:
        boxes = ", ".join(
            "x={0} y={1} w={2} h={3}".format(*(r.get("bbox") or [0, 0, 0, 0]))
            for r in impact_regions[:4])
        payload["impact_sfx_notice"] = (
            "Large impact-style SFX lettering is painted on this panel "
            f"(region(s): {boxes}). In manhwa this marks a physical impact — "
            "a strike, stab, blow, or crash. Describe the physical action "
            "accordingly, and transcribe the lettering if you can read it.")
    if nearby_dialogue:
        payload["nearby_dialogue"] = nearby_dialogue[:8]
    return payload


# pu_v4: hedged-subject wording that forces uncertain=True deterministically.
_UNCERTAIN_RE = re.compile(
    r"\b(?:or\b|unclear|possibly|unidentifi|indeterminate|hard to tell)",
    re.IGNORECASE)


def _norm_actions(raw: Any) -> List[Dict[str, str]]:
    """Pure: normalize pu_v5 `actions` — dict entries only, string-coerced,
    a verb plus at least one of actor/target, capped at 4."""
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for a in raw:
        if not isinstance(a, dict):
            continue
        actor = str(a.get("actor") or "").strip()
        verb = str(a.get("verb") or "").strip()
        target = str(a.get("target") or "").strip()
        if verb and (actor or target):
            out.append({"actor": actor, "verb": verb, "target": target})
        if len(out) >= 4:
            break
    return out


def unclear_strike(rec: Dict[str, Any]) -> bool:
    """Pure: a strike is being DELIVERED in this panel but the analyst could
    not commit to WHO strikes or WHO is struck — the forced-choice re-ask
    trigger for the pu_v5 direction contract."""
    if str(rec.get("strikes_or_weapons") or "") != "in_use":
        return False
    return any("unclear" in (str(a.get("actor") or "").lower(),
                             str(a.get("target") or "").lower())
               for a in rec.get("actions") or [])


def contested_strike(rec: Dict[str, Any]) -> bool:
    """Pure: a strike is being DELIVERED and TWO OR MORE people are drawn —
    the panel class where a wrong direction is both likely and invisible.

    pu_v5 verified only HEDGING (an 'unclear' actor), but the nano ch1
    inversion was stated CONFIDENTLY: 'a masked figure lunges at a young
    man' when the art shows the counter-kill. A confident wrong answer needs
    the same second look, with the neighbouring panels' dialogue as evidence
    — that dialogue is what settles it, and it is already on disk."""
    if str(rec.get("strikes_or_weapons") or "") != "in_use":
        return False
    from cast_identity import _looks_person
    people = [s for s in (rec.get("subjects") or []) if _looks_person(str(s))]
    return len(people) >= 2


def assemble_record(scene_file: str, parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure: normalize one model result into a panel record. A parse failure is
    recorded (never silently dropped) so resume can re-run just that panel."""
    if not isinstance(parsed, dict):
        # parse failure: no understanding -> treat as 'empty' so it is filtered
        # out of grouping (a panel we can't understand must not be narrated);
        # --resume still re-attempts it because error is recorded.
        return {"scene_file": scene_file, "description": "", "subjects": [],
                "action": "", "actions": [], "dialogue": "", "setting": "",
                "intensity": "unknown", "panel_kind": "empty",
                "strikes_or_weapons": "none", "sfx_text": "",
                "error": "parse_failed"}
    inten = str(parsed.get("intensity") or "").lower()
    sow = str(parsed.get("strikes_or_weapons") or "").strip().lower()
    description = str(parsed.get("description") or "").strip()
    subjects = [str(s) for s in (parsed.get("subjects") or []) if s]
    # pu_v4: the model is the describer, the regex is the guarantee — hedged
    # wording in a subject forces uncertain=True even when the model forgot
    # the flag (same philosophy as the husk override below).
    uncertain = bool(parsed.get("uncertain")) or any(
        _UNCERTAIN_RE.search(s) for s in subjects)
    kind = _norm_panel_kind(parsed.get("panel_kind"))
    # Deterministic husk override: a panel the model called 'story'/'system' that
    # is really ONLY a bubble/text on a plain background is a caption — its words
    # ride the narration and the bubble is never shown. Guarded so a real in-world
    # system/stat/HUD window or any drawn scene is never reclassified.
    if kind in ("story", "system") and _is_caption_bubble_on_plain(description, subjects):
        kind = "caption"
    return {
        "scene_file": scene_file,
        "description": description,
        "subjects": subjects,
        "action": str(parsed.get("action") or "").strip(),
        # pu_v5: structured direction (actor/verb/target, subjects-verbatim)
        "actions": _norm_actions(parsed.get("actions")),
        "dialogue": str(parsed.get("dialogue") or "").strip(),
        "setting": str(parsed.get("setting") or "").strip(),
        "intensity": inten if inten in
        ("calm", "tense", "intense", "explosive") else "unknown",
        "panel_kind": kind,
        # eyes wave: structured action fields (model-claimed; the DETECTOR
        # verdict `impact_sfx` is stamped separately in understand_panels).
        "strikes_or_weapons": sow if sow in ("none", "visible", "in_use")
        else "none",
        "sfx_text": str(parsed.get("sfx_text") or "").strip(),
        "uncertain": uncertain,
    }


def _detect_impact_regions(scene_path: Optional[str]) -> List[Dict[str, Any]]:
    """DETERMINISTIC impact-SFX regions for one scene image (the cheap CV pass
    that runs BEFORE the model call). Fail-soft: a missing/unreadable image or
    an unavailable cv2 returns [] — understanding must never crash on it.
    Imported lazily so this module keeps importing without cv2 (unit tests)."""
    if not scene_path or not os.path.exists(scene_path):
        return []
    try:
        import cv2
        from impact_lettering import detect_impact_lettering
        return detect_impact_lettering(cv2.imread(scene_path))
    except Exception:
        return []


def _scene_sha(scene_path: Optional[str]) -> str:
    """sha1 of the scene file's bytes; '' when scene_path is missing/falsy or
    unreadable. Fail-soft so a not-yet-materialized/synthetic scene_path
    (unit tests) never crashes the resume/emit path — it just never matches
    a real cached sha, which is the correct (re-run) outcome."""
    if not scene_path:
        return ""
    try:
        return input_sha(scene_path)
    except OSError:
        return ""


def understand_panels(items: List[Dict[str, Any]], call_fn: Callable[..., Any],
                      *, log: Callable[[str], None] = lambda _m: None,
                      prior: Optional[Dict[str, Dict[str, Any]]] = None,
                      concurrency: int = 1,
                      impact_fn: Optional[Callable[
                          [Optional[str]], List[Dict[str, Any]]]] = None
                      ) -> List[Dict[str, Any]]:
    """Describe each panel in order, threading rolling context (the last 2
    panels). `call_fn(payload, image_path) -> parsed dict|None` is injected.
    `prior` (scene_file -> good record) lets --resume skip done panels.
    `impact_fn(scene_path) -> regions` (default: the real impact-lettering
    detector, fail-soft) runs BEFORE each model call; its verdict is stamped
    on the record as `impact_sfx` — DETECTOR-owned, never model-claimed —
    and, when present, injects the impact context block into the payload.

    concurrency>1 runs panels in BATCHES of that size: every panel in a batch
    shares the SAME context (the descriptions taken BEFORE the batch), so order
    and continuity are preserved — only batch-mates can't see each other, which
    is negligible since the window is just 2 panels. The GPU then processes the
    batch at once (needs ollama OLLAMA_NUM_PARALLEL>=concurrency to parallelize)."""
    from concurrent.futures import ThreadPoolExecutor
    prior = prior or {}
    conc = max(1, int(concurrency))
    # Downscale over-tall images before EVERY model call (single-pass + both
    # re-asks + tall-strip windows) so the vision encoder can't OOM — one wrapper
    # instead of touching each of the ~5 call sites. No-op for images that fit
    # (short panels, already-small tall-strip windows).
    _base_call = call_fn

    def call_fn(payload, image_path):  # noqa: F811 — intentional shadow
        safe, tmp = _model_safe_image(image_path)
        try:
            return _base_call(payload, safe)
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    detect_impact = impact_fn if impact_fn is not None else _detect_impact_regions
    out: List[Dict[str, Any]] = []
    prev_descs: List[str] = []

    # Neighbouring OCR per panel (+-3 in reading order): the evidence that
    # settles who-struck-whom lives a panel or two away ("serves you right",
    # "how did a kid kill one of our members"). Precomputed, no model cost.
    _order = [it for it in items if it.get("scene_file")]
    _near: Dict[str, List[str]] = {}
    for _i, _it in enumerate(_order):
        _win = []
        for _j in range(max(0, _i - 3), min(len(_order), _i + 4)):
            _d = str(_order[_j].get("ocr_clean") or "").strip()
            if _d and _j != _i:
                _win.append(f"{_order[_j].get('scene_file')}: {_d[:160]}")
        _near[str(_it["scene_file"])] = _win

    def _understand(it: Dict[str, Any], ctx: List[str]) -> Dict[str, Any]:
        regions = detect_impact(it.get("scene_path")) or []
        dims = _tall_dims(it.get("scene_path"))
        if dims:
            parsed, wmeta = understand_tall_strip(
                it, ctx, call_fn, dims, impact_regions=regions)
            rec = assemble_record(it.get("scene_file"), parsed)
            rec["tall_windows"] = wmeta
            log(f"[panel] {it.get('scene_file')}: extreme-tall "
                f"{dims[0]}x{dims[1]} -> {len(wmeta)} windows "
                f"-> {rec.get('panel_kind')}")
        else:
            rec = assemble_record(
                it.get("scene_file"),
                call_fn(build_payload(it, ctx, impact_regions=regions),
                        it.get("scene_path")))
        # pu_v4 forced-choice re-ask: ONE retry that demands a commitment on a
        # hedged subject; accepted only when the second read actually commits
        # (uncertain cleared) — else the hedged record stands. Tall strips are
        # skipped (windowed re-runs are expensive; the flag alone suffices).
        # pu_v5 extends the trigger: a strike being DELIVERED (in_use) with an
        # 'unclear' actor/target demands commitment on WHO strikes WHOM —
        # accepted only when the second read clears the unclear direction.
        strike_hedge = unclear_strike(rec)
        if ((rec.get("uncertain") or strike_hedge)
                and not rec.get("error") and not dims):
            payload = build_payload(it, ctx, impact_regions=regions)
            if strike_hedge and not rec.get("uncertain"):
                notice = (
                    "Your first read marked a strike being DELIVERED in this "
                    "panel but could not tell WHO strikes or WHO is struck. "
                    "Look again at the pose, weapon direction, and impact "
                    "point, and COMMIT actor and target to specific subjects. "
                    "If you still cannot commit, keep 'unclear'.")
            else:
                notice = (
                    "Your first read could not identify a subject: "
                    + "; ".join(rec.get("subjects") or [])[:300]
                    + ". Look again and COMMIT to the single most likely "
                    "reading of each unclear subject (e.g. 'a tree branch', "
                    "'an arm'). If you still cannot commit, keep "
                    "uncertain=true.")
            payload["forced_choice_notice"] = notice
            try:
                second = assemble_record(
                    it.get("scene_file"),
                    call_fn(payload, it.get("scene_path")))
            except Exception:
                second = {"scene_file": it.get("scene_file"),
                          "error": "reask_failed"}
            if (not second.get("error") and not second.get("uncertain")
                    and not (strike_hedge and unclear_strike(second))):
                second["reask"] = True
                rec = second
                log(f"[panel] {it.get('scene_file')}: forced-choice re-ask "
                    "committed")
        # pu_v6 DIRECTION VERIFICATION: a confident read of a two-person
        # strike is the panel class that shipped the inverted kill. Re-ask
        # ONCE with the neighbouring dialogue attached — that text is what
        # decides it — and adopt only the ACTIONS from the second read, so a
        # bad second look can change direction but never corrupt the
        # description, subjects, or panel_kind.
        if (contested_strike(rec) and not rec.get("error") and not dims
                and not rec.get("reask")):
            near = _near.get(str(it.get("scene_file")) or "", [])
            payload = build_payload(it, ctx, impact_regions=regions,
                                    nearby_dialogue=near)
            payload["forced_choice_notice"] = (
                "A strike is being DELIVERED here and MORE THAN ONE person is "
                "drawn, so WHO strikes WHOM is easy to get backwards — and "
                "getting it backwards inverts the whole story. Re-read the "
                "panel: follow the weapon/arm direction, who is braced vs "
                "recoiling, whose body the impact lands on, and where the "
                "blood or impact marks actually sit (marks on a figure mean "
                "that figure was HIT). Then check nearby_dialogue: what "
                "characters SAY about this moment (a taunt, a shocked "
                "question, an accusation about who killed whom) is stronger "
                "evidence than the pose. Return `actions` with actor and "
                "target committed to the exact subject wording; use "
                "'unclear' ONLY if the evidence genuinely does not settle it.")
            try:
                second = assemble_record(
                    it.get("scene_file"),
                    call_fn(payload, it.get("scene_path")))
            except Exception:
                second = {"scene_file": it.get("scene_file"),
                          "error": "reask_failed"}
            new_actions = second.get("actions") or []
            if (not second.get("error") and new_actions
                    and not unclear_strike(second)):
                if new_actions != rec.get("actions"):
                    log(f"[panel] {it.get('scene_file')}: direction re-ask "
                        f"CHANGED attribution -> {new_actions}")
                rec["actions"] = new_actions
                rec["direction_reask"] = True
        # DETECTOR-owned impact verdict — stamped AFTER assemble_record so the
        # model can never claim or override it (the deterministic signal the
        # impact_mismatch QA gate reads back from understood.json).
        rec["impact_sfx"] = {"present": bool(regions), "regions": len(regions)}
        # Content-keyed provenance, stamped at emit time (see the prior.get(sf)
        # acceptance check below, which requires both to still match on resume).
        rec["scene_sha"] = _scene_sha(it.get("scene_path"))
        rec["prompt_version"] = _expected_version(it)
        return rec

    def _flush(batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        ctx = list(prev_descs)          # context snapshot taken BEFORE the batch
        if conc == 1 or len(batch) == 1:
            recs = [_understand(it, ctx) for it in batch]
        else:
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                recs = list(ex.map(lambda it: _understand(it, ctx), batch))
        for rec in recs:
            if rec.get("error"):
                log(f"[panel] {rec.get('scene_file')}: parse failed")
            out.append(rec)
            prev_descs.append(rec.get("description", ""))

    batch: List[Dict[str, Any]] = []
    for it in items:
        sf = it.get("scene_file")
        if not sf:
            continue
        done = prior.get(sf)
        # Resume acceptance is content-keyed, not name-keyed: the scene's own
        # pixels AND the prompt that produced the description must both still
        # match, else new art under an old filename (or a prompt rewrite)
        # would silently reuse stale understanding. A record missing either
        # key (pre-Task-12 cache) is a legacy record — NOT accepted; it
        # re-runs once and gets stamped, which is the intended one-time
        # migration cost.
        if (done and done.get("description") and not done.get("error")
                and done.get("scene_sha") == _scene_sha(it.get("scene_path"))
                and done.get("prompt_version") == _expected_version(it)):
            _flush(batch)                # emit the pending batch first (keep order)
            batch = []
            out.append(done)
            prev_descs.append(done.get("description", ""))
            continue
        batch.append(it)
        if len(batch) >= conc:
            _flush(batch)
            batch = []
    _flush(batch)
    return out


def _scene_items_in_order(vision: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = [it for it in (vision.get("items") or []) if it.get("scene_file")]
    items.sort(key=lambda it: (int(it.get("scene_id") or 0),
                               str(it.get("scene_file"))))
    return items


# --- publication-chrome text signal -----------------------------------------
# Keyword source of truth lives in two siblings with DIFFERENT match semantics:
#   tools/prep_qa.py        `_CHROME_NARR_RE`  (does narration mention chrome?)
#   tools/scene_chrome.py   `_CREDITS_RE`/`_SITE_PLUG_RE` (is OCR a chrome page?)
# Neither covers the recruitment/ad vocabulary that OVER-FIRES the in-world
# rescue (a "join our Discord to apply" recruitment card read as in-world chat).
# We mirror + extend that vocabulary here, kept in sync by the comment above,
# rather than importing — those regexes are tuned for OCR/narration strings, not
# the model's free-text description/action which is what we gate on. Phrasing is
# deliberately broad: a panel that READS like publication furniture is chrome
# even when it carries a speech-balloon, so the rescue must never promote it.
_CHROME_FURNITURE_RE = re.compile(
    r"\b("
    r"discord|patreon|subscrib\w*|"
    r"recruit\w*|"                       # "recruiting", "recruitment card"
    r"translator|translators|translat(?:ion|ed\s+by)|"
    r"scanlat\w*|typeset\w*|proofread\w*|redraw\w*|cleaner|cleaning\s+team|"
    r"raw\s+provider|"
    r"join\s+(?:our|the|us)|"            # "join our Discord", "join the team"
    r"support\s+(?:us|the\s+team)|"
    r"follow\s+us|"
    r"thanks?\s+for\s+reading|"
    r"next\s+chapter|next\s+episode|"
    r"end\s*card|"
    r"watermark|"
    r"credits?\s+page|staff\s+credits?|"
    # creator-credit roles on a title/cover/credits card. 'autor|artista' are the
    # Spanish/PT scanlation labels (the Nano-Machine end-card); a STORY panel's
    # OCR never carries these, and an in-world status/skill window never does.
    r"autor\w*|artista|art\s+by|story\s+by|written\s+by|illustrat\w*|"
    r"created\s+by|character\s+design|original\s+(?:work|story|webtoon|comic)|"
    r"read\s+(?:on|the\s+rest|more)\s+(?:at|on)|read\s+it\s+(?:on|at)|"
    r"early\s+(?:access|chapters?|release)"
    r")\b",
    re.IGNORECASE)


def _looks_like_chrome_furniture(*texts: str) -> bool:
    """True when any of the given strings (a panel's description / action /
    dialogue) reads like PUBLICATION furniture — scanlator credits, a Discord/
    Patreon promo, a recruitment card, a 'thanks for reading / next chapter'
    end-card. Such a panel is chrome even when it carries dialogue-like text, so
    the in-world rescue must NOT promote it to story. In-world chat / game-UI /
    status screens use none of this vocabulary and pass through untouched."""
    for t in texts:
        if t and _CHROME_FURNITURE_RE.search(str(t)):
            return True
    return False


# --- in-world screen rescue -------------------------------------------------
# The classifier reliably mis-buckets an IN-WORLD device/app screen as 'chrome'
# when it looks like platform UI (an episode list, a feed) — even though the
# prompt says such screens are story (ORV ep1 p000003: the in-world webnovel's
# episode list + the reader comment "WHY DOESN'T ANYONE READ THIS? IT'S A
# MASTERPIECE!" — iconic, must show). Real publication chrome (covers, episode/
# stat cards, publisher credits) carries NO character speech balloon; an
# in-world screen showing dialogue does. The balloon SHAPE is the signal — no
# hardcoded text. Trust the trained bubble detector: a CONFIDENT, COMPACT
# balloon over the panel's dialogue promotes chrome -> story.

def _is_inworld_balloon(dets, w: int, h: int, *,
                        conf_min: float = 0.70, area_max: float = 0.40) -> bool:
    """True when a detection list has a real speech balloon: at least one box
    that is both confident (>= conf_min) AND compact (<= area_max of the panel).
    The compactness gate rejects a screen-sized false positive (e.g. the whole
    stats box at ~0.6 area); the confidence gate rejects low-score UI-row
    detections (~0.2-0.5). The genuine balloon (ORV p000003: conf 0.96, ~0.14
    area) clears both."""
    area = float(max(1, w * h))
    for d in dets:
        x1, y1, x2, y2, s = d[0], d[1], d[2], d[3], d[4]
        af = (abs(int(x2) - int(x1)) * abs(int(y2) - int(y1))) / area
        if float(s) >= conf_min and af <= area_max:
            return True
    return False


def _load_bubble_detector(device: str = "mps"):
    cand = os.path.join(os.path.dirname(_TD), "manhwa-cropper")
    if cand not in sys.path:
        sys.path.insert(0, cand)
    from manhwa_cropper.detectors.bubbles import BubbleDetector
    return BubbleDetector(device=device)


def apply_inworld_screen_overrides(
        panels: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
        *, device: str = "mps",
        detect_fn: Optional[Callable[[str], Optional[Tuple[int, int, Any]]]] = None,
        log: Callable[[str], None] = print) -> int:
    """Promote chrome panels that carry a real speech balloon over dialogue to
    'story' (an in-world screen). Returns the count promoted. Fail-soft: if the
    detector or an image is unavailable, the classification is left untouched.

    A panel whose description/action/dialogue reads like PUBLICATION furniture
    (scanlator credits, a Discord/Patreon recruitment promo, a 'thanks for
    reading / next chapter' end-card) is NEVER promoted — text-heavy ad/credit
    cards otherwise over-fire this rescue (Ch141 p000068: a translator-
    recruitment card was read as in-world chat). Such panels stay chrome so the
    grouper drops them. `detect_fn(scene_path) -> (w, h, dets) | None` is an
    injectable seam (defaults to the real cv2 + bubble detector)."""
    # Structural demotion (the inverse of the rescue): a panel Gemma tagged
    # 'story'/'caption' whose OCR or description reads like a CREDITS / cover card
    # (author/artist roles, scanlator credits) is publication furniture mislabeled
    # as art — demote to chrome so the grouper drops it. 'system'/'empty' are NEVER
    # in scope, and a status/skill window carries none of this vocabulary, so a
    # plot-critical system panel cannot be swept up.
    ocr_by_file = {it.get("scene_file"): (it.get("ocr_clean") or "") for it in items}
    demoted = 0
    for p in panels:
        if p.get("panel_kind") in ("story", "caption") and _looks_like_chrome_furniture(
                ocr_by_file.get(p.get("scene_file"), ""),
                p.get("description"), p.get("action"), p.get("dialogue")):
            p["panel_kind"] = "chrome"
            demoted += 1
    if demoted:
        log(f"[credits] demoted {demoted} story/caption -> chrome (credits/cover card)")

    cand = [p for p in panels
            if p.get("panel_kind") == "chrome" and (p.get("dialogue") or "").strip()
            and not _looks_like_chrome_furniture(
                p.get("description"), p.get("action"), p.get("dialogue"))
            # WINDOWED CHROME EVIDENCE IS FINAL: an extreme-tall strip was
            # condemned because gemma SAW the title/credits at readable window
            # scale (b9e70c7). Its merged "dialogue" is novel/caption text and
            # a balloon somewhere in 7000px means nothing — never rescue it
            # (ORV Ep0: the 800x7540 cover strip was un-chromed exactly here).
            and not any(m.get("panel_kind") == "chrome"
                        for m in (p.get("tall_windows") or []))]
    if not cand:
        return 0
    path_by_file = {it.get("scene_file"): it.get("scene_path") for it in items}
    if detect_fn is None:
        try:
            import cv2
            det = _load_bubble_detector(device)
        except Exception as e:                                       # pragma: no cover
            log(f"[inworld] bubble detector unavailable ({e}) — override skipped")
            return 0

        def detect_fn(sp: str):                                      # noqa: F811
            img = cv2.imread(sp) if sp else None
            if img is None:
                return None
            h, w = img.shape[:2]
            try:
                return w, h, det.detect(img, imgsz=1024, conf=0.20)
            except Exception:                                        # pragma: no cover
                return None
    n = 0
    for p in cand:
        sp = path_by_file.get(p.get("scene_file"))
        res = detect_fn(sp) if sp else None
        if not res:
            continue
        w, h, dets = res
        if _is_inworld_balloon(dets, w, h):
            p["panel_kind"] = "story"
            # stamp the marker render_prep keys on to keep the screen text
            # (it routes in-world screens to the document treatment)
            subj = [s for s in (p.get("subjects") or []) if s]
            if not any("in-world screen" in str(s).lower() for s in subj):
                subj.append("an in-world screen")
            p["subjects"] = subj
            n += 1
            log(f"[inworld] {p.get('scene_file')}: chrome->story "
                f"(speech balloon over dialogue {(p.get('dialogue') or '')[:48]!r})")
    return n


# --- system-card override (deterministic, trained-detector backed) ----------
# An in-world SYSTEM / NOTIFICATION / STAT card (Nano ch1 p000114 "7TH
# GENERATION NANO MACHINE, STARTING ACTIVATION.") is text on a flat field the
# CHARACTER perceives as a UI element — its words ARE the on-screen story beat,
# so it must be kept + shown (panel_kind 'system'). gemma classifies it
# NON-deterministically: an earlier roll got it right, a fresh roll called it
# 'caption' (text-on-plain) -> the grouper folded it and it was never shown. The
# trained webtoon YOLO has a dedicated system-window class (system_box in the
# legacy 6-class model, system_ui in v3 — resolved by NAME from model.names)
# that fires reliably on these cards, so we use it as a DETERMINISTIC override:
# the verdict no longer depends on gemma's roll. AGNOSTIC — no series/word list.
#
# The detector is NOT a clean signal on its own (measured on Nano ch1): it ALSO
# fires class-1 on a tall multi-frame STORY strip (p000005, a falling character,
# cover 0.93) and on a plain SPEECH-BUBBLE husk (p000020 "PEASANT BLOOD...",
# cover 0.92). So the override is GUARDED three ways:
#   * only rescue panels the grouper would FOLD/DROP (caption/empty) — a 'story'
#     panel with real subjects is trusted and never demoted by a detector FP;
#   * never a panel the understanding describes as a speech/dialogue/thought
#     bubble (character speech, not a system message) — excludes the husk;
#   * require the system_box to DOMINATE the panel (a system card IS the panel),
#     so a spurious small box can't promote a real narration caption.
# This runs LAST (after the husk demotion + the in-world rescue) so a trained
# detection has the final word. Mirrors render_prep's `_sys_boxes` loader/predict.
_SPEECH_BUBBLE_RE = re.compile(
    r"\b(?:speech|dialogue|thought|talk|word)\s*(?:bubble|balloon)s?\b"
    r"|\b(?:bubbles?|balloons?)\b", re.IGNORECASE)
_SYS_BOX_MIN_COVER = 0.20
# v3 (2026-07-15): the legacy model's system_box fires 0.96-cover on PLAIN
# TEXT CARDS (measured on ORV Ep0 "BACK THEN," — promoted a caption to
# system and put a white card on screen); v3's system_ui is trained on real
# UI windows only (0.00 on the same card) and calls text cards free_text.
_DEFAULT_PANEL_WEIGHTS = os.path.join(
    os.path.dirname(_TD), "assets", "models", "webtoon_panels_v3.pt")


def _describes_speech_bubble(description: Any) -> bool:
    """True when the understanding describes the panel as a speech / dialogue /
    thought bubble or balloon — character speech, NOT a system message. A real
    in-world system / notification / stat card is a box / window / text-field,
    never a balloon, so this guards the bubble husk out of the override."""
    return bool(_SPEECH_BUBBLE_RE.search(str(description or "")))


def apply_system_card_overrides(
        panels: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
        *, weights_path: Optional[str] = None, device: str = "mps",
        detect_fn: Optional[Callable[[str], Optional[float]]] = None,
        log: Callable[[str], None] = print) -> int:
    """Force panel_kind='system' on a folded caption/empty panel that the trained
    system_box detector fires on (the in-world notification / stat card the
    grouper would otherwise drop). Returns the count promoted. Fail-SOFT: a
    missing weights file or an unavailable detector logs loudly and leaves every
    classification untouched — it never crashes the stage.

    `detect_fn(scene_path) -> system_box_coverage | None` is an injectable seam
    (defaults to the real cv2 + trained YOLO, mirroring render_prep's `_sys_boxes`:
    YOLO(weights), predict(conf=0.30), keep the system classes resolved by NAME
    via studio.detect.yolo_panels.system_class_ids). It returns the fraction of
    the panel covered by system-window detections, or None if the image is missing."""
    path_by_file = {it.get("scene_file"): it.get("scene_path") for it in items}
    cand = [p for p in panels
            if str(p.get("panel_kind") or "").strip().lower() in ("caption", "empty")
            and not _describes_speech_bubble(p.get("description"))]
    if not cand:
        return 0
    if detect_fn is None:
        wp = weights_path or _DEFAULT_PANEL_WEIGHTS
        if not os.path.exists(wp):
            log(f"[system-card] panel weights missing ({wp}) — "
                "system-card override DISABLED")
            return 0
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO
            from studio.detect.yolo_panels import system_class_ids
            model = YOLO(wp)
            sys_ids = system_class_ids(getattr(model, "names", None))
        except Exception as e:                                       # pragma: no cover
            log(f"[system-card] detector unavailable ({e}) — override skipped")
            return 0

        def detect_fn(sp: str):                                      # noqa: F811
            img = cv2.imread(sp) if sp else None
            if img is None:
                return None
            try:
                r = model.predict(img, conf=0.30, device=device,
                                  verbose=False)[0]
            except Exception:                                        # pragma: no cover
                return None
            boxes = []
            if r.boxes is not None:
                for (x1, y1, x2, y2), c in zip(
                        r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    if int(c) in sys_ids:            # system_box / system_ui
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
            h, w = img.shape[:2]
            if h <= 0 or w <= 0 or not boxes:
                return 0.0
            # union coverage on a downscaled grid (mirrors render_prep.bubble_coverage)
            s = 4
            grid = np.zeros((max(1, h // s), max(1, w // s)), np.uint8)
            for (x1, y1, x2, y2) in boxes:
                grid[max(0, y1 // s): max(0, y2 // s),
                     max(0, x1 // s): max(0, x2 // s)] = 1
            return float(grid.mean())
    n = 0
    for p in cand:
        sp = path_by_file.get(p.get("scene_file"))
        cov = detect_fn(sp) if sp else None
        if cov is not None and cov >= _SYS_BOX_MIN_COVER:
            prev = p.get("panel_kind")
            p["panel_kind"] = "system"
            n += 1
            log(f"[system-card] {p.get('scene_file')}: {prev}->system "
                f"(system_box cover {cov:.2f})")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="ollama")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="")
    # 0.0 (2026-07-16): understanding is ANALYSIS, not writing — its subjects
    # wording feeds cast resolution, panel_kind gates, and the narration's
    # factual source, so sampling variance here is pure downstream noise.
    # Creative temperature belongs to the writer (0.2) and punchup (0.7).
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=400)
    ap.add_argument("--resume", action="store_true",
                    help="keep good panel records in --out, redo only failures")
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("STUDIO_UNDERSTAND_CONCURRENCY", "3")),
                    help="panels understood per batch (needs ollama "
                         "OLLAMA_NUM_PARALLEL>=this to actually parallelize)")
    ap.add_argument("--panel-weights", default=_DEFAULT_PANEL_WEIGHTS,
                    help="trained webtoon YOLO — its system_box class (class 1) "
                         "deterministically forces an in-world system / "
                         "notification card the model folded as a caption to "
                         "panel_kind 'system' (kept + shown)")
    ap.add_argument("--device", default="mps",
                    help="torch device for the system_box detector")
    args = ap.parse_args()

    vision = load_json(args.vision_manifest)
    items = _scene_items_in_order(vision)
    if not items:
        raise SystemExit("no vision items (expected key: items)")

    client = None
    model = args.ollama_model
    if args.backend == "vertex":
        from google import genai
        if not args.project or not args.location:
            raise SystemExit("--project/--location required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)
        model = args.model

    prior: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.out):
        try:
            prior = {p.get("scene_file"): p for p in
                     (load_json(args.out).get("panels") or [])
                     if p.get("scene_file")}
        except Exception:
            prior = {}

    def call_fn(payload: Dict[str, Any], scene_path: Optional[str]):
        parsed, _raw, _usage = _call_model_with_backoff(
            client=client, model=model, system_instruction=SYSTEM,
            user_payload=payload, image_paths=[scene_path] if scene_path else [],
            response_schema=PANEL_SCHEMA, max_output_tokens=args.max_output_tokens,
            temperature=args.temperature, backoff_max=60.0, backend=args.backend)
        return parsed

    conc = max(1, int(args.concurrency)) if args.backend == "ollama" else 1
    if conc > 1:
        print(f"[understand] batched-parallel: {conc} panels/batch "
              f"({len(items)} panels)", flush=True)
    panels = understand_panels(items, call_fn,
                               log=lambda m: print(m, flush=True), prior=prior,
                               concurrency=conc)
    promoted = apply_inworld_screen_overrides(
        panels, items, log=lambda m: print(m, flush=True))
    if promoted:
        print(f"[ok] in-world screen rescue: {promoted} chrome->story")
    # Deterministic system-card override (trained system_box detector) — runs
    # LAST so a real in-world notification/stat card the model folded as a
    # caption is reliably kept + shown as 'system'. Fail-soft if weights missing.
    sysn = apply_system_card_overrides(
        panels, items, weights_path=args.panel_weights, device=args.device,
        log=lambda m: print(m, flush=True))
    if sysn:
        print(f"[ok] system-card override: {sysn} caption/empty->system")
    # Centralize the chrome/story verdict: stamp panel_kind back onto the vision
    # manifest so the SINGLE chrome chokepoint (scene_chrome.is_chrome_scene —
    # used by story_group, render_prep AND prep_qa) defers to the understanding
    # everywhere. No downstream module re-derives chrome from OCR and disagrees.
    # ORDER MATTERS: this write-back runs BEFORE the understood.json dump so
    # vision.mtime <= understood.mtime always holds (the historical inversion —
    # vision re-stamped ~0.03s AFTER understood — false-flagged freshness) and
    # the understood dump's _meta input-sha hashes the FINAL vision bytes.
    by_file = {p.get("scene_file"): p for p in panels if p.get("scene_file")}
    changed = False
    for it in (vision.get("items") or []):
        p = by_file.get(it.get("scene_file"))
        if not p:
            continue
        k = p.get("panel_kind")
        if k and it.get("panel_kind") != k:
            it["panel_kind"] = k
            changed = True
        # Also stamp the SUBJECTS the multimodal pass identified, so the narration
        # generator NAMES what's actually there and can't rename it (a 'beast' must
        # not become a 'hound', two must not become 'a pack'). Grounding via the
        # understanding itself — no creature wordlist to maintain.
        subj = [str(s) for s in (p.get("subjects") or []) if s]
        if subj and it.get("subjects") != subj:
            it["subjects"] = subj
            changed = True
    if changed:
        # A plain rewrite of vision's own content, made atomic (no declared
        # inputs — it is not derived FROM another manifest, it is patched in
        # place).
        write_manifest(args.vision_manifest, vision, inputs=(), tool="panel_understand")
        print(f"[ok] stamped panel_kind + subjects onto {os.path.basename(args.vision_manifest)}")

    write_manifest(args.out, {
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "model": model, "count": len(panels), "panels": panels},
        inputs=(args.vision_manifest,), tool="panel_understand")

    ok = sum(1 for p in panels if p.get("description") and not p.get("error"))
    print(f"[ok] wrote={args.out} panels={len(panels)} understood={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
