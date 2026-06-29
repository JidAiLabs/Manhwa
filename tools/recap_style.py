#!/usr/bin/env python3
"""Shared recap-channel writing rules and deterministic style diagnostics."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Sequence


RECAP_STYLE_RULES = """RECAP-CHANNEL WRITING RULES — apply these while preserving
the one-line-per-panel contract. Every surviving panel still gets a line and a
cut; compression removes verbal drag, NEVER panel coverage or plot:
1. NO SCREEN READING: do not narrate weather, lighting, hair, clothing, poses,
facial expressions, or other details the viewer can already see unless that
detail changes the plot. NEVER name the shot, camera, panel, image, or frame,
and NEVER open with "A close-up shot shows...", "The panel focuses on...", or
"A wide shot captures..." — narrate the STORY and the action, never the picture.
Narrate the ACTION and its impact/stakes, never how the panel is DRAWN: NEVER
describe visual effects or rendering — no "motion blur", "speed lines", "blurry
streaks", "creating ... effects", "is depicted", "the panel/image shows". For an
action/motion panel (a strike, a dash, an impact), say WHAT happens and the
consequence (who strikes whom, the force, the result) — e.g. "He whips his blade
around in a vicious arc" — not "a sword is being swung with motion blur".
Each line must advance action, cause, stakes, thought, revelation, or
consequence. A quick panel may continue the surrounding thought, but every panel
line must remain an independently speakable complete clause.
2. POINT, DON'T PAINT: when the source naturally evokes a familiar anime, game,
movie, superhero, internet, or everyday-life reference, one short comparison
can replace a paragraph of description. It is figurative framing, never a new
plot fact, and must fit the genre and visible moment.
3. RATION NAMES: introduce the protagonist by name once, then usually use a
pronoun or a relaxed genre-appropriate stand-in such as "our guy" or "our boy".
Repeat the real name only for clarity or emotional weight. Never let the full
name become a label repeated line after line.
4. ADD TEXTURE, NOT JOKES: eligible connective lines should occasionally carry
one dry observation, intimate aside, ironic understatement, or concise familiar
comparison. Aim for roughly one textured touch per four eligible lines. Serious
injury, grief, horror, and major dramatic reveals stay restrained.
5. COMPRESS DRAG: summarize dialogue, remove stage direction and repeated visual
description, and prefer short spoken clauses. Preserve every plot beat and every
panel line; do not target one-third of the panel count. Typical panel lines land
around 4-10 spoken words, quick actions can be 2-5, and only pivotal/reveal panels
need 12-18. This project's adaptation of the compression rule is higher
information density at the existing pace.
6. REVEAL PACING: NAME established cast members so the audience can follow who is
who — recognition is the priority, so NAME ESTABLISHED characters (including the
protagonist) normally on their OWN panels. Reserve a neutral handle ("the
stranger", "the intruder") ONLY for a figure THIS panel itself presents as
genuinely concealed, masked, hooded, glowing, silhouetted, transformed, or
newly-arrived AND not yet matched to a known character. Do NOT neutralize an
established character just because a separate mysterious figure is nearby, and do
NOT keep calling a clearly-shown, already-known character "the stranger". A
power/transformation reveal of an UNKNOWN figure is a mystery to preserve — but
once the story's own text or the character's established look identifies someone,
use their name."""


_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_POINTER_RE = re.compile(
    r"\b(?:like (?:a|an|he|she|they|it)|as if|basically|pretty much|"
    r"straight out of|the .*? version of|giving .*? vibes|think of|"
    r"god mode|speedrun\w*|tutorial|hud|stealth build|boss fight)\b",
    re.I,
)
_TEXTURE_RE = re.compile(
    r"\b(?:our guy|our boy|my guy|bro|bad timing|classic|apparently|"
    r"of course|somehow|at least|because why not|turns out|officially|"
    r"not exactly|so much for|take the hint|zero chill|god mode|"
    r"speedrun\w*|tutorial|stealth build)\b",
    re.I,
)
_VISUAL_ECHO_RE = re.compile(
    r"\b(?:pale moon|moonlight|misty|fog[- ](?:covered|drowned)|"
    r"shadows? (?:dance|loom|stretch)|hair (?:sways?|flows?|whips?)|"
    r"eyes? (?:widen|widened|narrow|narrows)|stares? in (?:pure )?shock|"
    r"wreathed in|crackling (?:blue )?(?:lightning|energy)|"
    r"glowing (?:blue )?(?:aura|silhouette|figure)|"
    r"leaving (?:him|her|them) reeling|the wind|silent mountains?)\b",
    re.I,
)
_CONCEALED_RE = re.compile(
    r"\b(?:silhouette|glowing|masked|hooded|disguised|transformed|unfamiliar|"
    r"mysterious|stranger|intruder|unknown|shadowy|newcomer|gear unlike|"
    r"cloaked|veiled)\b",
    re.I,
)
# Broader "this panel refers to the still-unresolved figure" cue: concealment
# (above) PLUS the transformation/power/advanced-gear signals a clear-view panel
# of that figure carries (a later panel of the blue-goggled, lightning-wreathed
# arrival). Used ONLY to disambiguate WHICH figure a panel shows once a window is
# already open — NOT to open one (opening stays gated to _CONCEALED_RE). Agnostic:
# generic power/gear vocabulary, never per-series words.
_POWER_GEAR_RE = re.compile(
    r"\b(?:lightning|sparks?|electric\w*|crackl\w*|energy|aura|blazing|radiant|"
    r"goggles|visor|helmet|futuristic|glow|glows|glowed)\b",
    re.I,
)
_IDENTITY_QUESTION_RE = re.compile(
    r"\b(?:who are you|who is (?:he|she|that|this)|who's (?:he|she|that)|"
    r"identify yourself|what are you|where did (?:he|she|they) come from)\b",
    re.I,
)
_GENERIC_PROTAGONIST_NAMES = {
    "our protagonist", "the protagonist", "protagonist", "our guy", "our boy",
    "the hero", "our hero", "main character", "mc",
}
# Generic, series-agnostic "familiar handle" the narrator slips to for the
# protagonist. These must NOT be attached to a figure whose identity the story
# has not resolved, so the unresolved-identity guard neutralizes them too.
_PROTAGONIST_HANDLE_RE = re.compile(
    r"\b(?:our guy|our boy|our hero|our protagonist|my guy)\b", re.I)
# Consecutive cue-free panels (with a different subject focus) after which an
# unresolved-identity window is treated as closed. Conservative on purpose:
# staying unresolved a few extra panels is safer than mislabeling the figure.
_UNRESOLVED_CLEAR_AFTER = 5

# SHOT/CAMERA prose: a line that NAMES the shot/panel/frame instead of narrating
# the story ("A close-up shot shows...", "The panel focuses on...", "A wide shot
# captures..."). This is the align-pad bug (D4) copying a panel's camera-prose
# understanding `description` verbatim. Pattern-based + agnostic: (a/an/the) +
# optional camera adjectives + a shot/picture noun, then a camera/presentation
# verb. The tight verb set keeps it off ordinary story lines ("The scene shifts.",
# "A long shadow falls...").
_SHOT_DESC_RE = re.compile(
    r"\b(?:an?|the)\s+"
    r"(?:(?:extreme|medium|wide|low|high|overhead|close[- ]?up|long|tight|"
    r"establishing|aerial|distant|dramatic|sweeping|bird['’]?s[- ]?eye)\s+){0,3}"
    r"(?:shot|panel|image|frame|scene|angle|view|close[- ]?up|"
    r"composition|perspective)\b"
    r"[^.?!]{0,40}?\b"
    r"(?:shows?|focus(?:es|ing)?(?:\s+on)?|captures?|depicts?|reveals?|"
    r"frames?|cuts?\s+to|zooms?(?:\s+(?:in|out))?|pans?\s+(?:to|across|over)|"
    r"displays?|showcases?|portrays?|presents?)\b",
    re.I,
)


# RENDERING / VISUAL-EFFECT prose: a line that describes HOW the panel is DRAWN —
# the motion blur, speed lines, streaks, or "X is depicted" — instead of WHAT
# happens in the story. These shipped in Nano ch1 on ACTION panels ("...is depicted
# through motion blur.", "A sword is being swung ... creating motion blur effects.",
# "A blade swings through the air with lethal speed."). The camera/shot regex above
# missed them. Agnostic: the vocabulary is rendering-craft language, never per-series
# words. PRECISE by construction — a CHARACTER moving/striking fast ("He moved with
# lethal speed.", "She lunged, blade flashing toward his throat.") names NO effect
# and is NOT matched. Two arms: (1) an explicit effect/depiction phrase; (2) an
# inanimate weapon shown swinging through EMPTY air (no actor, no impact) — the
# motion-blur panel rendered as prose.
_EFFECT_DESC_RE = re.compile(
    r"(?:\b(?:"
    r"motion[- ]blur"
    r"|speed[- ]lines?"
    r"|blurry\s+streaks?"
    r"|streaks?"
    r"|blur\s+effects?"
    r"|creating\b[^.?!]*?\beffects?"
    r"|(?:is|are)\s+depicted|depicted\s+through"
    r"|high[- ]speed\s+(?:motion|effects?)"
    r"|the\s+(?:panel|image|artwork)\s+(?:shows?|depicts?)"
    r")\b)"
    r"|"
    r"(?:\b(?:an?|the)\s+"
    r"(?:blade|sword|saber|sabre|katana|knife|dagger|spear|lance|axe|"
    r"scythe|hammer|fist|claw|whip|staff|weapon|arrow)\s+"
    r"(?:swings?|swung|swinging|slashes?|slashing|slices?|slicing|sweeps?|"
    r"sweeping|cuts?|cutting|arcs?|arcing|whips?|whipping)\s+"
    r"through\s+the\s+air\b)",
    re.I,
)


def is_shot_description(text: str) -> bool:
    """True when a narration line names the shot/camera/panel/frame ('A close-up
    shot shows...') OR describes the ARTWORK'S RENDERING / a visual effect ('motion
    blur', 'speed lines', '...is depicted', a weapon 'swings through the air')
    instead of narrating the story. Series-agnostic."""
    clean = _TAG_RE.sub("", str(text or ""))
    return bool(_SHOT_DESC_RE.search(clean) or _EFFECT_DESC_RE.search(clean))


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(_TAG_RE.sub("", str(text or "")))


def is_spoken_fragment(text: str) -> bool:
    """True for a panel line that cannot stand as its own TTS clip."""
    s = _TAG_RE.sub("", str(text or "")).strip()
    if not s:
        return True
    if s.endswith((",", ";", ":")):
        return True
    if re.match(r"^(?:\.{2,}|…)\s*", s):
        return True
    first = re.search(r"[A-Za-z]", s)
    if first and s[first.start()].islower():
        return True
    return False


def repair_spoken_line(text: str) -> str:
    """Make a dangling panel line independently speakable without new facts."""
    s = str(text or "").strip()
    tag = ""
    m = _TAG_RE.match(s)
    if m:
        tag, s = m.group(0), s[m.end():].strip()
    s = re.sub(r"^(?:\.{2,}|…)\s*", "", s).strip()
    low = s.lower()
    if low.startswith("leaving "):
        s = "That leaves " + s[len("leaving "):]
    elif low.startswith("wondering if "):
        s = "The question is whether " + s[len("wondering if "):]
    elif low.startswith("wondering who "):
        s = "The question is who " + s[len("wondering who "):]
    elif low.startswith("wondering what "):
        s = "The question is what " + s[len("wondering what "):]
    elif low.startswith("and "):
        s = s[len("and "):]
    elif low.startswith("with "):
        s = "The sequence continues with " + s[len("with "):]
    elif low.startswith("that "):
        s = "The truth is that " + s[len("that "):]
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s.endswith((",", ";", ":")):
        s = s[:-1].rstrip() + "."
    return (tag + s).strip()


def repair_spoken_fragments(beats_obj: Dict[str, Any]) -> int:
    """Repair fragment fallbacks in-place and keep joined narration in sync."""
    changed = 0
    for beat in beats_obj.get("beats") or []:
        panels = beat.get("panel_narration") or []
        for panel in panels:
            line = str(panel.get("line") or "")
            if not is_spoken_fragment(line):
                continue
            fixed = repair_spoken_line(line)
            if fixed and fixed != line:
                panel["line"] = fixed
                changed += 1
        if panels:
            beat["narration"] = " ".join(
                str(p.get("line") or "").strip() for p in panels
                if str(p.get("line") or "").strip())
    return changed


def _norm_line(text: str) -> str:
    """Normalize a panel line for duplicate comparison: drop mood tags, lowercase,
    collapse to alphanumeric tokens. So 'Ancestor...?' == '[panicked] Ancestor?'."""
    s = _TAG_RE.sub("", str(text or "")).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def dedupe_consecutive_panel_lines(beats_obj: Dict[str, Any]) -> int:
    """Merge out a panel whose narration line is an EXACT (normalized) duplicate of
    the IMMEDIATELY-PRECEDING panel's line (within or across beats) — the p95/p96
    "Ancestor...?" bug, where the same spoken line shipped over two panels. The
    earlier (better-placed) panel survives with the single line; the duplicate
    panel + its scene_file + its scene_selection entry are dropped, so the two cuts
    collapse to ONE and the line is voiced once. Never empties a beat (a beat whose
    only panel is the duplicate keeps it; the render-side hold then covers it).
    Returns the number of panels merged out. Agnostic + deterministic."""
    removed = 0
    prev_norm: str | None = None
    for beat in beats_obj.get("beats") or []:
        panels = beat.get("panel_narration") or []
        if not panels:
            continue
        kept: List[Dict[str, Any]] = []
        for idx, panel in enumerate(panels):
            norm = _norm_line(panel.get("line"))
            is_dup = bool(norm) and norm == prev_norm
            last = idx == len(panels) - 1
            # never empty a beat: if nothing kept yet and this is the last panel,
            # keep it even when it duplicates the previous line.
            if is_dup and not (last and not kept):
                removed += 1
                continue
            kept.append(panel)
            if norm:
                prev_norm = norm
        if len(kept) != len(panels):
            kept_files = {str(p.get("scene_file")) for p in kept
                          if p.get("scene_file")}
            beat["panel_narration"] = kept
            if isinstance(beat.get("scene_files"), list):
                beat["scene_files"] = [f for f in beat["scene_files"]
                                       if str(f) in kept_files]
            if isinstance(beat.get("scene_selection"), list):
                beat["scene_selection"] = [
                    s for s in beat["scene_selection"]
                    if isinstance(s, dict)
                    and str(s.get("scene_file")) in kept_files]
            beat["narration"] = " ".join(
                str(p.get("line") or "").strip() for p in kept
                if str(p.get("line") or "").strip())
    return removed


def script_lines(script_obj: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for sec in script_obj.get("sections") or []:
        values = sec.get("script_paragraphs") or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            text = value if isinstance(value, str) else str(
                (value or {}).get("text") or (value or {}).get("line") or "")
            text = _TAG_RE.sub("", text).strip()
            if text:
                out.append(text)
    return out


def panel_rows(beats_obj: Mapping[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for beat in beats_obj.get("beats") or []:
        panels = beat.get("panel_narration") or []
        if panels:
            for panel in panels:
                line = str(panel.get("line") or "").strip()
                if line:
                    out.append({
                        "scene_file": str(panel.get("scene_file") or ""),
                        "line": line,
                    })
        else:
            line = str(beat.get("narration") or "").strip()
            if line:
                files = beat.get("scene_files") or [""]
                out.append({"scene_file": str(files[0] if files else ""),
                            "line": line})
    return out


def _protagonist_names(cast_obj: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    for member in cast_obj.get("cast") or cast_obj.get("characters") or []:
        if not isinstance(member, dict):
            continue
        if not (member.get("is_protagonist")
                or str(member.get("role") or "").lower() == "protagonist"):
            continue
        candidates = [member.get("canonical_name")] + list(member.get("aliases") or [])
        for candidate in candidates:
            name = str(candidate or "").strip()
            if (not name or name.lower() in _GENERIC_PROTAGONIST_NAMES
                    or not re.search(r"[A-Za-z]", name)):
                continue
            names.append(name)
    names = sorted(set(names), key=lambda n: (-len(n.split()), -len(n)))
    chosen: List[str] = []
    for name in names:
        low = name.lower()
        if any(low in kept.lower() or kept.lower() in low for kept in chosen):
            continue
        chosen.append(name)
    return chosen


def _count_names(lines: Sequence[str], names: Sequence[str]) -> int:
    joined = " ".join(lines)
    return sum(len(re.findall(rf"(?<!\w){re.escape(name)}(?!\w)", joined, re.I))
               for name in names)


def _cap_at(text: str, start: int) -> bool:
    """True if a replacement at `start` begins a sentence (so it's capitalized)."""
    prefix = text[:start].rstrip()
    return start == 0 or (bool(prefix) and prefix[-1] in ".!?")


def _neutralize_protagonist_refs(text: str, names: Sequence[str]) -> str:
    """Swap protagonist NAMES/aliases AND generic handles ("our guy") for a
    neutral handle, capitalized at a sentence start. Agnostic: names come from
    the cast, the handle list is generic."""
    def _replacer(current: str):
        def _do(m: "re.Match[str]") -> str:
            return "The stranger" if _cap_at(current, m.start()) else "the stranger"
        return _do

    for name in names:
        pat = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.I)
        text = pat.sub(_replacer(text), text)
    text = _PROTAGONIST_HANDLE_RE.sub(_replacer(text), text)
    return text


def _has_protagonist_ref(text: str, names: Sequence[str]) -> bool:
    if _PROTAGONIST_HANDLE_RE.search(text):
        return True
    return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.I)
               for n in names)


def _subject_tokens(*srcs: Mapping[str, Any]) -> set:
    """Significant (len>3) subject words from understood/vision sources, for the
    conservative scene-break heuristic that closes an unresolved window."""
    toks: set = set()
    for src in srcs:
        if not isinstance(src, Mapping):
            continue
        subs = src.get("subjects")
        if isinstance(subs, (list, tuple)):
            for s in subs:
                toks.update(w for w in _WORD_RE.findall(str(s).lower())
                            if len(w) > 3)
    return toks


def _protagonist_desc_tokens(cast_obj: Mapping[str, Any]) -> set:
    """Significant (len>3) tokens of the protagonist's cast visual_description —
    the fingerprint used to recognize the ESTABLISHED protagonist on a panel so
    he is never neutralized to 'the stranger'. Empty when the cast carries no
    description (then the disambiguation falls back to the cue-only path)."""
    toks: set = set()
    for member in cast_obj.get("cast") or cast_obj.get("characters") or []:
        if not isinstance(member, dict):
            continue
        if not (member.get("is_protagonist")
                or str(member.get("role") or "").lower() == "protagonist"):
            continue
        desc = str(member.get("visual_description") or "")
        toks.update(w for w in _WORD_RE.findall(desc.lower()) if len(w) > 3)
    return toks


def _panel_understanding_tokens(*srcs: Mapping[str, Any]) -> set:
    """Significant (len>3) tokens from a panel's UNDERSTANDING (subjects +
    description/action/setting) — what the panel actually shows, used to match it
    against the protagonist's visual_description fingerprint."""
    toks: set = set()
    for src in srcs:
        if not isinstance(src, Mapping):
            continue
        for key in ("description", "action", "setting"):
            toks.update(w for w in _WORD_RE.findall(str(src.get(key) or "").lower())
                        if len(w) > 3)
        subs = src.get("subjects")
        if isinstance(subs, (list, tuple)):
            for s in subs:
                toks.update(w for w in _WORD_RE.findall(str(s).lower())
                            if len(w) > 3)
    return toks


def _concealment_blob(line: str, *srcs: Mapping[str, Any]) -> str:
    """Text scanned for a concealment/transformation cue: the panel line plus
    its understood/vision description, action, and subjects."""
    parts: List[str] = [str(line or "")]
    for src in srcs:
        if not isinstance(src, Mapping):
            continue
        parts.append(str(src.get("description") or ""))
        parts.append(str(src.get("action") or ""))
        subs = src.get("subjects")
        if isinstance(subs, (list, tuple)):
            parts.extend(str(s) for s in subs)
    return " ".join(p for p in parts if p)


def neutralize_identity_reveal_leaks(
    beats_obj: Dict[str, Any],
    cast_obj: Mapping[str, Any],
    vision_by_file: Mapping[str, Mapping[str, Any]],
    understood_by_file: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    """PER-PANEL, SUBJECT-AWARE neutralization: replace a protagonist NAME or
    generic HANDLE ("our guy") with "the stranger" ONLY on a panel that actually
    shows the still-unresolved concealed figure — NEVER as a blanket carry-across
    that nukes the established protagonist too.

    Agnostic by construction: concealment/power/gear cues and the familiar handles
    are generic; identity comes only from the cast (``cast_obj`` names +
    visual_description) and the per-panel understanding (``understood_by_file`` /
    ``vision_by_file`` subjects/description), never from hardcoded series words.

    A panel's protagonist reference is neutralized ONLY when ALL hold:
      * an unresolved-figure window is OPEN from an EARLIER panel (opened by a
        genuine concealment cue ``_CONCEALED_RE``; an OCR "who are you" question is
        an optional extra trigger). The window itself is just a gate.
      * THIS panel's own understanding refers to the UNRESOLVED figure — its line/
        subjects/description carry a concealment OR power/transformation/gear cue
        (``_POWER_GEAR_RE``): the later clear-view panel of the masked arrival.
      * THIS panel does NOT clearly match the ESTABLISHED protagonist — its
        understanding tokens do not overlap the protagonist's cast
        visual_description (>=2 shared significant tokens). When a panel plainly
        shows the established protagonist, he is NAMED, never neutralized.
    The window closes when the story's OWN text (OCR) names the protagonist, or
    after a conservative scene/topic break (``_UNRESOLVED_CLEAR_AFTER`` cue-free
    panels with a different subject focus).
    """
    names = _protagonist_names(cast_obj)
    protag_desc = _protagonist_desc_tokens(cast_obj)
    understood_by_file = understood_by_file or {}

    refs: List[Dict[str, Any]] = []
    for beat in beats_obj.get("beats") or []:
        for panel in beat.get("panel_narration") or []:
            refs.append({
                "beat": beat,
                "panel": panel,
                "scene_file": str(panel.get("scene_file") or ""),
            })

    changed = 0
    changed_beats: set = set()
    unresolved = False
    cue_subjects: set = set()
    panels_since_cue = 0

    for ref in refs:
        panel = ref["panel"]
        sf = ref["scene_file"]
        line = str(panel.get("line") or "")
        understood = understood_by_file.get(sf) or {}
        vis = vision_by_file.get(sf) or {}
        ocr = str(vis.get("ocr_clean") or vis.get("text") or "")
        blob = _concealment_blob(line, understood, vis)

        # RESOLUTION: the story's own text names the protagonist -> identity is
        # established; stop neutralizing (this panel included).
        if unresolved and names and any(
                re.search(rf"(?<!\w){re.escape(n)}(?!\w)", ocr, re.I)
                for n in names):
            unresolved = False
            panels_since_cue = 0
            cue_subjects = set()

        # NEUTRALIZE: only a carried window AND only when THIS panel's own
        # understanding refers to the unresolved figure (concealment/power/gear)
        # AND does not clearly match the established protagonist.
        if unresolved and _has_protagonist_ref(line, names):
            refers_to_unresolved = bool(_CONCEALED_RE.search(blob)
                                        or _POWER_GEAR_RE.search(blob))
            panel_tokens = _panel_understanding_tokens(understood, vis)
            matches_protagonist = (len(protag_desc) >= 2
                                   and len(protag_desc & panel_tokens) >= 2)
            if refers_to_unresolved and not matches_protagonist:
                rewritten = _neutralize_protagonist_refs(line, names)
                if rewritten != line:
                    panel["line"] = rewritten
                    changed += 1
                    changed_beats.add(id(ref["beat"]))

        # UPDATE STATE from THIS panel's cues (governs the panels that follow).
        has_cue = bool(_CONCEALED_RE.search(blob))
        if not has_cue and _IDENTITY_QUESTION_RE.search(ocr):
            has_cue = True  # optional extra trigger
        cur_subjects = _subject_tokens(understood, vis)
        if has_cue:
            unresolved = True
            panels_since_cue = 0
            cue_subjects |= cur_subjects
        elif unresolved:
            panels_since_cue += 1
            focus_changed = bool(cur_subjects) and bool(cue_subjects) \
                and not (cur_subjects & cue_subjects)
            if panels_since_cue >= _UNRESOLVED_CLEAR_AFTER and focus_changed:
                unresolved = False
                panels_since_cue = 0
                cue_subjects = set()

    if changed:
        for beat in beats_obj.get("beats") or []:
            if id(beat) in changed_beats:
                beat["narration"] = " ".join(
                    str(p.get("line") or "").strip()
                    for p in beat.get("panel_narration") or []
                    if str(p.get("line") or "").strip())
    return changed


def analyze_recap_style(
    script_obj: Mapping[str, Any],
    beats_obj: Mapping[str, Any],
    story_obj: Mapping[str, Any],
    cast_obj: Mapping[str, Any],
    vision_by_file: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return deterministic metrics plus WARN-worthy style/reveal issues."""
    lines = script_lines(script_obj)
    rows = panel_rows(beats_obj)
    panel_lines = [r["line"] for r in rows] or lines
    issues: List[Dict[str, str]] = []

    names = _protagonist_names(cast_obj)
    name_uses = _count_names(lines, names)
    name_allowance = max(2, math.ceil(max(1, len(lines)) / 20))
    if names and name_uses > name_allowance:
        issues.append({
            "code": "name_ration",
            "detail": (f"protagonist name used {name_uses} times across "
                       f"{len(lines)} lines (soft cap {name_allowance}); introduce "
                       "once, then prefer pronouns/casual stand-ins"),
        })

    sauce_count = sum(1 for line in lines
                      if _TEXTURE_RE.search(line) or _POINTER_RE.search(line))
    pointer_count = sum(1 for line in lines if _POINTER_RE.search(line))
    eligible_count = 0
    for beat in beats_obj.get("beats") or []:
        selection = {
            str(item.get("scene_file") or ""): item
            for item in beat.get("scene_selection") or []
            if isinstance(item, dict)
        }
        panels = beat.get("panel_narration") or []
        if not panels:
            eligible_count += 1
            continue
        for panel in panels:
            item = selection.get(str(panel.get("scene_file") or "")) or {}
            intensity = str(item.get("intensity") or "").lower()
            if intensity not in {"intense", "explosive"}:
                eligible_count += 1
    sauce_density = sauce_count / max(1, eligible_count)
    if len(lines) >= 12 and sauce_density < 0.18:
        issues.append({
            "code": "sauce_density",
            "detail": (f"only {sauce_count}/{eligible_count} eligible lines carry a detectable "
                       "persona/texture touch; target roughly one in four eligible "
                       "lines while keeping dramatic beats restrained"),
        })
    if len(lines) >= 12 and pointer_count == 0:
        issues.append({
            "code": "pointing_fits",
            "detail": ("no concise familiar comparison/reference detected; use "
                       "an earned genre-appropriate pointer where the art naturally "
                       "supports one, never as a new plot fact"),
        })

    visual_echo_count = sum(1 for line in lines if _VISUAL_ECHO_RE.search(line))
    visual_echo_density = visual_echo_count / max(1, len(lines))
    if len(lines) >= 12 and visual_echo_density > 0.20:
        issues.append({
            "code": "no_describe",
            "detail": (f"{visual_echo_count}/{len(lines)} lines match visible-only "
                       "description patterns; move the plot/stakes forward instead "
                       "of reading scenery or reactions back to the viewer"),
        })

    shot_desc_count = sum(1 for line in panel_lines if is_shot_description(line))
    if shot_desc_count:
        issues.append({
            "code": "shot_description",
            "detail": (f"{shot_desc_count} line(s) name the shot/camera/panel or "
                       "describe the artwork's rendering / a visual effect (e.g. 'A "
                       "close-up shot shows...', 'motion blur', '...is depicted') "
                       "instead of narrating the story; describe what HAPPENS and "
                       "its impact, never the picture or how it is drawn"),
        })

    word_counts = [len(_words(line)) for line in panel_lines if line]
    avg_words = (sum(word_counts) / len(word_counts)) if word_counts else 0.0
    long_count = sum(1 for n in word_counts if n > 22)

    fragment_count = sum(1 for line in panel_lines if is_spoken_fragment(line))
    if fragment_count:
        issues.append({
            "code": "spoken_fragment",
            "detail": (f"{fragment_count} panel line(s) are not independently "
                       "speakable complete clauses; flow between clips with "
                       "meaning, not dangling grammar"),
        })

    identity_leaks = 0
    for i, row in enumerate(rows):
        line = row["line"]
        if not names or not any(re.search(
                rf"(?<!\w){re.escape(name)}(?!\w)", line, re.I)
                for name in names):
            continue
        nearby_lines = " ".join(r["line"] for r in rows[max(0, i - 2):i + 1])
        if not _CONCEALED_RE.search(nearby_lines):
            continue
        future_ocr: List[str] = []
        for future in rows[i:min(len(rows), i + 7)]:
            vit = vision_by_file.get(future["scene_file"]) or {}
            future_ocr.append(str(vit.get("ocr_clean") or vit.get("text") or ""))
        if _IDENTITY_QUESTION_RE.search(" ".join(future_ocr)):
            identity_leaks += 1
            issues.append({
                "code": "identity_reveal_leak",
                "detail": (f"{row['scene_file'] or 'panel'} names the protagonist "
                           "while the nearby narration presents a concealed/new "
                           "arrival and the story has not resolved the identity; "
                           "use a neutral handle until the reveal"),
                "scene": row["scene_file"],
            })

    return {
        "metrics": {
            "protagonist_names": names,
            "protagonist_name_uses": name_uses,
            "protagonist_name_allowance": name_allowance,
            "sauce_lines": sauce_count,
            "sauce_eligible_lines": eligible_count,
            "sauce_density": round(sauce_density, 3),
            "pointing_lines": pointer_count,
            "visual_echo_lines": visual_echo_count,
            "visual_echo_density": round(visual_echo_density, 3),
            "panel_lines": len(word_counts),
            "average_words_per_panel_line": round(avg_words, 2),
            "overlong_panel_lines": long_count,
            "spoken_fragments": fragment_count,
            "shot_description_lines": shot_desc_count,
            "identity_reveal_leaks": identity_leaks,
        },
        "issues": issues,
    }
