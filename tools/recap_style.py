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
detail changes the plot. Each line must advance action, cause, stakes, thought,
revelation, or consequence. A quick panel may continue the surrounding thought,
but every panel line must remain an independently speakable complete clause.
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
6. REVEAL PACING: cast matching is not permission to spoil an identity. When a
new, transformed, masked, hooded, glowing, silhouetted, disguised, or otherwise
unfamiliar arrival is treated as unknown by the story, call them "the stranger",
"the intruder", or another grounded neutral handle. Use a cast name only after
the current or earlier story text has actually established that identity."""


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
    r"\b(?:silhouette|stranger|intruder|unknown|mysterious|masked|hooded|"
    r"disguised|transformed|unfamiliar|shadowy figure|glowing figure)\b",
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


def neutralize_identity_reveal_leaks(
    beats_obj: Dict[str, Any],
    cast_obj: Mapping[str, Any],
    vision_by_file: Mapping[str, Mapping[str, Any]],
) -> int:
    """Replace a premature protagonist name with a neutral handle in-place.

    The trigger is deliberately narrow: the nearby narration must frame the
    arrival as concealed/new, and either the same line does so or upcoming OCR
    explicitly questions the identity. This fixes reveal pacing without any
    series-specific names or chapter corrections.
    """
    names = _protagonist_names(cast_obj)
    if not names:
        return 0

    refs: List[Dict[str, Any]] = []
    for beat in beats_obj.get("beats") or []:
        for panel in beat.get("panel_narration") or []:
            refs.append({
                "beat": beat,
                "panel": panel,
                "scene_file": str(panel.get("scene_file") or ""),
            })

    changed = 0
    changed_beats: set[int] = set()
    for i, ref in enumerate(refs):
        panel = ref["panel"]
        line = str(panel.get("line") or "")
        hit_names = [name for name in names if re.search(
            rf"(?<!\w){re.escape(name)}(?!\w)", line, re.I)]
        if not hit_names:
            continue
        nearby = " ".join(str(r["panel"].get("line") or "")
                          for r in refs[max(0, i - 2):i + 1])
        if not _CONCEALED_RE.search(nearby):
            continue
        future_ocr = " ".join(
            str((vision_by_file.get(r["scene_file"]) or {}).get("ocr_clean")
                or (vision_by_file.get(r["scene_file"]) or {}).get("text") or "")
            for r in refs[i:min(len(refs), i + 7)])
        if not _IDENTITY_QUESTION_RE.search(future_ocr):
            continue
        rewritten = line
        for name in hit_names:
            rewritten = re.sub(
                rf"(?<!\w){re.escape(name)}(?!\w)",
                lambda m: "The stranger" if m.start() == 0 else "the stranger",
                rewritten, flags=re.I)
        if rewritten != line:
            panel["line"] = rewritten
            changed += 1
            changed_beats.add(id(ref["beat"]))

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
            "identity_reveal_leaks": identity_leaks,
        },
        "issues": issues,
    }
