#!/usr/bin/env python3
"""narration_heal.py — turn prep-QA ERROR flags into a per-group corrections map
for gemini_narrative_pass --corrections, so the auto-heal regenerates ONLY the
failing groups (from their panels) and leaves every good line untouched.

The point of auto-heal: never DROP a line to empty/silent to satisfy QA — re-
narrate that one group from the art until QA is green.

CLI: narration_heal.py --qa <prep_qa.json> --out <corrections.json>
  exit 0 + prints "groups=N" (N may be 0 = nothing heal-able -> caller stops).
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List

# ERROR codes a targeted per-group regeneration can plausibly fix (re-narrate
# the group from its panels). Codes about cropping/montage/audio are NOT here —
# they aren't narration problems.
#
# `narration_stale` is DELIBERATELY EXCLUDED: it means the plan/script text has
# DIVERGED from the beats narration ("script.json predates manifest.beats.json")
# — only RE-SCRIPTING + re-planning can clear it. Re-narration changes the beats
# AGAIN, so the script stays behind and the flag never clears → the heal loop
# burns to its cap (the 2.6h-chapter non-convergence). The worker handles
# staleness by re-running the scripted stage, not by re-narrating.
HEALABLE = {
    "caption_unvoiced", "chrome_narration", "fragment_dangle",
    "filler_narration", "beats_incomplete",
    "empty_item", "silent_group", "grounding_weak",
    "shot_description", "filename_in_narration", "impact_marker_leak",
    "figures_leak",
    # a voiced line opens with a bare (unbracketed) mood/tone word before its
    # real sentence ("Dramatic: He's tumbling…") — pipeline/authoring
    # vocabulary read aloud, the SAME leak channel as figures_leak/
    # impact_marker_leak; the re-roll simply never writes the label.
    "mood_tag_leak",
    # detector-verified impact SFX on a span panel but no impact wording in
    # the line — re-narration IS the fix: the regenerated group's writer
    # payload carries the [IMPACT SFX on panel] marker, so the re-roll sees
    # the very signal the original miss lacked. Heal-THEN-block: this code is
    # also in the worker's _CRITICAL_QA_CODES, so blocking only survives when
    # healing can't clear it.
    "impact_mismatch",
    # a line fits a +-1-shifted panel window better than its own span (the
    # one-panel lead/lag class from the 2026-07-06 vision review) — a group
    # re-roll re-derives spans (unpinned) or re-writes each line against its
    # pinned span; either way the re-roll goes through span_align_pass /
    # per-span writing, which is the fix. NOT in the worker blocking set yet
    # (first run measures precision).
    "narration_offset",
    # a voiced line stops mid-sentence ("...no mercy to be found, only the")
    # — a writer-truncated final sentence; a re-roll writes the full thought.
    "truncated_line",
    # a line's actor-noun contradicts the span's cast-resolved figures ("the
    # assassin draws his steel" over Cheon's counter-draw) — the re-roll's
    # writer payload carries the per-panel `figures` ground truth the
    # original roll lacked. NOT worker-blocking yet (precision is measured
    # on the first production run).
    "actor_mismatch",
    # a line PLURALIZES its actor ("our guy and his assassins") over a span
    # whose every panel shows ONE person — the invented-companions class from
    # the 2026-07-16 audit. Same posture as actor_mismatch: heal-target,
    # NOT worker-blocking until precision is measured.
    "actor_count_mismatch",
    # a segment line past its span's word budget (escaped the writer
    # validator via its fallback path) — a re-roll with the explicit word
    # cap in the note converges: length is fully in the writer's control.
    "line_overlong",
}

_GID_RE = re.compile(r"g0*(\d+)")
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")


def _gid(segment_id: str) -> int | None:
    m = _GID_RE.match(str(segment_id or ""))
    return int(m.group(1)) if m else None


def _note_for(code: str, detail: str) -> str:
    if code == "caption_unvoiced":
        q = _QUOTED_RE.findall(detail or "")
        cap = q[-1] if q else ""
        return ("The narration SKIPPED an on-panel caption. Weave its words into "
                f"the narration (this is mandatory): \"{cap}\".")
    if code == "chrome_narration":
        return ("Narrate the panel's content as STORY, not as an interface: turn "
                "any on-screen numbers into prose (e.g. 'over 3,000 episodes and "
                "almost no readers'). NEVER use interface words — 'view count', "
                "'comments', 'tap', 'swipe', 'next episode', 'displays statistics', "
                "'the screen/chapter shows'.")
    if code == "fragment_dangle":
        return "The narration is a dangling fragment — make it a complete sentence."
    if code == "truncated_line":
        return ("This line STOPS MID-SENTENCE — the thought never ends. "
                "Re-narrate it as a complete sentence that finishes the "
                "thought and ends with terminal punctuation.")
    if code == "shot_description":
        return ("This line describes the artwork or a visual effect (motion blur, "
                "speed lines, 'is depicted') instead of the story — re-narrate the "
                "ACTION and its impact dramatically; never name an effect, blur, "
                "the panel, image, or camera.")
    if code == "filename_in_narration":
        return ("This line reads an image FILE NAME aloud (e.g. 'p000032.jpg') — "
                "that is pipeline bookkeeping, never story. Re-narrate what "
                "actually HAPPENS across these panels: who does what, the force, "
                "the outcome. Never mention a file, an image name, or 'the series "
                "of images'.")
    if code == "impact_marker_leak":
        return ("This line reads the impact-SFX bracket marker aloud (e.g. "
                "'[IMPACT SFX on panel]') — that is pipeline bookkeeping, "
                "never story. Re-narrate the actual strike/stab/blow: who "
                "hits, what lands, the force and the damage. Never mention "
                "a bracket, a tag, or the marker itself.")
    if code == "figures_leak":
        return ("This line reads the unresolved-figure payload wrapper "
                "aloud (e.g. 'unknown (a masked figure...)') — that is "
                "pipeline bookkeeping, never story. Re-narrate using "
                "neutral phrasing for the figure (the masked figure, the "
                "man in the hood); a RESOLVED cast name is fine to say, "
                "just never the raw evidence text or the word 'unknown' "
                "followed by a parenthesis.")
    if code == "mood_tag_leak":
        return ("This line opens with a bare mood/tone word read aloud "
                "(e.g. 'Dramatic: He's tumbling…', 'Comic: The masked "
                "guy…') — that is pipeline bookkeeping, never story. A mood "
                "tag is ALWAYS bracketed ([dramatic]) and added by the "
                "pipeline automatically; never type a mood/tone label into "
                "the narration itself. Re-narrate starting directly with "
                "the real sentence.")
    if code == "impact_mismatch":
        return ("Painted IMPACT-SFX lettering is on this panel — a strike, "
                "stab, blow, or crash is landing HERE, and the current line "
                "misses it entirely. Re-narrate the physical impact "
                "explicitly: who strikes, what lands, the force and the "
                "damage. Never describe this panel as calm or uneventful.")
    if code == "actor_mismatch":
        issue = (detail or "").split(":", 1)[0].strip()
        return ("This line attributes the action/thought to the WRONG "
                "character" + (f" ({issue})" if issue else "") + ". "
                "Each panel's `figures` list is ground truth — re-narrate "
                "naming the actor ONLY from the panel's figures; when a "
                "figure is unknown, use neutral phrasing (the masked "
                "figure), never a guessed name.")
    if code == "actor_count_mismatch":
        return ("This line PLURALIZES the actor but every panel in its span "
                "shows ONE figure — re-narrate with the single actor the "
                "panel's figures name; never invent companions, groups, or "
                "'and his …' phrasing the art does not show.")
    if code == "line_overlong":
        cap = ""
        m = re.search(r"/ (\d+) words", detail or "")
        if m:
            cap = f" at most {m.group(1)} words —"
        return ("This line runs far past its panels' watchable screen time. "
                f"Re-narrate the SAME moment in{cap} one or two punchy "
                "sentences: keep the concrete facts and any caption words, "
                "cut everything else.")
    if code == "phrase_echo":
        return ("This line repeats an earlier line's phrase nearly "
                "verbatim — re-narrate the same moment with FRESH wording "
                "(new verbs and imagery, same facts); never re-use a "
                "sentence you already spoke.")
    if code == "narration_offset":
        return ("This line describes the NEIGHBORING panel's moment, not its "
                "own — a one-panel lead/lag (e.g. narrating the impact while "
                "the pre-impact panel is on screen). Re-narrate the group so "
                "every line describes exactly the panel(s) it is voiced over: "
                "the strike lands when the strike panel shows, the reaction "
                "when the reaction panel shows.")
    if code in ("beats_incomplete", "empty_item", "silent_group"):
        return ("The narration is empty — describe what actually happens in this "
                "panel (and cover any on-panel caption).")
    if code == "grounding_weak":
        issue = (detail or "").split(":", 1)[-1].strip()
        return ("The narration is weak or mis-grounded"
                + (f" ({issue})" if issue else "")
                + ". Re-narrate this group to name EXACTLY what the panel shows: "
                "fix any mis-named or invented subject (beasts are 'beasts', not "
                "'dogs'; do not invent quantities or a crowd) and replace vague "
                "filler with a concrete, vivid line.")
    return "Rewrite the narration to match exactly what is shown in the panel."


def corrections_from_qa(report: Dict[str, Any], *,
                        include_grounding_warn: bool = False) -> Dict[int, str]:
    """{group_id: combined correction note} from the ERROR flags QA can heal."""
    notes: Dict[int, List[str]] = {}
    for f in report.get("flags") or []:
        code = f.get("code")
        # A chrome/meta leak is a rule violation at ANY severity (the channel
        # never voices interface chatter). Grounding WARNs are report-only by
        # default; opt in when running the slower semantic-heal experiment.
        if code == "chrome_narration":
            pass
        elif code == "phrase_echo":
            pass   # WARN by design (heal-target): a near-verbatim repeated
            #        phrase is a wording fix, never worth blocking a chapter
        elif code == "grounding_weak" and include_grounding_warn:
            pass   # a rule/quality violation worth healing at ANY severity
        elif f.get("severity") == "ERROR" and code in HEALABLE:
            pass
        else:
            continue
        gid = _gid(f.get("segment_id"))
        if gid is None:
            continue
        note = _note_for(str(f.get("code")), str(f.get("detail") or ""))
        notes.setdefault(gid, [])
        if note not in notes[gid]:
            notes[gid].append(note)
    return {gid: " ".join(ns) for gid, ns in notes.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="prep_qa.json")
    ap.add_argument("--out", required=True, help="corrections.json to write")
    ap.add_argument("--include-grounding-warn", action="store_true",
                    help="treat WARN-level grounding_weak flags as healable; "
                         "default keeps them in QA but does not regenerate")
    args = ap.parse_args()
    try:
        report = json.load(open(args.qa))
    except Exception:
        report = {}
    corr = corrections_from_qa(
        report, include_grounding_warn=args.include_grounding_warn)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in corr.items()}, f,
                  ensure_ascii=False, indent=2)
    print(f"groups={len(corr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
