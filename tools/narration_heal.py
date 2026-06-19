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
HEALABLE = {
    "caption_unvoiced", "chrome_narration", "fragment_dangle",
    "filler_narration", "beats_incomplete", "narration_stale",
    "empty_item", "silent_group", "grounding_weak",
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
