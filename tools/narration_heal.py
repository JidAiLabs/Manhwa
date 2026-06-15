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
    "empty_item", "silent_group",
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
        return ("Do NOT mention view counts, episode-count statistics as UI, "
                "'the chapter/series displays', screenshots, or the recap format "
                "itself — narrate only the events and dialogue shown in the panel.")
    if code == "fragment_dangle":
        return "The narration is a dangling fragment — make it a complete sentence."
    if code in ("beats_incomplete", "empty_item", "silent_group"):
        return ("The narration is empty — describe what actually happens in this "
                "panel (and cover any on-panel caption).")
    return "Rewrite the narration to match exactly what is shown in the panel."


def corrections_from_qa(report: Dict[str, Any]) -> Dict[int, str]:
    """{group_id: combined correction note} from the ERROR flags QA can heal."""
    notes: Dict[int, List[str]] = {}
    for f in report.get("flags") or []:
        if f.get("severity") != "ERROR" or f.get("code") not in HEALABLE:
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
    args = ap.parse_args()
    try:
        report = json.load(open(args.qa))
    except Exception:
        report = {}
    corr = corrections_from_qa(report)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in corr.items()}, f,
                  ensure_ascii=False, indent=2)
    print(f"groups={len(corr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
