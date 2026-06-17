#!/usr/bin/env python3
"""narration_safe_rules — the PRIMARY advertiser-safety layer.

The post-hoc regex sanitizer (narration_sanitize.py) is the NET. This module is
the source of truth for the *generator-side* rules: the same risk categories,
phrased as narration-writing constraints, injected into the narration prompt so
the script comes out clean BEFORE it ever reaches the sanitizer. Keep these two
in sync — when you add a category to narration_denylist.json, add its writing
rule here.

Usage (in the narration writer prompt):
    from narration_safe_rules import SAFE_NARRATION_RULES
    system = base_prompt + "\n\n" + SAFE_NARRATION_RULES
"""
from __future__ import annotations

SAFE_NARRATION_RULES = (
    "ADVERTISER-SAFE NARRATION (YouTube monetization — the transcript is what gets "
    "scanned, so write clean the FIRST time):\n"
    "- VIOLENCE: never say 'killed'/'murdered' plainly — use 'took out', 'didn't "
    "survive', 'was finished off', 'was eliminated'. Keep only the consequence of "
    "gore/torture, never the graphic method.\n"
    "- SEXUAL: keep everything implication-level. NEVER narrate sexual assault "
    "explicitly — imply it ('crossed a line he never should have', 'violated her "
    "trust') or skip the beat. No explicit anatomy. Soften setting terms "
    "(concubine->consort).\n"
    "- SELF-HARM / SUICIDE: never graphic, never name a method — 'ended her own "
    "life', 'couldn't go on', 'chose to leave this world'.\n"
    "- SUBSTANCES: say 'substances' / 'narcotics', not specific drug names.\n"
    "- PROFANITY: no slurs, ever. Soften strong profanity ('freaking', 'scoundrel', "
    "'jerk'). Paraphrase insults; do not voice them verbatim.\n"
    "- FIRST LINE: the cold open (first ~7 seconds) must be especially clean — no "
    "strong, violent, or sexual language in the opening sentence.\n"
    "Write the beat softened from the start; do not rely on a later pass to fix it."
)

# First-line stricter note the narrator can apply only to the opening beat.
SAFE_OPENING_NOTE = (
    "This is the FIRST line of the video (the cold open). Keep it especially "
    "advertiser-clean: no violence/sexual/strong language in this opening sentence."
)


if __name__ == "__main__":
    print(SAFE_NARRATION_RULES)
