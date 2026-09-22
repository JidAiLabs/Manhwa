#!/usr/bin/env python3
"""
thumbnail_styles.py — the proven recap-thumbnail style library + deterministic
style selection, distilled from 18 competitor references (assets/thumbnail_refs/).

The dominant formula is ONE composition with interchangeable modules:
  powered hero (aura) + reacting crowd + a big yellow LABEL-with-ARROW.
Each module = (a) a Nano Banana ART prompt (composition only — NO text; text is
a deterministic overlay) + (b) an overlay layout. The concept stage picks the
module from the beats (genre / intensity / bubble_mode), and the hook word is
rendered as the label.

Everything here is title-AGNOSTIC and pure (no model, no I/O) — unit-tested.
"""
from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Style modules. art_prompt is appended to a shared base in thumbnail_gen; it
# describes ONLY the composition (no text). overlay describes where the
# deterministic text layer goes.
# ---------------------------------------------------------------------------
STYLE_MODULES: Dict[str, Dict[str, Any]] = {
    "power_reveal": {
        "art_prompt": (
            "Center the hero mid-power with a vivid energy aura (electric blue "
            "by default); around/behind him place 2-4 onlookers reacting in "
            "shock — wide eyes, recoiling. Dramatic rim light, particle FX, "
            "high contrast."),
        "overlay": {"label_pos": "upper_right", "arrow": "to_hero",
                    "marks": ["!", "?"], "speech_slots": 1},
        "default": True,
    },
    "stat_callout": {
        "art_prompt": (
            "Hero with a glowing game-UI / status-window motif (floating panels "
            "or rune circles). Cool blue/cyan glow, sharp sci-fantasy lighting. "
            "Leave clear space top and side for big stat numbers."),
        "overlay": {"label_pos": "upper_right", "arrow": "to_hero",
                    "marks": ["!"], "speech_slots": 0, "stat_style": True},
    },
    "feat_object": {
        "art_prompt": (
            "Hero performing an impossible physical feat with a prominent OBJECT "
            "(huge weight, giant weapon/hammer). The object is large and clear "
            "in frame. Onlookers reacting in the background."),
        "overlay": {"label_pos": "on_object", "arrow": "to_object",
                    "marks": ["?"], "speech_slots": 1},
    },
    "humiliation": {
        "art_prompt": (
            "The hero standing or walking confidently while one or more "
            "opponents are fallen/kneeling/defeated around him. Contrast: hero "
            "calm and powered, opponents wrecked. Moody dramatic grade."),
        "overlay": {"label_pos": "upper_left", "arrow": "none",
                    "marks": [], "speech_slots": 2},
    },
    "vs_monster": {
        "art_prompt": (
            "The hero facing a massive monster / dragon / towering rival, with a "
            "clash of energy between them. Epic scale, the threat looming large, "
            "the hero small but defiant. Fiery/dark dramatic palette."),
        "overlay": {"label_pos": "lower_right", "arrow": "to_monster",
                    "marks": [], "speech_slots": 1},
    },
    "before_after": {
        "art_prompt": (
            "A split composition: LEFT the hero at his weakest (beaten, dim, "
            "cold blue grade, defeated posture); RIGHT the same hero transformed "
            "and powered up (bright aura, confident, warm/electric grade). Clear "
            "vertical divide."),
        "overlay": {"label_pos": "split", "arrow": "none",
                    "marks": [], "speech_slots": 2, "split": True},
    },
}

DEFAULT_STYLE = "power_reveal"

# The TONE of the series claim: which moment of the hook the scene, the labels
# and the headline go for. Owner, 2026-09-22: "i prefer absurd and a little
# erotique". Counted in the owner's 20 examples: the three biggest outliers are
# absurd (EXP GLITCH over 100x, MC (LOSER)/FIANCE over 100x, MARRY HER 47x);
# the three whose only hook is the body are the batch's weakest (1.1x, 1.1x,
# 2 views/hour). So absurd is the default and erotic is a per-series switch
# that must fall back when the story has nothing to show. Every tone forbids
# inventing: the click must be a promise the video can pay off.
CLAIM_TONES: Dict[str, str] = {
    "absurd": (
        "TONE: ABSURD. Choose the moment of THE HOOK that is the most WRONG at "
        "first sight -- a viewer stares for a second because it makes no sense: "
        "someone calm while everyone screams, a system window ordering "
        "something ridiculous, a mundane act while the world ends. The scene, "
        "the labels and the headline share that deadpan wrongness. Never invent "
        "it: only what the story actually shows."),
    "erotic": (
        "TONE: SUGGESTIVE. Choose the moment of THE HOOK with the most heat -- a "
        "charged pose, gaze or touch the story actually draws -- suggestive, "
        "never explicit, and never invented. If the opening has no such moment, "
        "go ABSURD instead: the most wrong-at-first-sight moment of the hook."),
    "dramatic": (
        "TONE: DRAMATIC. Choose the moment of THE HOOK with the highest stakes "
        "-- the reveal, the transformation, the threat. Never invent it: only "
        "what the story actually shows."),
}
DEFAULT_TONE = "absurd"

# HOOK DESIGNS: the LABEL layer on a one-scene thumbnail, counted from the
# owner's examples (20 on 2026-09-21: nametag+arrow 3, contrast pair 4,
# nametag+headline 4, system window 4, roll-call 2, split 1, no text 2).
# A design is ranked from the TEASER by code (publish_concept.rank_designs),
# never chosen by the model. Only designs whose cast is the LEAD alone are
# here: contrast_pair and roll_call need a counterpart ref no code can find yet.
# ponytail: add each one when its counterpart ref finder exists.
# The arrow lands on its subject BY CONSTRUCTION: art_clause fixes where the
# lead stands and overlay.arrow_to aims there (face detection misses anime
# faces). No grammar carries a sample label: an example is an answer.
_NAMETAG_GRAMMAR = (
    "A label is a NAMETAG, not a caption: it names WHAT THE LEAD IS or WHAT "
    "THEY BECAME -- a role, a title, a rank, a status -- so a viewer could "
    "draw an arrow from it to the person. 1-3 words, this story's OWN words, "
    "at an EXTREME of a ladder a viewer reads instantly. Never a mood, never "
    "a sentence. A number or rank must be one the story states. ")

HOOK_DESIGNS: Dict[str, Dict[str, Any]] = {
    "nametag": {
        "label_grammar": _NAMETAG_GRAMMAR + "Write 5 candidate nametags.",
        "art_clause": ("Place the hero's face right of centre, about 60% "
                       "across and 40% down. Keep the upper-left third calm "
                       "and uncluttered."),
        "overlay": {"label_pos": "upper_left", "arrow": "to_hero",
                    "arrow_to": [0.60, 0.40], "marks": [], "speech_slots": 0},
    },
    "nametag_headline": {
        "label_grammar": (_NAMETAG_GRAMMAR + "Write 5 candidate nametags, and "
                          "5 candidate HEADLINES: 2-3 words that land the "
                          "promise of the hook, or the lead's change written "
                          "as LOW -> HIGH. Only facts of this story."),
        # "keep the bottom fifth calm and uncluttered" painted an EMPTY BAR
        # under the scene (owner's first card, 2026-09-22)
        "art_clause": ("Place the hero's face right of centre, about 60% "
                       "across and 38% down. Keep the upper-left third calm. "
                       "The scene continues to the bottom edge; keep the "
                       "lower-left area darker and free of faces."),
        "overlay": {"label_pos": "upper_left", "arrow": "to_hero",
                    "arrow_to": [0.60, 0.38], "marks": [], "speech_slots": 0,
                    "headline_pos": "lower_left"},
    },
    "system_window": {
        "label_grammar": (_NAMETAG_GRAMMAR + "Write 5 candidate nametags. The "
                          "system window's words are QUOTED from the story "
                          "by code; do not write them."),
        # The label and the face may never share a corner. The first build had
        # label_pos=upper_right with the face at 68%/42%: the corner label's
        # column covered it BY CONSTRUCTION (tests pin the geometry now). Label
        # upper-left, the window below it, the lead alone on the right.
        "art_clause": ("Place the hero on the right half, face about 70% "
                       "across and 40% down, nothing in front of the face. "
                       "Leave the LEFT 45% of the frame dark and uncluttered: "
                       "no figures, no faces, no panels, no glyphs there."),
        "overlay": {"label_pos": "upper_left", "arrow": "to_hero",
                    "arrow_to": [0.70, 0.40], "marks": [], "speech_slots": 0,
                    "card": {"pos": [0.04, 0.52], "size": [0.42, 0.30]}},
    },
}


def _genre_key(genre: str) -> str:
    g = (genre or "").lower()
    if any(k in g for k in ("system", "regress", "reincarnat", "rebirth", "game")):
        return "system"
    if any(k in g for k in ("murim", "wuxia", "martial", "cultivat")):
        return "murim"
    if any(k in g for k in ("modern", "apocalypse", "hunter", "tower", "dungeon")):
        return "modern"
    return "generic"


_INTENSITY = {"calm": 0, "unknown": 0, "tense": 1, "intense": 2, "explosive": 3}


def beat_signals(beats_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate cheap signals the style picker reads from the beats."""
    beats = beats_obj.get("beats") or []
    max_i = 0
    bubble_modes: List[str] = []
    text_blob: List[str] = []
    for b in beats:
        for s in b.get("scene_selection") or []:
            if isinstance(s, dict):
                max_i = max(max_i, _INTENSITY.get(str(s.get("intensity") or "").lower(), 0))
                if s.get("bubble_mode"):
                    bubble_modes.append(str(s.get("bubble_mode")).lower())
        for k in ("hook", "what_happens", "beat_title"):
            if b.get(k):
                text_blob.append(str(b[k]).lower())
    blob = " ".join(text_blob)
    return {"max_intensity": max_i, "bubble_modes": bubble_modes, "text": blob}


def select_style(beats_obj: Dict[str, Any], *, genre: str = "") -> str:
    """Deterministically pick a thumbnail style module from the beats content.

    Priority is by how distinctive the signal is; falls back to the default
    power-reveal flex. Title-agnostic — keys on story signals, any manhwa.
    """
    sig = beat_signals(beats_obj)
    blob, gk = sig["text"], _genre_key(genre)
    has = lambda *ws: any(w in blob for w in ws)

    # 1. system/stat UI present → stat callout
    if gk == "system" or "system" in sig["bubble_modes"] or has(
            "level", "stat", "rank ", " rank", "status window", "skill",
            "system", "quest", "exp", "tier"):
        return "stat_callout"
    # 2. a giant foe / tower boss → vs monster
    if has("monster", "dragon", "beast", "demon king", "boss", "god ",
           "titan", "leviathan"):
        return "vs_monster"
    # 3. an impossible physical feat with an object
    if has("weight", "lift", "kg", "hammer", "barbell", "sword too",
           "carries", "one hand"):
        return "feat_object"
    # 4. explicit transformation arc → before/after
    if has("transform", "weakest", "used to be", "from zero", "grew stronger",
           "leveled up", "trained", "100x", "reborn"):
        return "before_after"
    # 5. domination/humiliation beat
    if has("humiliate", "mock", "defeat", "kneel", "crush", "look down",
           "underestimat", "expel"):
        return "humiliation"
    # 6. high-intensity power reveal (default flex)
    return DEFAULT_STYLE


def style_for(name: str) -> Dict[str, Any]:
    return STYLE_MODULES.get(name, STYLE_MODULES[DEFAULT_STYLE])
