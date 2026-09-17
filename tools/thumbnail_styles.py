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
