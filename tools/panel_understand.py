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
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from gemini_narrative_pass import (                                   # noqa: E402
    load_json, dump_json, _call_model_with_backoff)

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
    },
    "required": ["description", "action", "intensity", "panel_kind"],
}

SYSTEM = (
    "You are a manhwa recap analyst. You see ONE webtoon panel image plus its "
    "OCR text. Describe what is LITERALLY happening in this panel — specific and "
    "vivid, but strictly faithful to what is shown (never invent characters or "
    "events). Return JSON:\n"
    "  description: 1-2 concrete sentences of the action/scene in this panel.\n"
    "  subjects: the characters / creatures / key objects visible.\n"
    "  action: the single key event or beat of this panel.\n"
    "  dialogue: any spoken line or caption, copied VERBATIM from the OCR; '' if "
    "none. Do not paraphrase dialogue.\n"
    "  setting: where/what the scene is (a train, a city street, a flashback "
    "screen, etc.).\n"
    "  intensity: calm | tense | intense | explosive.\n"
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
    "e.g. a black card 'BACK THEN, I HAD NO IDEA.'), OR a lone speech / shout / "
    "dialogue bubble (or any text) floating on a PLAIN / BLANK / WHITE / EMPTY "
    "background with NO drawn scene, character, or object behind it (e.g. 'a single "
    "white speech bubble against a plain white background'). Its words go in "
    "'dialogue'; it is not a picture. A panel with REAL ART (a character, a place, "
    "an object) AND a bubble/caption is 'story', not 'caption'.\n"
    "    'system' = an IN-WORLD GAME / SYSTEM INTERFACE the CHARACTER perceives — "
    "a QUEST window, a STATUS / STAT / SKILL screen, a NOTIFICATION / ALARM / level-up "
    "toast, or a SYSTEM MESSAGE (e.g. 'QUEST DIRECTIONS', 'STATUS', 'NOTIFICATION — You "
    "have defeated a [Steel-Fanged Lycan]', '7TH GENERATION NANO MACHINE, STARTING "
    "ACTIVATION'). It can be ANY length, ANY case, ANY color/art style, and may be drawn "
    "OVER character art. These are PLOT and MUST be kept and shown.\n"
    "    'story' = the STORY WORLD — real scene art AND in-world device screens a "
    "character uses in-story (a reader app, chat, feed), a place/organization name card. "
    "A panel with real character art is 'story' even if a system window is drawn over it. "
    "When unsure between system/story (both are always kept), pick either; only an AUTHOR "
    "narrative caption is 'caption' and only platform furniture is 'chrome'.\n"
    "The 'previous_panels' field is context for continuity only — describe THIS "
    "panel, not the previous ones."
)


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


def build_payload(panel: Dict[str, Any], prev_descs: List[str]) -> Dict[str, Any]:
    """Pure: the per-panel model input (OCR + cheap vision signals + rolling
    context for continuity). Image is attached separately by the caller."""
    v = panel.get("vision") or {}
    labels = [x.get("desc") for x in (v.get("labels") or []) if x.get("desc")]
    objects = [x.get("name") for x in (v.get("objects") or []) if x.get("name")]
    return {
        "scene_file": panel.get("scene_file"),
        "ocr": (panel.get("ocr_clean") or "")[:900],
        "labels": labels[:12],
        "objects": objects[:12],
        "previous_panels": [d for d in prev_descs[-2:] if d],
    }


def assemble_record(scene_file: str, parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure: normalize one model result into a panel record. A parse failure is
    recorded (never silently dropped) so resume can re-run just that panel."""
    if not isinstance(parsed, dict):
        # parse failure: no understanding -> treat as 'empty' so it is filtered
        # out of grouping (a panel we can't understand must not be narrated);
        # --resume still re-attempts it because error is recorded.
        return {"scene_file": scene_file, "description": "", "subjects": [],
                "action": "", "dialogue": "", "setting": "",
                "intensity": "unknown", "panel_kind": "empty",
                "error": "parse_failed"}
    inten = str(parsed.get("intensity") or "").lower()
    description = str(parsed.get("description") or "").strip()
    subjects = [str(s) for s in (parsed.get("subjects") or []) if s]
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
        "dialogue": str(parsed.get("dialogue") or "").strip(),
        "setting": str(parsed.get("setting") or "").strip(),
        "intensity": inten if inten in
        ("calm", "tense", "intense", "explosive") else "unknown",
        "panel_kind": kind,
    }


def understand_panels(items: List[Dict[str, Any]], call_fn: Callable[..., Any],
                      *, log: Callable[[str], None] = lambda _m: None,
                      prior: Optional[Dict[str, Dict[str, Any]]] = None,
                      concurrency: int = 1) -> List[Dict[str, Any]]:
    """Describe each panel in order, threading rolling context (the last 2
    panels). `call_fn(payload, image_path) -> parsed dict|None` is injected.
    `prior` (scene_file -> good record) lets --resume skip done panels.

    concurrency>1 runs panels in BATCHES of that size: every panel in a batch
    shares the SAME context (the descriptions taken BEFORE the batch), so order
    and continuity are preserved — only batch-mates can't see each other, which
    is negligible since the window is just 2 panels. The GPU then processes the
    batch at once (needs ollama OLLAMA_NUM_PARALLEL>=concurrency to parallelize)."""
    from concurrent.futures import ThreadPoolExecutor
    prior = prior or {}
    conc = max(1, int(concurrency))
    out: List[Dict[str, Any]] = []
    prev_descs: List[str] = []

    def _understand(it: Dict[str, Any], ctx: List[str]) -> Dict[str, Any]:
        return assemble_record(
            it.get("scene_file"),
            call_fn(build_payload(it, ctx), it.get("scene_path")))

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
        if done and done.get("description") and not done.get("error"):
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
                p.get("description"), p.get("action"), p.get("dialogue"))]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="ollama")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-output-tokens", type=int, default=400)
    ap.add_argument("--resume", action="store_true",
                    help="keep good panel records in --out, redo only failures")
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("STUDIO_UNDERSTAND_CONCURRENCY", "3")),
                    help="panels understood per batch (needs ollama "
                         "OLLAMA_NUM_PARALLEL>=this to actually parallelize)")
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
    dump_json(args.out, {
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "model": model, "count": len(panels), "panels": panels})

    # Centralize the chrome/story verdict: stamp panel_kind back onto the vision
    # manifest so the SINGLE chrome chokepoint (scene_chrome.is_chrome_scene —
    # used by story_group, render_prep AND prep_qa) defers to the understanding
    # everywhere. No downstream module re-derives chrome from OCR and disagrees.
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
        dump_json(args.vision_manifest, vision)
        print(f"[ok] stamped panel_kind + subjects onto {os.path.basename(args.vision_manifest)}")

    ok = sum(1 for p in panels if p.get("description") and not p.get("error"))
    print(f"[ok] wrote={args.out} panels={len(panels)} understood={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
