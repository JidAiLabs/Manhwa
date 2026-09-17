#!/usr/bin/env python3
"""
panel_identity.py — WHO is in a panel, decided from the IMAGE.

`cast_identity` joins two TEXTS: gemma's description of the panel ("a young man
with short dark hair") and a text description of each character. In a cast where
several characters are young dark-haired men that join cannot separate them, and
it doesn't: ORV Episode 6 narrated the protagonist as "Namwoon Kim" 67 times
(the chapter's looks were swapped), and 5 of 8 "clear shots of the MC" thumbnail
suggestions were other men.

This module asks the local multimodal model to LOOK, against exemplar panels the
owner confirmed. Measured on 50 Ep6 panels + 8 cross-chapter tiles (2026-09-17):

  candidates/call   tiles (3 real)     Ep6 precision/recall   wrong names
  word matching     3 of 8 suggested   0.69 / 0.68            22
  1 (A or OTHER)    2 found, 3 FALSE   0.70 / 0.51             8
  2 (A/B/OTHER)     3 found, 0 false   0.96 / 0.65             6
  4 (A/B/C/D/OTHER) 0 found            0.75 / 0.24             3 + 12 claims
                                                                about characters
                                                                absent from every
                                                                panel

So TWO candidates is a hard cap, not a default: with one, look-alikes are forced
onto the lead; with four, it recognises nobody and invents the extras. Recall
~0.65 is the accepted cost — an unconfirmed figure keeps a neutral handle ("the
white-haired guy"), which is the owner's rule: never mix names, never drop a
panel.

Whole panels, no crops: neither YOLO model we have can find manhwa characters
(the legacy `character` class fired 2 times in 60 panels at conf 0.01, COCO
person 5 of 60).

Design: docs/plans/2026-09-18-image-based-character-identity.md
"""
from __future__ import annotations

import io
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)

MAX_CANDIDATES = 2          # measured ceiling, see the table above
MODEL = os.environ.get("STUDIO_IDENTITY_MODEL", "gemma4:26b")
# panels these kinds describe carry no drawn person to identify
_SKIP_KINDS = frozenset({"system", "caption", "empty"})
_LETTERS = "AB"


def _members(registry: Any) -> List[Dict[str, Any]]:
    if isinstance(registry, dict):
        registry = registry.get("cast") or registry.get("members") or []
    return [m for m in (registry or []) if isinstance(m, dict)]


def candidates(registry: Any, max_n: int = MAX_CANDIDATES) -> List[Dict[str, Any]]:
    """The <=2 registry members that carry exemplars, protagonist first.

    Capped hard: a third candidate measurably destroys recall and invents
    appearances of the characters it was just shown.
    """
    out = []
    for m in _members(registry):
        ex = [str(p) for p in (m.get("exemplars") or []) if str(p).strip()]
        if not ex:
            continue
        out.append({"name": str(m.get("canonical_name") or m.get("id") or "").strip(),
                    "exemplars": ex,
                    "_prot": bool(m.get("is_protagonist"))})
    out.sort(key=lambda m: 0 if m["_prot"] else 1)
    return [{k: v for k, v in m.items() if k != "_prot"} for m in out[:max_n]]


def build_prompt(names: List[str]) -> str:
    """The measured forced-choice question.

    Three things are load-bearing and were each measured: an explicit OTHER (a
    yes/no question makes the model say yes to any similar-looking man), the
    FACE instruction (outfit and hair colour are exactly what the old text
    matching already got wrong), and naming the letters only — a character NAME
    in the prompt is a hint to pattern-match on instead of looking.
    """
    letters = _LETTERS[:len(names)]
    shown = " ".join(
        "Images %d and %d show CHARACTER %s." % (2 * i + 1, 2 * i + 2, L)
        for i, L in enumerate(letters))
    n = 2 * len(names) + 1
    answers = ", ".join('"%s"' % L for L in letters) + ' or "OTHER"'
    return (
        "%s They are different characters. Image %d is a panel from the same "
        "manhwa, which has MANY other characters who look similar: dark hair, "
        "suits, young men. For EACH person visible in image %d, answer %s. "
        "Answer a letter only when the FACE clearly matches (eyes, face shape, "
        "hairstyle); a similar outfit or hair colour alone is OTHER. "
        'Return ONLY JSON: {"people": [%s, ...]}'
        % (shown, n, n, answers.replace('"', ""), answers))


def parse_reply(raw: Any, names: List[str]) -> Dict[str, Any]:
    """{"names": [confirmed], "others": count}. Anything unusable confirms
    NOBODY — a guess is the failure this module exists to remove."""
    from ollama_compat import first_json
    got = first_json(str(raw or "")) or {}
    people = got.get("people")
    if not isinstance(people, list):
        return {"names": [], "others": 0}
    found: List[str] = []
    others = 0
    for p in people:
        tok = str(p or "").strip().upper()
        i = _LETTERS.find(tok) if len(tok) == 1 else -1
        if 0 <= i < len(names):
            if names[i] not in found:
                found.append(names[i])
        elif tok == "OTHER":
            others += 1
    return {"names": found, "others": others}


def _jpeg(path: str, side: int = 640) -> bytes:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def identify_panel(panel_path: str, cands: List[Dict[str, Any]], *,
                   chat: Optional[Callable] = None, model: str = MODEL,
                   exemplar_images: Optional[List[bytes]] = None,
                   load: Callable[[str], bytes] = _jpeg) -> Dict[str, Any]:
    """One call. A failed or unparseable call confirms nobody (never raises)."""
    names = [c["name"] for c in cands]
    if chat is None:
        from ollama_compat import chat as chat            # noqa: PLW0127
    imgs = list(exemplar_images if exemplar_images is not None
                else [load(p) for c in cands for p in c["exemplars"]])
    try:
        resp = chat(model=model, think=False,
                    options={"temperature": 0, "num_ctx": 8192, "num_predict": 80},
                    messages=[{"role": "user", "content": build_prompt(names),
                               "images": imgs + [load(panel_path)]}])
        return parse_reply((resp.get("message") or {}).get("content"), names)
    except Exception as e:                                # noqa: BLE001
        print("[identity] %s: %s -> nobody confirmed"
              % (os.path.basename(panel_path), type(e).__name__))
        return {"names": [], "others": 0}


def identify_panels(ep_dir: str, registry: Any, *,
                    chat: Optional[Callable] = None, model: str = MODEL,
                    out_path: str = "",
                    load: Callable[[str], bytes] = _jpeg
                    ) -> Optional[Dict[str, Any]]:
    """Identify every panel of *ep_dir* that shows people; write
    manifest.identity.json. None when the series has no exemplars (the chapter
    then keeps today's keyword behaviour rather than a half-trusted mix)."""
    cands = candidates(registry)
    if not cands:
        return None
    understood = os.path.join(ep_dir, "manifest.panels.understood.json")
    with open(understood, encoding="utf-8") as f:
        panels = (json.load(f) or {}).get("panels") or []
    exemplar_images = [load(p if os.path.isabs(p) else os.path.abspath(p))
                       for c in cands for p in c["exemplars"]]
    out: Dict[str, Any] = {}
    for p in panels:
        fn = os.path.basename(str(p.get("scene_file") or ""))
        if (not fn or not (p.get("subjects") or [])
                or str(p.get("panel_kind") or "").lower() in _SKIP_KINDS):
            continue
        path = os.path.join(ep_dir, "scenes", fn)
        if not os.path.exists(path):
            continue
        out[fn] = identify_panel(path, cands, chat=chat, model=model,
                                 exemplar_images=exemplar_images, load=load)
    obj: Dict[str, Any] = {"panels": out}
    from manifest_io import write_manifest
    target = out_path or os.path.join(ep_dir, "manifest.identity.json")
    write_manifest(target, obj, inputs=[understood], tool="panel_identity",
                   extra_meta={"candidates": [c["name"] for c in cands],
                               "model": model})
    return obj


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--series-cast", required=True,
                    help="cast/<slug>.json — the owner's exemplars live there")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    try:
        with open(args.series_cast, encoding="utf-8") as f:
            registry = json.load(f)
    except OSError:
        print("[identity] no series cast %s — skipped" % args.series_cast)
        return 0
    got = identify_panels(args.episode_dir, registry, model=args.model,
                          out_path=args.out)
    if got is None:
        print("[identity] no exemplars in %s — skipped (keyword identity stands)"
              % args.series_cast)
        return 0
    named = sum(1 for v in got["panels"].values() if v["names"])
    print("[ok] identity: %d panels, %d with a confirmed character"
          % (len(got["panels"]), named))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
