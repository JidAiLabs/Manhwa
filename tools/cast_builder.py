"""
tools/cast_builder.py — build a chapter CAST registry so recap narration names the
same character consistently instead of re-introducing "a figure" every scene.

One Gemini multimodal pass over all OCR + a sampling of panels -> manifest.cast.json:
  {cast: [{id, canonical_name, aliases[], role, visual_description, is_protagonist}]}

Names come from the dialogue/OCR (proper names or address terms like "Ancestor-nim");
unnamed-but-recurring figures get a stable descriptive handle. gemini_narrative_pass
then threads this in (--cast) so every group's narration matches faces to the cast.

Runs on the local ollama Gemma — no credentials, no cloud.

  V=.eval_venv/bin/python
  $V tools/cast_builder.py \
      --groups-manifest  ongoing/nano-machine/Chapter_1/manifest.groups.json \
      --vision-manifest  ongoing/nano-machine/Chapter_1/manifest.vision.json \
      --out              ongoing/nano-machine/Chapter_1/manifest.cast.json \
      --project <proj> --location us-central1
"""

import argparse
import json
import re
import os
import sys
from typing import Any, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from manifest_io import write_manifest                                # noqa: E402


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _vision_items(vision: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = vision.get("items") or vision.get("scenes") or []
    if isinstance(items, dict):
        items = list(items.values())
    return items


def _sample_images(items: List[Dict[str, Any]], cap: int) -> List[str]:
    """Pick visually-informative panels spread across the chapter (low text coverage
    first within an even stride), returning on-disk scene_paths."""
    withpath = [it for it in items if it.get("scene_path")]
    if not withpath:
        return []
    # even stride across the chapter so we see characters from start to end
    stride = max(1, len(withpath) // max(1, cap))
    picked = withpath[::stride][:cap]

    def tc(it: Dict[str, Any]) -> float:
        try:
            return float(it.get("text_coverage")) if it.get("text_coverage") is not None else 0.3
        except Exception:
            return 0.3
    picked.sort(key=tc)  # most visual first
    return [str(it["scene_path"]) for it in picked]


def _page_words(items: List[Dict[str, Any]]) -> str:
    """Text that is really ON the page for name validation: per-word OCR with
    sane geometry (word boxes inside the image, not glyph-sized) when the
    vision manifest carries ocr_words; else the cleaned line text."""
    try:
        from apple_vision import sane_text_region  # sibling tool
    except Exception:  # pragma: no cover
        sane_text_region = None
    out: List[str] = []
    for it in items:
        words = (it.get("vision") or {}).get("ocr_words") or []
        if words and sane_text_region is not None:
            for wd in words:
                bb = wd.get("bbox") or []
                if len(bb) == 4 and sane_text_region(*[float(v) for v in bb]):
                    out.append(str(wd.get("t") or ""))
        else:
            out.append(str(it.get("ocr_clean") or ""))
    return "\n".join(out)


def _all_ocr(items: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for it in items:
        o = (it.get("ocr_clean") or "").strip()
        if o:
            out.append(f"{it.get('scene_file')}: {o[:200]}")
    return out


_DESCRIPTIVE_LEAD = ("the ", "our ", "a ", "an ", "one ", "their ", "his ", "her ",
                     "that ", "this ", "some ", "two ", "three ")


def _looks_like_proper_name(name: str) -> bool:
    n = (name or "").strip()
    if not n or n.lower().startswith(_DESCRIPTIVE_LEAD):
        return False
    toks = [t for t in re.split(r"[\s\-]+", n) if t]
    return bool(toks) and all(t[:1].isupper() for t in toks)


def _name_in_text(name: str, ocr_lower: str) -> bool:
    """Every alphabetic token of *name* occurs in the chapter text (address
    terms like 'Ancestor-nim' match on 'ancestor' + 'nim')."""
    toks = [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if len(t) >= 2]
    return bool(toks) and all(re.search(r"\b" + re.escape(t) + r"\b", ocr_lower) for t in toks)


def _handle_from_description(desc: str, role: str) -> str:
    """'A masked figure wearing a dark brown hooded cloak…' -> 'the masked
    figure'; falls back to 'the <role>'. Deterministic, no model."""
    d = re.sub(r"\s+", " ", (desc or "").strip())
    d = re.sub(r"^(a|an|the)\s+", "", d, flags=re.I)
    d = re.split(r"[,.;:(]| wearing | with | who | in a | in an | holding | carrying ", d, maxsplit=1)[0]
    words = [w for w in d.split(" ") if w][:3]
    if 1 <= len(words) <= 3 and all(re.match(r"^[A-Za-z\-]+$", w) for w in words):
        return "the " + " ".join(words).lower()
    return "the " + (role.strip().lower() or "figure")


def sanitize_cast_names(cast: Dict[str, Any], ocr_text: str):
    """Names must come from the page. The prompt says so, but the model still
    coins one now and then (nano ch1: the unnamed masked leader became "Alden"
    and every line shipped it). A proper-name-looking canonical_name / alias
    whose tokens never occur in the chapter OCR is replaced by the descriptive
    spoken_name (else "the <id words>"); invented aliases are dropped. The
    protagonist's 'our protagonist' handle and descriptive handles are left
    alone. Returns (cast, [(id, invented, replacement), ...])."""
    ocr_lower = (ocr_text or "").lower()
    fixes = []
    for m in (cast or {}).get("cast") or []:
        name = str(m.get("canonical_name") or "")
        if _looks_like_proper_name(name) and not _name_in_text(name, ocr_lower):
            spoken = str(m.get("spoken_name") or "").strip()
            bad_toks = {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}
            id_words = re.sub(r"[_\-]+", " ", str(m.get("id") or "")).strip().lower()
            if spoken and spoken.lower().startswith(_DESCRIPTIVE_LEAD):
                repl = spoken
            elif id_words and not (bad_toks & set(id_words.split())):
                repl = "the " + id_words
            else:
                repl = _handle_from_description(str(m.get("visual_description") or ""),
                                                str(m.get("role") or ""))
            m["canonical_name"] = repl
            if spoken and spoken.lower() == name.lower():
                m["spoken_name"] = repl
            fixes.append((str(m.get("id") or ""), name, repl))
        if m.get("aliases"):
            m["aliases"] = [a for a in m["aliases"]
                            if not (_looks_like_proper_name(str(a)) and not _name_in_text(str(a), ocr_lower))]
    return cast, fixes


def recurring_figures(understood: Any, min_panels: int = 3) -> List[str]:
    """Recurring-figure candidate lines from the per-panel understanding —
    evidence-driven cast coverage (2026-07-20 story-state wave: the blue-sash
    helper who performs the ch1 kill recurred across panels yet was absent
    from the sampled-images cast, so the identity gate had nothing to attach
    the strike to).

    Pure code, no model call: subjects are clustered by their cast_identity
    appearance tokens (cluster core = INTERSECTION of member token sets, so a
    cluster is defined by what its subjects have in COMMON and can never
    drift into swallowing everyone); a cluster seen in >= min_panels distinct
    panels yields one line '<most common subject wording> — seen in N panels
    (files…)'. The prompt then requires each listed figure to appear in the
    cast (entry or alias) — the model decides which."""
    from collections import Counter

    from cast_identity import _looks_person, _subject_tokens
    clusters: List[Dict[str, Any]] = []
    for p in ((understood or {}).get("panels") or []):
        if not isinstance(p, dict):
            continue
        fn = str(p.get("scene_file") or "")
        for s in (p.get("subjects") or []):
            s = str(s).strip()
            if not _looks_person(s):
                continue                 # props/scenery are not cast material
            toks = _subject_tokens(s)
            if len(toks) < 2:
                continue                 # generic-only: no identity evidence
            best, best_core = None, ()
            for c in clusters:
                core = c["tokens"] & toks
                if len(core) >= 2 and len(core) > len(best_core):
                    best, best_core = c, core
            if best is None:
                clusters.append({"tokens": set(toks), "files": {fn},
                                 "reps": Counter([s])})
            else:
                best["tokens"] = set(best_core)
                best["files"].add(fn)
                best["reps"][s] += 1
    out: List[str] = []
    for c in sorted(clusters, key=lambda c: -len(c["files"])):
        if len(c["files"]) < min_panels:
            continue
        rep = c["reps"].most_common(1)[0][0]
        files = ", ".join(sorted(c["files"])[:3])
        out.append(f"{rep} — seen in {len(c['files'])} panels ({files}…)")
        if len(out) >= 8:
            break
    return out


CAST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "cast": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "canonical_name": {"type": "STRING"},
                    # What the NARRATOR says aloud. canonical_name is an
                    # identity KEY (it must distinguish a leader from a
                    # member); spoken_name is prose. "Assassin Member" is a
                    # fine key and unusable narration — a viewer heard "The
                    # Assassin Member watches him with a sharp gaze". NOT in
                    # `required`: older casts fall back to canonical_name.
                    "spoken_name": {"type": "STRING"},
                    "aliases": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "role": {"type": "STRING"},
                    "visual_description": {"type": "STRING"},
                    "is_protagonist": {"type": "BOOLEAN"},
                },
                "required": ["id", "canonical_name", "role", "visual_description", "is_protagonist"],
            },
        }
    },
    "required": ["cast"],
}

SYSTEM = (
    "You are building a CAST LIST for ONE webtoon/manhwa chapter so a recap video can name "
    "the same character consistently instead of calling them 'a figure' every scene.\n"
    "You are given a sampling of panels (across the whole chapter) plus all OCR/dialogue text.\n"
    "Identify the recurring or story-important characters. For EACH:\n"
    "  id: short snake_case handle (e.g. 'protagonist', 'dying_master', 'hooded_leader').\n"
    "  canonical_name: a REAL name if one appears in the dialogue/OCR (a proper name, or an\n"
    "    address term actually used such as 'Ancestor-nim'); otherwise a short descriptive\n"
    "    handle ('the dying old master'). This is what narration will call them.\n"
    "    EXCEPTION — the PROTAGONIST: if the main character has no real name in the text,\n"
    "    canonical_name MUST be exactly 'our protagonist' (recap-native). NEVER a descriptive\n"
    "    phrase like 'the web novel reader' or 'the young man'.\n"
    "  spoken_name: what a NARRATOR would actually SAY out loud for this\n"
    "    character — natural spoken prose, not a catalogue label. A viewer\n"
    "    hears this sentence, so 'one of the assassins', 'their leader', 'the\n"
    "    stranger', 'the old master' are right; 'Assassin Member',\n"
    "    'Mysterious Figure', 'Character 2' are WRONG. When the character has\n"
    "    a real name, the spoken_name IS that name. canonical_name may stay a\n"
    "    precise label for telling look-alikes apart; spoken_name is the\n"
    "    words that get read aloud.\n"
    "  aliases: other ways they're named/addressed in the text.\n"
    "  role: one of protagonist | antagonist | ally | mentor | minor | group (for a faceless band).\n"
    "  visual_description: appearance cues a downstream model can MATCH in a panel — age, hair,\n"
    "    clothing, weapon, distinctive features. Be concrete.\n"
    "  is_protagonist: true for the single main character the recap follows.\n"
    "Ignore anonymous crowds. Prefer 6-14 entries. Return ONLY JSON matching the schema."
)

STORY_CAST_RULE = (
    "CHARACTERS THE CHAPTER'S DIALOGUE ALREADY IDENTIFIED (authoritative):\n"
    "{CHARS}\n"
    "These were read from what the characters actually SAY, so they are "
    "authoritative for WHO is in this chapter and what they are called.\n"
    "  - EVERY character listed must appear in the cast, including one-scene "
    "figures such as a stranger or a rescuer: a character missing from the "
    "cast cannot be recognised in a panel later, and the narration ends up "
    "calling them something generic.\n"
    "  - Record each character's REAL name (the one used in the dialogue) in "
    "`aliases` so the narration can introduce them by it.\n"
    "  - KEEP the recap-native convention for the PROTAGONIST: his "
    "canonical_name stays 'our protagonist' — that is the relaxed handle the "
    "narration uses most of the time — with his real name in `aliases`. Do "
    "NOT promote a real name to the protagonist's canonical_name; that turns "
    "it into a label repeated every line.\n"
    "Your job is to add the APPEARANCE (visual_description) for each, which "
    "the dialogue could not provide."
)

SERIES_CAST_RULE = (
    "LOCKED SERIES CAST (owner-verified, authoritative over anything you see "
    "in this chapter):\n"
    "{CAST}\n"
    "  - Use these names exactly. Their look is FIXED: whatever you write for "
    "them will be replaced by the locked description, so spend your effort on "
    "everyone else.\n"
    "  - Any character NOT listed must be described CONTRASTIVELY: state what "
    "tells them apart from the locked member they most resemble (glasses, "
    "coat colour, beard, hair length). Never give a new character a look that "
    "also fits a locked description — two indistinguishable descriptions mean "
    "the narration names the wrong person.\n"
)

RECURRING_RULE = (
    "RECURRING FIGURES OBSERVED (from per-panel analysis of the whole chapter):\n"
    "{FIGURES}\n"
    "Every listed recurring figure MUST appear in the cast — either as its own "
    "entry or as an alias/visual_description of an existing member. A recurring "
    "figure who performs or receives violence is story-important by definition; "
    "never omit them."
)


_REGISTRY_FIELDS = ("canonical_name", "spoken_name", "visual_description",
                    "not", "is_protagonist")


def _name_keys(m: Dict[str, Any]) -> set:
    keys = {str(m.get("canonical_name") or "").strip().lower()}
    keys.update(str(a).strip().lower() for a in (m.get("aliases") or []))
    keys.discard("")
    return keys


def apply_series_cast(cast: Dict[str, Any], registry: Any):
    """The owner's series registry wins over this chapter's guess.

    A member gemma emitted that matches a registry entry (both protagonist,
    or any shared name/alias) takes the entry's name, look and `not` traits
    VERBATIM — cast_builder re-guessed appearance per chapter and once gave
    two characters the same look (ORV Ep128). A second gemma member matching
    the same entry is folded into it (aliases kept). Entries absent from the
    chapter are NOT injected. Returns (cast, members_locked)."""
    entries = [e for e in ((registry or {}).get("cast") or [])
               if isinstance(e, dict) and e.get("canonical_name")]
    out: List[Dict[str, Any]] = []
    claimed: Dict[int, Dict[str, Any]] = {}
    for m in cast.get("cast") or []:
        hit = next((i for i, e in enumerate(entries)
                    if (bool(m.get("is_protagonist")) and bool(e.get("is_protagonist")))
                    or (_name_keys(m) & _name_keys(e))), None)
        if hit is None:
            out.append(m)
            continue
        extra = (set(m.get("aliases") or []) | {str(m.get("canonical_name") or "")}) - {""}
        if hit in claimed:
            locked = claimed[hit]
            locked["aliases"] = sorted((set(locked.get("aliases") or []) | extra)
                                       - {locked["canonical_name"]})
            continue
        e = entries[hit]
        locked = dict(m)
        for k in _REGISTRY_FIELDS:
            if k in e:
                locked[k] = e[k]
        locked["aliases"] = sorted((set(e.get("aliases") or []) | extra)
                                   - {locked["canonical_name"]})
        claimed[hit] = locked
        out.append(locked)
    return dict(cast, cast=out), len(claimed)


def _series_cast_block(registry: Any) -> str:
    lines = []
    for e in ((registry or {}).get("cast") or []):
        if not isinstance(e, dict) or not e.get("canonical_name"):
            continue
        aka = ", ".join(str(a) for a in (e.get("aliases") or []))
        never = ", ".join(str(x) for x in (e.get("not") or []))
        lines.append(f"  - {e['canonical_name']}" + (f" (aka {aka})" if aka else "")
                     + f" — look: {e.get('visual_description') or '?'}"
                     + (f"; NEVER: {never}" if never else ""))
    return SERIES_CAST_RULE.replace("{CAST}", "\n".join(lines)) if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-manifest", required=True)
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", "--ollama-model", dest="model",
                    default="gemma4:26b")
    ap.add_argument("--backend", choices=["ollama"], default="ollama",
                    help="deprecated no-op: local ollama is the only backend")
    ap.add_argument("--max-images", type=int, default=24)
    ap.add_argument("--understood", default="",
                    help="manifest.panels.understood.json — recurring-figure "
                         "coverage (every recurring figure must land in the cast)")
    ap.add_argument("--chapter-story", default="",
                    help="manifest.chapter_story.json (story_pass) — the "
                         "character list read from the chapter's own dialogue. "
                         "Its NAMES are authoritative; this pass supplies the "
                         "appearance the story pass cannot see.")
    ap.add_argument("--series-cast", default="",
                    help="ongoing/<slug>/series_cast.json — the owner's locked "
                         "cast: names + looks restored verbatim, never re-guessed")
    args = ap.parse_args()

    registry = None
    if args.series_cast and os.path.exists(args.series_cast):
        registry = _load(args.series_cast)
    series_block = _series_cast_block(registry)

    vision = _load(args.vision_manifest)
    items = _vision_items(vision)
    images = _sample_images(items, args.max_images)
    ocr = _all_ocr(items)

    story_block = ""
    if args.chapter_story and os.path.exists(args.chapter_story):
        try:
            chars = (_load(args.chapter_story).get("cast") or [])
        except Exception:
            chars = []
        if chars:
            story_block = STORY_CAST_RULE.replace("{CHARS}", "\n".join(
                f"  - {c.get('name')} — {c.get('role')}" for c in chars
                if isinstance(c, dict) and c.get("name")))

    figures_block = ""
    if args.understood and os.path.exists(args.understood):
        try:
            figs = recurring_figures(_load(args.understood))
        except Exception:
            figs = []
        if figs:
            figures_block = RECURRING_RULE.replace(
                "{FIGURES}", "\n".join(f"  - {f}" for f in figs))

    import ollama
    text_blobs = [SYSTEM]
    if series_block:
        text_blobs.append(series_block)
    if story_block:
        text_blobs.append(story_block)
    if figures_block:
        text_blobs.append(figures_block)
    text_blobs += ["ALL DIALOGUE / OCR IN THE CHAPTER (scene_file: text):\n"
                   + "\n".join(ocr),
                   f"\n{len(images)} sample panels follow:"]
    import re as _re

    def _lower_types(o):
        if isinstance(o, dict):
            return {k: (v.lower() if k == "type" and isinstance(v, str)
                        else _lower_types(v))
                    for k, v in o.items() if k != "propertyOrdering"}
        if isinstance(o, list):
            return [_lower_types(x) for x in o]
        return o

    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from ollama_compat import chat as _ollama_chat
    resp = _ollama_chat(
        model=args.model,
        messages=[{"role": "user", "content": "\n".join(text_blobs),
                   "images": [str(pth) for pth in images]}],
        format=_lower_types(CAST_SCHEMA), think=False,
        options={"temperature": 0.2, "num_ctx": 16384,
                 "num_predict": 2400})
    try:
        cast = json.loads(resp["message"]["content"])
    except json.JSONDecodeError as e:
        # self-repair retry (nano ch1 job 144: an unterminated string cost a
        # 12-min job restart) — one more roll with more room to finish
        print(f"[warn] cast JSON unparseable ({e}); retrying once with a "
              "larger num_predict")
        # `_ollama_chat`, NOT `client.chat`: a stale `client.chat` here
        # raised UnboundLocalError, so the self-repair never ran once.
        resp = _ollama_chat(
            model=args.model,
            messages=[{"role": "user", "content": "\n".join(text_blobs),
                       "images": [str(pth) for pth in images]}],
            format=_lower_types(CAST_SCHEMA), think=False,
            options={"temperature": 0.3, "num_ctx": 16384,
                     "num_predict": 4000})
        cast = json.loads(resp["message"]["content"])
    # names come from the page — replace invented proper names (nano ch1: "Alden")
    cast, name_fixes = sanitize_cast_names(cast, _page_words(items))
    for cid, bad, good in name_fixes:
        print(f"[fix] cast {cid}: invented name {bad!r} (not in chapter OCR) -> {good!r}")
    if registry is not None:
        # AFTER sanitize: a locked name absent from this chapter's OCR must
        # survive (sanitize would have replaced it with a handle)
        cast, n_locked = apply_series_cast(cast, registry)
        print(f"[series] {n_locked} cast member(s) locked from {args.series_cast}")
    inputs = [args.groups_manifest, args.vision_manifest]
    if args.understood and os.path.exists(args.understood):
        inputs.append(args.understood)
    if args.chapter_story and os.path.exists(args.chapter_story):
        inputs.append(args.chapter_story)
    if registry is not None:
        inputs.append(args.series_cast)
    write_manifest(args.out, cast, inputs=tuple(inputs),
                   tool="cast_builder",
                   extra_meta={"model": args.model, "images_used": len(images),
                               "ocr_lines": len(ocr)})

    members = cast.get("cast") or []
    print(f"[ok] {args.out} — {len(members)} cast members")
    for c in members:
        star = " *PROTAGONIST*" if c.get("is_protagonist") else ""
        print(f"  - {c.get('canonical_name')} ({c.get('role')}){star}: {c.get('visual_description','')[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
