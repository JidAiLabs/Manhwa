#!/usr/bin/env python3
"""
publish_concept.py — ONE coherent publish package per unit (single chapter now;
bundle range later): title + thumbnail hook + thumbnail style + synopsis +
hashtags + description + pinned comment.

Coherence by construction: title and the thumbnail label both read from the same
concept, so they can't drift. Copyright-safe: the licensed series name never
appears in title / description / thumbnail — only in the PINNED COMMENT (user
decision). $0 — local Gemma for the copy, deterministic for style/templates.

Output: <episode>/render/publish_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from thumbnail_styles import STYLE_MODULES, select_style, style_for  # noqa: E402
from youtube_meta import chapter_digest, extract_json          # noqa: E402

# Channel-static boilerplate (edit Patreon / email once). Real series name is NOT
# here — it goes only in the pinned comment.
CHANNEL = {
    "name": "OriginPower Manhwa Recap",
    "patreon": "https://www.patreon.com/originpowermanhwa",
    "email": "originpowermanhwa@gmail.com",
}
_DISCLAIMER = (
    "I don't own the manhwa/artwork. All rights to their respective owners. "
    "For any concern or removal, contact {email} before a copyright claim.")
_BASE_TAGS = ("manhwa recap, manhwa, webtoon, manhwa recaps, manga recap, "
              "manhua recap, anime recap, recap, manhwa summary, webtoon recap")


# What a hook must LOOK LIKE for each style, so pick_hook's style branches are
# actually reachable. Without this the model was only told "punchy", so the
# before_after branch (which looks for an "A|B" pair) never matched and the
# split composition always fell back to a generic literal BEFORE / AFTER.
# What a thumbnail label IS, learned from the thumbnails that actually perform.
# Every one of them is a NAMETAG you could point an arrow at -- what a character
# IS or BECAME (a role, a rank, a title), a power number, or a status flash. NOT
# one is atmospheric. Mood phrases ("the story bleeds in", "no more hiding")
# read as captions on a picture and are exactly what this spec exists to stop.
# Deliberately no worked example: handing the model a finished label is how
# "LEVEL 999" got copied out of this very prompt and onto a live thumbnail.
_HOOK_GRAMMAR = (
    "A label is a NAMETAG, not a caption: it names WHAT SOMEONE IS or WHAT THEY "
    "BECAME — a role, a title, a rank, a status — so a viewer could draw an "
    "arrow from it to a person in the picture. Never a mood, never atmosphere, "
    "never a sentence or a clause. Use this story's OWN words for the role. ")

_HOOK_SHAPE = {
    "triptych": (_HOOK_GRAMMAR +
                 '3 labels. EVERY one is a THREE-part progression written as '
                 '"FIRST|MIDDLE|LAST" (1-2 words per part) — what this '
                 'character is at the start, at the turning point, and at the '
                 'end of the arc. Each part is a role or status this story '
                 'actually gives them, in its own words. A part may end in "?" '
                 'where the story leaves it open'),
    "before_after": (_HOOK_GRAMMAR +
                     '3 labels. EVERY one is a contrasting PAIR of ROLES or '
                     'RANKS written as "BEFORE SIDE|AFTER SIDE" (1-2 words per '
                     'side): what this character was, then what they became. '
                     'Both sides must be states this story actually gives them'),
    # NO literal example here, deliberately. The previous wording ended
    # 'e.g. "LEVEL 999", "RANK SSS"' and the model returned BOTH verbatim as
    # hooks -- ORV shipped "LEVEL 999 PROPHET" on its series thumbnail while the
    # highest number anywhere in 54 chapters of narration is 11. A worked
    # example of the exact thing being asked for is an answer, not a format hint.
    "stat_callout": (_HOOK_GRAMMAR +
                     '3 labels, 1-4 words each; at least two must contain a '
                     'NUMBER or RANK that ACTUALLY APPEARS in the STORY DIGEST '
                     'below. Never invent one; if the digest has no numbers or '
                     'ranks, use the role or title it does give instead'),
}
_HOOK_DEFAULT = (_HOOK_GRAMMAR +
                 '3 labels, 1-3 words each. Each names a role, title, rank or '
                 'status this story gives someone — the kind of tag that could '
                 'sit beside a character with an arrow pointing at them')


def build_concept_prompt(digest: str, banned: str, style: str) -> str:
    hook_spec = _HOOK_SHAPE.get(style, _HOOK_DEFAULT)
    return (
        "You write copyright-safe metadata for a manhwa RECAP video. NEVER use "
        f"this licensed title (or any part of it): {banned or '(none)'}.\n"
        f"Chosen thumbnail style: {style} (the hook should suit it).\n\n"
        "From the STORY DIGEST, return ONLY JSON:\n"
        "{\n"
        '  "title": "clickbait recap title, 60-95 chars, trope-based, CAPS for '
        'emphasis, NO real names",\n'
        f'  "hooks": ["{hook_spec}"],\n'
        '  "tags": ["2-3 SHORT thumbnail tags, 1-2 words each. Each must name '
        'something THIS story actually contains, using ITS OWN words from the '
        'STORY DIGEST — the creature, role, place or in-world term it really '
        'uses. Do NOT fall back on generic fantasy labels (demon king, S-rank, '
        'chosen one) unless the digest itself uses them: a tag that would fit '
        'any manhwa is worthless. Never invent a number or rank. A '
        'transformation may be written as \\"BEFORE -> AFTER\\""],\n'
        '  "synopsis": "2-4 sentence teaser with emojis, trope framing, NO real '
        'names",\n'
        '  "hashtags": ["6-10 hashtags incl #manhwa #manga + genre/theme"]\n'
        "}\n\nSTORY DIGEST:\n" + digest)


_DIGIT = re.compile(r"\d|\bS+\b|\brank\b|\blevel\b|\blvl\b|\bSSS?\b", re.I)


# The concrete NUMBER / RANK tokens a hook ASSERTS about the story. These are
# factual claims printed on the thumbnail, so each one has to exist in the
# source. Plain words are not claims and are never checked.
_HOOK_CLAIM_RE = re.compile(r"\d+|\bS{2,}\b", re.IGNORECASE)


def hook_claims(hook: str) -> List[str]:
    """Number/rank tokens *hook* asserts (upper-cased, de-duplicated in order)."""
    seen: List[str] = []
    for m in _HOOK_CLAIM_RE.finditer(str(hook or "")):
        t = m.group(0).upper()
        if t not in seen:
            seen.append(t)
    return seen


def hook_is_grounded(hook: str, corpus: str) -> bool:
    """True when every number/rank *hook* claims actually occurs in *corpus*.

    An empty corpus returns True: absence of evidence is not evidence of
    fabrication, and silently rejecting every hook would be worse than the
    occasional invented one. Word-bounded so "11" does not ground "999".
    """
    if not str(corpus or "").strip():
        return True
    hay = str(corpus).upper()
    return all(re.search(r"(?<![0-9A-Z])%s(?![0-9A-Z])" % re.escape(c), hay)
               for c in hook_claims(hook))


# Words a tag can contain without naming anything — never evidence of grounding.
_TAG_STOPWORDS = frozenset({
    "the", "and", "for", "with", "into", "from", "that", "this", "then",
    "his", "her", "its", "their", "our", "you", "your", "are", "was", "were",
    "has", "have", "who", "what", "when", "where", "will", "wont", "cant",
})


def story_vocabulary(ep_dirs: Optional[List[str]]) -> set:
    """Terms that are TRUE BY CONSTRUCTION, for validating subject tags.

    NOT a corpus search. Checking a tag against the narration blob does not
    work: at ~208k words nearly every common English word appears somewhere, so
    "WEAK -> GOD" and "DEMON KING" both passed. Word frequency fails the other
    way -- it ranks 'king' (95) and 'god' (48) above 'script' (10), admitting
    the generic trope while rejecting this story's most central idea.

    So the vocabulary is ENUMERATED instead of inferred, from two sources that
    cannot contain a word the story does not actually use:
      * cast names the extractor found ON the pages (manifest.cast.json)
      * words printed on panels the understanding stamped as in-world SYSTEM
        screens -- their OCR lives in manifest.vision.json, not the understood
        manifest, whose ocr_clean is empty for these panels.
    Stopwords are dropped and a system word must appear at least twice, so OCR
    noise and ordinary English do not become "story terms".
    """
    vocab: set = set()
    counts: Dict[str, int] = {}
    for d in (ep_dirs or []):
        try:
            cast = json.load(open(os.path.join(d, "manifest.cast.json")))
        except Exception:
            cast = {}
        for m in (cast.get("cast") or cast.get("members") or []):
            for w in re.findall(r"[a-z']{3,}",
                                str((m or {}).get("name") or
                                    (m or {}).get("id") or "").lower()):
                vocab.add(w)
        try:
            u = json.load(open(os.path.join(d, "manifest.panels.understood.json")))
            v = json.load(open(os.path.join(d, "manifest.vision.json")))
        except Exception:
            continue
        sysf = {os.path.basename(str(p.get("scene_file") or ""))
                for p in (u.get("panels") or [])
                if str(p.get("panel_kind") or "").lower() == "system"}
        for it in (v.get("items") or []):
            if os.path.basename(str(it.get("scene_file") or "")) not in sysf:
                continue
            for w in re.findall(r"[a-z']{3,}",
                                str(it.get("ocr_clean") or "").lower()):
                counts[w] = counts.get(w, 0) + 1
    vocab |= {w for w, n in counts.items()
              if n >= 2 and w not in _TAG_STOPWORDS}
    return vocab - _TAG_STOPWORDS


def tag_is_grounded(tag: str, vocab: set) -> bool:
    """True when every CONTENT word of *tag* is in the enumerated *vocab*.

    A tag NAMES something, so it is checked word by word. A tag with nothing
    checkable ("THE ONE") is rejected: it asserts a thing that cannot be traced
    to the story, which is how a generic power-fantasy trope attaches itself to
    a story that is not one. An empty vocabulary rejects every tag rather than
    waving them through -- an unverifiable tag is not a safe default here, and
    the badge still carries the layout on its own.
    """
    words = [w for w in re.findall(r"[a-z']{3,}", str(tag or "").lower())
             if w not in _TAG_STOPWORDS]
    if not words or not vocab:
        return False
    return all(w in vocab for w in words)


def pick_tags(tags: Any, vocab: set, corpus: str = "",
              limit: int = 2) -> List[Dict[str, Any]]:
    """Grounded subject tags with layout positions, most specific first.

    Positions are assigned here (not by the model): lower-left then upper-left,
    which is where the working layouts put their subject labels — opposite the
    main hook so the two never stack."""
    slots = [("lower_left", True), ("mid_left", False)]
    cands = [str(t or "").strip() for t in (tags or []) if str(t or "").strip()]
    # stable sort: tags whose words appear in the enumerated story vocabulary
    # (cast names, in-world system screens) lead, the rest keep their order.
    if vocab:
        cands.sort(key=lambda s: 0 if tag_is_grounded(s, vocab) else 1)
    out: List[Dict[str, Any]] = []
    for s in cands:
        if len(out) >= min(limit, len(slots)):
            break
        # Guard only what is FALSIFIABLE. A number or rank is a claim about the
        # story ("LEVEL 999" was false: the real maximum is 11), so it is
        # checked. A thematic word is a description, not a claim -- and the
        # model reading 17k chars of this chapter's narration judges that far
        # better than any word list. An enumerated vocabulary was tried and
        # rejected THE SCRIPT and THE PROPHET, this story's two most central
        # ideas, because they are narration prose and never printed on a system
        # screen. `vocab`, when supplied, only ORDERS tags (most story-specific
        # first); it never rejects one.
        if not hook_is_grounded(s, corpus):
            continue
        pos, arrow = slots[len(out)]
        out.append({"text": s.upper(), "pos": pos, "arrow": arrow})
    return out


def beats_text_corpus(beats_obj: Dict[str, Any]) -> str:
    """The TEXT a hook may legitimately draw a number from.

    Deliberately narrow: narration lines and beat summaries ONLY, never the raw
    manifest. Searching serialized JSON would ground a hook on geometry -- a
    normalized bbox like 0.9995 contains "999" and would have "verified" the
    exact fabrication this guards against.
    """
    parts: List[str] = []
    for b in (beats_obj or {}).get("beats") or []:
        if not isinstance(b, dict):
            continue
        for k in ("narration", "hook", "what_happens", "summary"):
            v = b.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        for s in b.get("segments") or []:
            if isinstance(s, dict) and isinstance(s.get("line"), str):
                parts.append(s["line"])
    return "\n".join(parts)


def pick_hook(hooks: List[str], style: str, *, corpus: str = "") -> str:
    """Choose the thumbnail label that best fits the style (deterministic).

    *corpus* is the story text a stat-style hook's number must appear in. The
    stat branch used to return the FIRST hook containing any digit, which meant
    it actively preferred an invented stat over a grounded plain label (ORV:
    "LEVEL 999 PROPHET" was chosen over "THE SCRIPT IS BROKEN").
    """
    hooks = [str(h).strip() for h in (hooks or []) if str(h).strip()]
    if not hooks:
        return ""
    if style == "stat_callout":
        grounded = [h for h in hooks if hook_is_grounded(h, corpus)]
        for h in grounded:
            if _DIGIT.search(h):
                return h
        # every stat-shaped hook invents its number: ship a grounded plain
        # label rather than print a false claim on the thumbnail.
        if grounded:
            return grounded[0]
    if style == "triptych":
        # needs THREE parts: a two-part hook would leave the last panel unlabelled
        for h in hooks:
            if h.count("|") >= 2:
                return h
    if style == "before_after":
        for h in hooks:
            if "|" in h:
                return h
    return hooks[0]


def build_description(synopsis: str, hashtags: List[str]) -> str:
    tags = " ".join(t if t.startswith("#") else "#" + t.lstrip("#")
                    for t in (hashtags or []) if str(t).strip())
    return "\n\n".join(filter(None, [
        synopsis.strip(),
        tags,
        f"▶ Patreon: {CHANNEL['patreon']}",
        f"📩 Business: {CHANNEL['email']}",
        _DISCLAIMER.format(email=CHANNEL["email"]),
        "Tags: " + _BASE_TAGS,
    ]))


def pinned_comment(real_title: str, official_link: str = "") -> str:
    t = (real_title or "").strip() or "(see description)"
    tail = f" — read the official release: {official_link}" if official_link else \
           " — please support the official release."
    return f"Manhwa: {t}{tail}"


_INTENSITY = {"calm": 0, "unknown": 0, "tense": 1, "intense": 2, "explosive": 3}


def sample_arc_indices(n: int, *, max_chapters: int,
                       climax_index: Optional[int] = None) -> List[int]:
    """Which chapter indices to describe, preserving the ARC SHAPE.

    Always keeps the opening, the ending and the climax — the three points a
    title/synopsis actually needs — then spreads the remaining budget evenly
    across the middle so setup->payoff stays visible.
    """
    if n <= max_chapters:
        return list(range(n))
    keep = {0, n - 1}
    if climax_index is not None and 0 <= climax_index < n:
        keep.add(climax_index)
    remaining = max_chapters - len(keep)
    if remaining > 0:
        step = (n - 1) / (remaining + 1)
        for k in range(1, remaining + 1):
            keep.add(min(n - 1, max(0, round(k * step))))
    return sorted(keep)[:max_chapters]


def bundle_digest(beats_objs: List[Dict[str, Any]], *,
                  per_chapter_chars: int = 700,
                  max_chapters: int = 24,
                  climax_index: Optional[int] = None) -> str:
    """Aggregate MANY chapters into one arc digest that fits the LLM context:
    a compact per-chapter summary (hooks + the punchiest beats), so the title
    can span the whole arc (setup -> payoff), which a single chapter can't.

    BOUNDED at max_chapters. This used to describe EVERY chapter, which grew
    the prompt linearly and without limit: measured at ~713 chars/chapter, a
    300-chapter series produced ~213,000 chars ≈ 53,000 tokens. The MLX
    backend ignores num_ctx and simply processes that, so it did not fail
    loudly — it just spent ~3 minutes of prefill (at the ~307 tok/s measured
    on this hardware) and a large KV cache to write a title and three hooks.

    Chapters are SAMPLED, not truncated: the opening, the ending and the
    climax are always kept, with the rest spread evenly across the middle.
    Labels carry the REAL chapter position so the model still sees where each
    excerpt sits in the arc. The climax itself is chosen separately, in pure
    Python over ALL chapters (select_bundle_climax) — that scan is cheap and
    stays exhaustive, so bounding the digest does not affect which moment the
    thumbnail depicts.
    """
    idxs = sample_arc_indices(len(beats_objs), max_chapters=max_chapters,
                              climax_index=climax_index)
    parts: List[str] = []
    for i in idxs:
        b = beats_objs[i]
        lines: List[str] = []
        for bt in b.get("beats") or []:
            t = (str(bt.get("hook") or "").strip()
                 or str(bt.get("what_happens") or "").strip())
            if t:
                lines.append(t)
        blob = " ".join(lines)[:per_chapter_chars]
        if blob:
            tag = " (CLIMAX)" if i == climax_index else ""
            parts.append(f"[Chapter {i + 1} of {len(beats_objs)}{tag}] {blob}")
    return "\n".join(parts)


def select_bundle_climax(beats_objs: List[Dict[str, Any]]):
    """Pick the most thumbnail-worthy moment across the bundle from BEATS: the
    highest-intensity kept beat. Returns (chapter_index, scene_files).

    This is the FALLBACK path — a plain argmax over a 4-value intensity enum,
    so on a long arc many beats tie at 'explosive' and the strict '>' keeps the
    FIRST one, i.e. the earliest, not the best. When the understood-panel
    manifests are available (the normal case) build_bundle_concept prefers
    select_bundle_climax_scored, which ranks by the same weighted signal model
    the teaser uses so the two agree on the arc's peak.
    """
    best = (-1, 0, [])  # (intensity, chapter_index, scene_files)
    for ci, b in enumerate(beats_objs):
        for bt in b.get("beats") or []:
            scenes = [s for s in (bt.get("scene_selection") or [])
                      if isinstance(s, dict)]
            inten = max((_INTENSITY.get(str(s.get("intensity") or "").lower(), 0)
                         for s in scenes), default=0)
            if inten > best[0]:
                files = [str(s.get("scene_file")) for s in scenes
                         if s.get("role", "keep") != "redundant" and s.get("scene_file")]
                best = (inten, ci, files[:3])
    return best[1], best[2]


def select_bundle_climax_scored(ep_dirs: List[str]):
    """Pick the arc's peak from the UNDERSTOOD PANELS, scored by the same
    weighted signal model the teaser uses (teaser_planner.score_panel), so the
    thumbnail and the cold open agree on what the climax is.

    Returns (ep_index, [ref_basename]) matching select_bundle_climax's shape,
    or None when no understood manifests exist (caller falls back to beats
    intensity). ref is a bare basename resolved against that chapter's scenes/.
    """
    try:
        import teaser_planner as _tp
    except Exception:
        return None
    panels: List[Dict[str, Any]] = []
    for i, d in enumerate(ep_dirs or []):
        man = os.path.join(d, "manifest.panels.understood.json")
        if not os.path.exists(man):
            continue
        try:
            data = json.load(open(man))
        except Exception:
            continue
        for p in data.get("panels") or []:
            q = dict(p)
            q["_ep_index"] = i        # rides back on the returned climax panel
            q["scene_file"] = os.path.basename(str(p.get("scene_file") or ""))
            panels.append(q)
    if not panels:
        return None
    climax = _tp.select_climax_panel(panels)
    if not climax:
        return None
    sf = climax.get("scene_file")
    return climax["_ep_index"], ([sf] if sf else [])


def _kept_panels(beats_obj: Dict[str, Any]):
    """(scene_file, intensity) for kept panels, in reading order."""
    out = []
    for bt in beats_obj.get("beats") or []:
        for s in bt.get("scene_selection") or []:
            if not isinstance(s, dict):
                continue
            if str(s.get("role") or "keep") == "redundant":
                continue
            f = str(s.get("scene_file") or "")
            if f:
                out.append((f, str(s.get("intensity") or "calm").lower()))
    return out


def select_before_ref(beats_objs: List[Dict[str, Any]], ep_dirs: List[str], *,
                      climax_ci: int) -> str:
    """A 'weakest moment' reference panel from BEFORE the climax, as an
    ABSOLUTE path (it usually lives in a different chapter than the climax).

    The before_after style promises the same character weak on the left and
    transformed on the right. But bundle refs all came from the single climax
    beat, so both halves were painted from the SAME moment — there was no
    "before" at all, and the composition's whole premise was unsupported.

    Searches the earliest chapters first for a calm/tense kept panel, which is
    where a protagonist is most likely to be shown at their weakest.
    """
    for ci in range(0, max(1, min(climax_ci, len(beats_objs)))):
        for fn, inten in _kept_panels(beats_objs[ci]):
            if inten in ("calm", "tense"):
                if ci < len(ep_dirs):
                    return os.path.join(ep_dirs[ci], "scenes", fn)
                return fn
    return ""


def assemble_concept(beats_obj: Dict[str, Any], llm: Dict[str, Any], *,
                     series_title: str, genre: str = "",
                     official_link: str = "",
                     style: str = "",
                     vocab: Optional[set] = None) -> Dict[str, Any]:
    """Build the concept from beats (style) + the LLM copy. Pure/testable.

    *style* forces the thumbnail style instead of deriving it from the beats.
    The hook SHAPE differs per style (before_after wants an "A|B" pair), so the
    same value must reach build_concept_prompt and this call or the hooks the
    model wrote will not match the style they are picked for."""
    style = style or select_style(beats_obj, genre=genre)
    hooks = llm.get("hooks") or []
    corpus = beats_text_corpus(beats_obj)
    hook = pick_hook(hooks, style, corpus=corpus)
    tags = pick_tags(llm.get("tags"), vocab or set(), corpus=corpus)
    synopsis = str(llm.get("synopsis") or "").strip()
    hashtags = llm.get("hashtags") or ["#manhwa", "#manga", "#manhwarecap"]
    return {
        "title": str(llm.get("title") or "").strip(),
        "style": style,
        "style_overlay": style_for(style)["overlay"],
        "hook": hook,
        "hooks": hooks,
        "tags": tags,
        "synopsis": synopsis,
        "hashtags": hashtags,
        "description": build_description(synopsis, hashtags),
        "pinned_comment": pinned_comment(series_title, official_link),
    }


def _fmt_ts(sec: float) -> str:
    s = int(sec)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parts_timestamps(durations: List[float],
                     labels: Optional[List[str]] = None) -> List[str]:
    """YouTube-chapter 'Parts' list: cumulative offsets, first MUST be 0:00."""
    out: List[str] = []
    t = 0.0
    for i, d in enumerate(durations):
        out.append(f"{_fmt_ts(t)} {labels[i] if labels else f'Part {i + 1}'}")
        t += float(d or 0.0)
    return out


def build_bundle_concept(beats_list: List[Dict[str, Any]], llm: Dict[str, Any],
                         *, durations: List[float], series_title: str,
                         genre: str = "", official_link: str = "",
                         labels: Optional[List[str]] = None,
                         ep_dirs: Optional[List[str]] = None,
                         style: str = "") -> Dict[str, Any]:
    """Concept for a VIDEO (bundle of N chapters): arc title/synopsis from the
    aggregated chapters, style+refs from the bundle's CLIMAX chapter, and the
    Parts (YouTube-chapter) timestamps appended to the description."""
    # prefer the weighted understood-panel scorer (agrees with the teaser);
    # fall back to beats intensity when understood manifests aren't present
    scored = select_bundle_climax_scored(ep_dirs or [])
    climax_ci, refs = scored if scored else select_bundle_climax(beats_list)
    style_beats = beats_list[climax_ci] if 0 <= climax_ci < len(beats_list) else {}
    # the vocabulary spans the WHOLE bundle: a system term or cast name printed
    # in any chapter of this video is fair to tag, not just the climax chapter's
    c = assemble_concept(style_beats, llm, series_title=series_title,
                         genre=genre, official_link=official_link, style=style,
                         vocab=story_vocabulary(ep_dirs))
    c["parts"] = parts_timestamps(durations, labels)
    c["climax_chapter_index"] = climax_ci
    # Status badge: a FACT about this upload, never a claim about the story, so
    # it carries the competitor layout's badge slot with nothing to fabricate.
    n = len(beats_list or [])
    if n:
        c["badge"] = "%d CHAPTERS" % n
    # The before_after composition needs BOTH halves: a weak "before" panel and
    # the transformed climax. Climax refs alone painted both halves from one
    # moment. The before panel usually lives in an earlier chapter, so it is an
    # ABSOLUTE path — refs from the climax chapter stay bare filenames.
    if c.get("style") == "before_after":
        before = select_before_ref(beats_list, ep_dirs or [],
                                   climax_ci=climax_ci)
        if before:
            refs = [before] + [r for r in refs if r != before]
    c["refs"] = refs
    c["description"] = c["description"] + "\n\n" + "\n".join(c["parts"])
    return c


def _gemma(prompt: str, model: str) -> Dict[str, Any]:
    from ollama_compat import chat as _chat
    resp = _chat(model=model, think=False,
                 messages=[{"role": "user", "content": prompt}],
                 options={"temperature": 0.8, "num_predict": 800})
    raw = (resp.get("message") or {}).get("content") or ""
    got = extract_json(raw)
    if not got:
        # LOUD. This returned {} silently, so a run that produced nothing wrote
        # an EMPTY concept (hook='' title='') and reported [ok]. Measured on
        # qwen3.6:27b: it wrote a good title and three good hooks, then emitted
        # one unquoted hashtag, and the whole reply was discarded without a word.
        raise RuntimeError(
            "%s returned no parseable JSON (%d chars). First 200: %r"
            % (model, len(raw), raw[:200]))
    return got


def _plan_duration(ep: str) -> float:
    for fn in ("render.plan.clean.json", "render.plan.json"):
        try:
            return float(json.load(open(os.path.join(ep, fn))).get("total_duration_sec") or 0.0)
        except Exception:
            continue
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", default="", help="single-chapter mode")
    ap.add_argument("--episode-dirs", default="", help="comma-separated chapter "
                    "dirs = a BUNDLE/video (arc title + Parts). Use for videos.")
    ap.add_argument("--series-title", default="", help="licensed title — BAN list "
                    "(never in title/desc/thumb) + pinned-comment credit")
    ap.add_argument("--genre", default="")
    ap.add_argument("--official-link", default="")
    ap.add_argument("--style", default="",
                    # DERIVED from the registry, never hand-listed: a
                    # hardcoded list silently went stale the moment `triptych`
                    # was added, and argparse rejected --style triptych with
                    # exit 2 after the style itself was already working.
                    choices=[""] + sorted(STYLE_MODULES),
                    help="force the thumbnail style instead of deriving it "
                         "from the beats — for generating variants to compare. "
                         "Empty (default) = auto-select, the production path.")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--digest-chapters", type=int, default=24,
                    help="max chapters described to the LLM (bundle mode). "
                         "Sampled to keep opening/climax/ending — the climax "
                         "SCAN still covers every chapter.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.episode_dirs:
        eps = [e for e in args.episode_dirs.split(",") if e]
        beats_list = [json.load(open(os.path.join(e, "manifest.beats.json"))) for e in eps]
        durations = [_plan_duration(e) for e in eps]
        # the climax scan is exhaustive (cheap, pure Python over every
        # chapter); only the LLM DIGEST is bounded — see bundle_digest. Same
        # scorer build_bundle_concept uses, so style/digest/refs all agree on
        # which chapter is the peak.
        _sc = select_bundle_climax_scored(eps)
        climax_ci = (_sc[0] if _sc
                     else (select_bundle_climax(beats_list)[0] if beats_list else 0))
        style = args.style or select_style(
            beats_list[climax_ci] if beats_list else {}, genre=args.genre)
        digest = bundle_digest(beats_list, max_chapters=args.digest_chapters,
                               climax_index=climax_ci)
        if len(beats_list) > args.digest_chapters:
            print(f"[..] digest: sampled {args.digest_chapters} of "
                  f"{len(beats_list)} chapters (climax #{climax_ci + 1} kept) "
                  f"— {len(digest):,} chars")
        llm = _gemma(build_concept_prompt(digest, args.series_title, style),
                     args.ollama_model)
        concept = build_bundle_concept(beats_list, llm, durations=durations,
                                       series_title=args.series_title, genre=args.genre,
                                       official_link=args.official_link,
                                       ep_dirs=eps, style=style)
        out = args.out or os.path.join(eps[0], "render", "bundle_publish_meta.json")
    else:
        if not args.episode_dir:
            ap.error("need --episode-dir (single) or --episode-dirs (bundle)")
        beats_obj = json.load(open(os.path.join(args.episode_dir, "manifest.beats.json")))
        style = args.style or select_style(beats_obj, genre=args.genre)
        llm = _gemma(build_concept_prompt(chapter_digest(beats_obj), args.series_title, style),
                     args.ollama_model)
        concept = assemble_concept(beats_obj, llm, series_title=args.series_title,
                                   genre=args.genre, official_link=args.official_link,
                                   style=style,
                                   vocab=story_vocabulary([args.episode_dir]))
        out = args.out or os.path.join(args.episode_dir, "render", "publish_meta.json")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(concept, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote={out} style={concept['style']} "
          f"hook={concept['hook']!r} title={concept['title']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
