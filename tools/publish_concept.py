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
import unicodedata
from typing import Any, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from thumbnail_styles import (  # noqa: E402
    CLAIM_TONES,
    DEFAULT_STYLE,
    DEFAULT_TONE,
    HOOK_DESIGNS,
    STYLE_MODULES,
    select_style,
    style_for,
)
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

# The layouts the AUTO path may choose: one cohesive scene each. before_after is
# only ever built as its own option (--style before_after); stat_callout only as
# an explicit variant (it produced "LEVEL 999" on a story whose max level is 11).
# Only styles whose ref slots code can FILL: feat_object asks for "a giant
# weapon/hammer" and no code finds an object ref, so ORV got an invented stone
# hammer (job 1694). vs_monster / humiliation have the same lead-only gap.
# ponytail: re-add each one when its counterpart ref finder exists.
SCENE_STYLES = ["power_reveal"]

_HOOK_SHAPE = {
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


def assemble_package(beats_obj: Dict[str, Any], brief: Dict[str, Any],
                     pkg: Dict[str, Any], *, series_title: str,
                     official_link: str = "",
                     styles: Optional[List[str]] = None,
                     design: str = "",
                     card_lines: Optional[List[str]] = None
                     ) -> Dict[str, Any]:
    """Concept from the two-stage understanding. Pure/testable.

    The model's own layout choice wins, validated against the registry (an
    unknown name falls back to the default rather than crashing the run). The
    labels it wrote for THAT layout become the hook, so the shape it chose and
    the text it wrote can never disagree -- the failure mode when a keyword
    heuristic picked the layout and the model wrote for a different one.
    """
    # The offered list is ENFORCED, not just asked for: a prompt is a request,
    # and the scene option must never come back as a split (or vice versa).
    allowed = [s for s in (styles or SCENE_STYLES) if s in STYLE_MODULES]
    style = str(pkg.get("thumbnail_style") or "").strip()
    if style not in allowed:
        style = allowed[0]
    labels = _clean_labels(pkg.get("labels"))
    # a pair written as "A|B" on a one-scene layout is a transformation label
    # ("DEAD -> KING"), never a literal pipe. (The split's labels are fixed
    # BEFORE / AFTER below, so no model label is ever joined for it.)
    labels = [s.replace("|", " -> ") for s in labels]
    if design and design not in HOOK_DESIGNS:
        raise ValueError("unknown hook design: %r" % design)

    corpus = beats_text_corpus(beats_obj)
    # numbers stay guarded: a rank or level is a checkable claim about the
    # story, and an invented one is what put LEVEL 999 on a live thumbnail.
    # Only grounded candidates are OFFERED (the owner switches between them).
    labels = [s for s in labels if hook_is_grounded(s, corpus)]
    if style == "before_after":
        # Owner: "use before / after, not invent words such as Observer /
        # protagonist". The split's labels are literally BEFORE and AFTER.
        labels = ["BEFORE|AFTER"]
    hook = labels[0] if labels else ""
    synopsis = str(pkg.get("description") or "").strip()
    hashtags = pkg.get("hashtags") or ["#manhwa", "#manga", "#manhwarecap"]
    c = {
        "title": normalize_title(pkg.get("title")),
        "style": style,
        "style_reason": str(pkg.get("style_reason") or "").strip(),
        "style_overlay": style_for(style)["overlay"],
        "hook": hook,
        "hooks": labels,
        "brief": brief,
        "synopsis": synopsis,
        "hashtags": hashtags,
        "description": build_description(synopsis, hashtags),
        "pinned_comment": pinned_comment(series_title, official_link),
    }
    if design:
        # the DESIGN owns the label layer; the style still owns the art
        overlay = HOOK_DESIGNS[design]["overlay"]
        c["design"] = design
        c["style_overlay"] = overlay
        heads = [s.replace("|", " -> ")
                 for s in _clean_labels(pkg.get("headlines"))]
        heads = [s for s in heads if hook_is_grounded(s, corpus)]
        if overlay.get("headline_pos") and heads:
            c["headlines"] = heads
            c["tags"] = [{"text": heads[0], "pos": overlay["headline_pos"],
                          "arrow": False}]
        if card_lines:
            c["card"] = list(card_lines)
    return c


def _clean_labels(raw: Any) -> List[str]:
    """`labels` may come back as a bare STRING ("READER|SURVIVOR|AUTHOR")
    rather than a list -- iterating that yields one CHARACTER per label and
    the hook becomes "R". A JSON schema in a prompt is a request, not a
    guarantee."""
    raw = raw or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(x).strip() for x in raw if str(x).strip()]


def _names_the_title(line: str, banned: str) -> bool:
    """True when *line* carries the licensed title: the whole phrase, or any
    two consecutive title words. ONE shared word is the story's own vocabulary
    ("reader"), not the name."""
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", str(x or "").lower()).split()
    title, text = norm(banned), " " + " ".join(norm(line)) + " "
    if not title:
        return False
    pairs = ([title] if len(title) == 1
             else [title[i:i + 2] for i in range(len(title) - 1)])
    return any(" " + " ".join(p) + " " in text for p in pairs)


def system_card_lines(montage: List[Dict[str, Any]], *,
                      max_lines: int = 3, banned: str = "") -> List[str]:
    """The system window's words, QUOTED from the teaser's system panels:
    usable printed text only, no repeats, in teaser order. Code quotes;
    the model never writes a card (an invented stat shipped once)."""
    out: List[str] = []
    for p in montage:
        if str(p.get("panel_kind") or "") != "system":
            continue
        text = printable_card_line(p.get("printed"))
        # integrity rule 1, the one with legal weight: the licensed series name
        # is never rendered, and a system window can name it
        if text and text not in out and not _names_the_title(text, banned):
            out.append(text)
    return out[:max_lines]


def build_brief_prompt(digest: str, banned: str,
                       hook_block: str = "") -> str:
    """STAGE 1: understand the story before writing a word of copy.

    The digest is raw narration prose -- 17k chars of moment-to-moment
    voiceover stitched across 24 chapters. Asked to write a title straight from
    that, a model pattern-matches on fragments and returns fragment-shaped
    copy ("THE STORY IS REAL", "TARGET"). No amount of validating the OUTPUT
    fixes that; the input was never a story, just its debris.

    So first make it state what the story IS. A human writing this copy reads
    the series, forms a view, and only then writes the title. This is that step.
    """
    return (
        "You are reading the narration of a manhwa recap to UNDERSTAND the "
        "story. Do not write any marketing copy yet.\n"
        f"NEVER repeat this licensed title or any part of it: {banned or '(none)'}\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "premise": "one sentence: the situation the story sets up",\n'
        '  "protagonist": "who they are and what makes THEM different from '
        'the usual manhwa lead — their actual edge, not a power level",\n'
        '  "engine": "the thing that keeps happening that drives the story",\n'
        '  "arc": "how the protagonist''s situation CHANGES from the first of '
        'these chapters to the last",\n'
        '  "distinctive": ["2-4 things a reader would remember about THIS '
        'series that they would not find in a generic power-fantasy manhwa"],\n'
        '  "why_watch": "the curiosity a viewer would click to satisfy"\n'
        "}\n\n" + (hook_block + "\n\n" if hook_block else "")
        + "STORY NARRATION:\n" + digest)


def build_package_prompt(brief: Dict[str, Any], banned: str,
                         styles: Optional[List[str]] = None,
                         thumb_labels: str = "",
                         design: str = "") -> str:
    """STAGE 2: write the whole publish package FROM the understanding.

    Takes the stage-1 brief, not the raw narration -- the model is now writing
    about a story it has stated in its own words, which is the difference
    between describing a series and echoing its debris. It also CHOOSES the
    thumbnail layout: which composition suits the story is a judgement about
    the story, and a keyword heuristic kept picking stat_callout for a series
    whose largest number is 11.

    The auto path offers SINGLE-SCENE layouts only. before_after is always
    built as its own option (--style before_after) and the owner picks between
    them on the Series page -- the model never chooses between a split and a
    scene. 27 of 27 example thumbnails are one scene; 0 are split at all.

    *thumb_labels* is the picked series thumbnail's text: a video title reuses
    its status words so the thumbnail and the title make the same claim.
    """
    opts = styles or SCENE_STYLES
    same_claim = (
        f"The series thumbnail already shows: {thumb_labels}. Reuse those "
        "status words in the title so the thumbnail and title make the same "
        "claim.\n\n" if thumb_labels else "")
    return (
        "You are writing the YouTube package for a manhwa recap. You have "
        "already read the series; your understanding of it is below.\n"
        f"NEVER use this licensed title or any part of it: {banned or '(none)'}\n"
        "No real character names anywhere in the output.\n\n"
        "STORY UNDERSTANDING:\n" + json.dumps(brief, indent=2) + "\n\n"
        "Write copy that could ONLY belong to this story. Anything that would "
        "fit any manhwa is a failure.\n\n" + same_claim +
        "Return ONLY JSON:\n"
        "{\n"
        '  "title": "YouTube title, 45-80 characters. Shape: WHO the '
        'protagonist starts as (a low status, a humiliating role or an odd '
        'identity, in this story\'s own terms), then the TURN (the verb of '
        'change), then their specific EDGE (the concrete power or trick, with '
        'its number when the story states one), then the PAYOFF (what it does '
        'to the people or world around them). FULL CAPS on the 2-4 words that '
        'carry status or power. No emoji, no character names, no question '
        'marks, no channel name. A number must be one the story states",\n'
        '  "description": "3-5 sentences a viewer reads to decide. Open with '
        'the premise, say what changes, end on the open question. No hashtag '
        'list, no boilerplate — prose only",\n'
        f'  "thumbnail_style": "ONE of: {", ".join(opts)} — choose the '
        'composition that fits THIS story\'s shape, and say why in reason",\n'
        '  "style_reason": "one sentence: why that composition suits it",\n'
        + (_design_labels_ask(design) if design else
           '' if opts == ["before_after"] else
           # before_after is always labelled BEFORE / AFTER (owner), so it asks
           # for no labels at all
           '  "labels": ["5 candidate labels for the thumbnail, 1-4 words each. '
           'Each puts the protagonist at an EXTREME on a ladder a viewer '
           'understands instantly -- rank, power level, wealth, age or social '
           'status: the very bottom, the very top, or the jump from bottom to top '
           'written as LOW -> HIGH. It must make a viewer ask HOW. Only facts of '
           'THIS story; a number or rank must be one the story states. Never a '
           'plain role or job name that any story could have, never a mood"],\n')
        + '  "hashtags": ["6-10 hashtags incl #manhwa #manga + genre/theme"]\n'
        "}")


def thumb_label_words(concept: Dict[str, Any]) -> str:
    """Every word the PICKED thumbnail shows, for the title to reuse so both
    make one claim. A hook design draws its headline as a tag, so the hook
    alone would leave the title blind to half the thumbnail."""
    words = [x.strip() for x in str(concept.get("hook") or "").split("|")]
    words += [str((t or {}).get("text") or "").strip()
              for t in concept.get("tags") or []]
    return " / ".join(w for w in words if w)


def _design_labels_ask(design: str) -> str:
    """The labels ask for a ranked hook design: its own grammar, plus a
    headlines key only when the design draws one."""
    d = HOOK_DESIGNS[design]
    ask = '  "labels": ["' + d["label_grammar"] + '"],\n'
    if d["overlay"].get("headline_pos"):
        ask += ('  "headlines": ["the 5 candidate headlines described '
                'above, 2-3 words each"],\n')
    return ask


_SCENE_ASK = (
    '  "scene": "ONE sentence an illustrator could paint: the single moment of '
    'THE HOOK that shows what THIS story is. Say WHERE it happens, WHAT is '
    'happening, what the protagonist is DOING and what the people around them '
    'are doing. Concrete nouns from the story (the place, the object, the '
    'creature), never a generic power pose. Any window, screen, sign or book in '
    'it is BLANK and glowing: no letters, no numbers. Nothing from later than '
    'the hook. No real character names",\n')


def build_claim_prompt(brief: Dict[str, Any], banned: str,
                       tone: str = DEFAULT_TONE) -> str:
    """ONE claim per series: the title, the labels, the headlines AND the scene
    to paint, written in a single call from the teaser-led understanding.

    Every option card, the image prompt and the video titles read this one
    answer. Before it, each option made its own two calls (two summaries, two
    label lists, two titles for the same series), and the understanding never
    reached the image model: every series was painted as "a hero with an aura
    and shocked onlookers". Owner, 2026-09-22: "we dont see a story on the
    thumbnail, the video title, teaser and thumbnail must in complementary".
    *tone* (thumbnail_styles.CLAIM_TONES) picks WHICH moment of the hook.
    """
    if tone not in CLAIM_TONES:
        raise ValueError("unknown claim tone: %r" % tone)
    return (
        "You are writing the ONE publish claim for a manhwa recap series. You "
        "have already read its opening; your understanding is below.\n"
        f"NEVER use this licensed title or any part of it: {banned or '(none)'}\n"
        "No real character names anywhere in the output.\n\n"
        "STORY UNDERSTANDING:\n" + json.dumps(brief, indent=2) + "\n\n"
        "Everything you write must belong to THIS story only. Anything that "
        "would fit any manhwa is a failure.\n\n"
        + CLAIM_TONES[tone] + "\n\n"
        "Return ONLY JSON:\n{\n"
        '  "title": "YouTube title, 45-80 characters: WHO the protagonist starts '
        'as, the TURN, their specific EDGE, the PAYOFF. FULL CAPS on the 2-4 '
        'status words. No emoji, no character names, no question marks. A number '
        'must be one the story states",\n'
        '  "description": "3-5 sentences a viewer reads to decide. Prose only",\n'
        + _SCENE_ASK
        + _design_labels_ask("nametag_headline")
        + '  "hashtags": ["6-10 hashtags incl #manhwa #manga + genre/theme"]\n'
        "}")


def concept_from_claim(claim: Dict[str, Any], design: str) -> Dict[str, Any]:
    """One option card's concept, built from the series claim with NO model
    call: the cards share labels, scene, refs and title by construction and
    differ only in the label DESIGN."""
    if design not in HOOK_DESIGNS:
        raise ValueError("unknown hook design: %r" % design)
    overlay = HOOK_DESIGNS[design]["overlay"]
    labels = [str(x) for x in claim.get("labels") or []]
    c = {k: claim.get(k) for k in (
        "title", "style", "brief", "scene", "refs", "claim_source",
        "teaser_panels", "design_reason", "climax_chapter_index", "badge",
        "synopsis", "hashtags", "description", "pinned_comment", "parts")
         if claim.get(k) is not None}
    c.update({"design": design, "style_overlay": overlay,
              "hook": labels[0] if labels else "", "hooks": labels})
    heads = [str(x) for x in claim.get("headlines") or []]
    if overlay.get("headline_pos") and heads:
        c["headlines"] = heads
        c["tags"] = [{"text": heads[0], "pos": overlay["headline_pos"],
                      "arrow": False}]
    if overlay.get("card"):
        lines = [str(x) for x in claim.get("card") or [] if str(x).strip()]
        if not lines:
            raise ValueError("%s needs a printable system line and the claim "
                             "has none" % design)
        # ONE line: the examples' windows hold a single short line in huge
        # type. Three sentences of fine print in a box read as nothing.
        c["card"] = lines[:1]
    return c


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
    if style == "before_after":
        for h in hooks:
            if "|" in h:
                return h
    return hooks[0]


_TITLE_SUFFIX = " - Manhwa Recap"
_SUFFIX_RE = re.compile(
    r"[\s\-|:(\[\u2013\u2014]*manhwa\s+recaps?[\s)\]!.]*$", re.IGNORECASE)


def normalize_title(title: Any) -> str:
    """The title in the format that performs, enforced where it is mechanical.

    Counted over 27 example titles (18 refs + the owner's 9, 2026-09-17):
    every one ends in the "Manhwa Recap" suffix (26 with a dash, 1 with a
    pipe), none carries an emoji, and 7 of the owner's 9 capitalise every word
    while the status words stay FULL CAPS. The model is asked for the
    shape; these three things are not left to it. Empty stays empty.
    """
    t = "".join(ch for ch in str(title or "")
                if unicodedata.category(ch) not in ("So", "Sk", "Cs")
                and ch not in "\u200d\ufe0f")
    t = _SUFFIX_RE.sub("", " ".join(t.split()))
    if not t:
        return ""
    return " ".join(w[:1].upper() + w[1:] for w in t.split()) + _TITLE_SUFFIX


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


# A reference panel exists to show the image model WHO the character is. The
# scorer ranks panels by dramatic signal alone, and in a system-driven story the
# most dramatic panel is very often a status window -- so the ORV thumbnail was
# generated from two references containing no person at all, and the prompt's
# "use the EXACT character designs from the reference images" was unfollowable.
# The model invented a military uniform because nothing showed it a human.
_UI_SUBJECT_RE = re.compile(
    r"\b(?:window|notification|screen|display|message|panel of text|text box|"
    r"dialog|interface|status|menu|card|logo|caption|subtitle|watermark|"
    r"speech bubble|thought bubble|sound effect|sfx)\b", re.IGNORECASE)


def panel_shows_a_character(panel: Dict[str, Any]) -> bool:
    """True when a panel actually depicts a person, per the understanding.

    Deliberately an EXCLUSION test, not a list of person-words: a character can
    be described in ways no keyword list anticipates ("a large glowing red eye
    on a pale distorted face"), whereas UI panels describe themselves in a small
    and stable vocabulary. A panel with no recorded subjects is rejected -- no
    evidence of a character is not evidence of one.
    """
    kind = str((panel or {}).get("panel_kind") or "").strip().lower()
    if kind in {"system", "caption", "empty"}:
        return False
    subs = [str(s).strip() for s in ((panel or {}).get("subjects") or [])
            if str(s).strip()]
    if not subs:
        return False
    return not all(_UI_SUBJECT_RE.search(s) for s in subs)


def _norm_name(s: Any) -> str:
    """Cast ids and resolved figure names for the SAME person differ in form
    ('our_protagonist' vs 'our protagonist'), so compare them normalised."""
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


_HAIR_LEN_RE = re.compile(r"\b(long|short)\b[^.;]{0,30}?\bhair\b", re.IGNORECASE)


def hair_length(text: Any) -> Optional[str]:
    """'long' / 'short' when *text* says so of HAIR ("long, dark, messy hair"),
    else None. "a long white coat over dark hair" is a coat: a garment word
    between the two breaks the match."""
    m = _HAIR_LEN_RE.search(str(text or ""))
    if not m or re.search(r"\b(coat|jacket|robe|cloak|scarf|sleeve)s?\b",
                          m.group(0), re.IGNORECASE):
        return None
    return m.group(1).lower()


_FEMALE_RE = re.compile(r"\b(woman|women|girl|lady|female|she|her)\b", re.IGNORECASE)
_MALE_RE = re.compile(r"\b(man|men|boy|guy|male|he|his)\b", re.IGNORECASE)


def gender_word(text: Any) -> Optional[str]:
    """'f' / 'm' from gender words in a description, None when it has none or
    both. ("woman" contains no word-bounded "man".)"""
    t = str(text or "")
    f, m = bool(_FEMALE_RE.search(t)), bool(_MALE_RE.search(t))
    return "f" if f and not m else "m" if m and not f else None


def protagonist_name(cast_obj: Dict[str, Any]) -> str:
    """The cast entry that is the lead, by its own id/name."""
    members = (cast_obj or {}).get("cast") or (cast_obj or {}).get("members") or []
    for m in members:
        ident = str((m or {}).get("id") or (m or {}).get("name") or "").lower()
        if "protagonist" in ident or ident in {"mc", "hero", "lead"}:
            return str((m or {}).get("name") or (m or {}).get("id") or "")
    return str((members[0].get("name") or members[0].get("id") or "")
               ) if members else ""


def protagonist_portrait_files(ep_dir: str, max_figures: int = 2) -> set:
    """Panels where the LEAD is present and the frame is not a crowd."""
    return _lead_panels(ep_dir, max_figures)[0]


def _lead_panels(ep_dir: str, max_figures: int = 2):
    """(portraits, look): lead portraits, and the subset whose subjects show
    POSITIVE evidence of the registry look (its stated hair length).

    Portraits: panels where the LEAD is present and the frame is not a crowd.

    Requiring merely "a person" was not enough. The chosen reference had EIGHT
    subjects -- a boot, two system windows, a fire-breathing creature, an
    armoured figure and three men -- and the image model copied one of them
    faithfully: a side character in a navy polo. It was never inventing; it was
    given a crowd and no way to know which figure was the lead.
    """
    try:
        u = json.load(open(os.path.join(ep_dir, "manifest.panels.understood.json")))
        c = json.load(open(os.path.join(ep_dir, "manifest.cast.json")))
    except Exception:
        return set(), set()
    lead = protagonist_name(c)
    if not lead:
        return set(), set()
    # The IMAGE pass, when the chapter has one: same authority as the
    # narration. Keyword matching suggested 5 other men among 8 "MC" tiles;
    # the image pass got 8 of 8 in the 2026-09-17 spike.
    try:
        with open(os.path.join(ep_dir, "manifest.identity.json"),
                  encoding="utf-8") as f:
            ident = (json.load(f) or {}).get("panels") or {}
    except (OSError, ValueError):
        ident = {}
    if ident:
        conf = {fn for fn, rec in ident.items()
                if _norm_name(lead) in {_norm_name(n) for n in (rec.get("names") or [])}
                and len(rec.get("names") or []) + int(rec.get("others") or 0)
                <= max_figures}
        return conf, conf          # image-confirmed IS the look evidence
    try:
        from cast_identity import resolve_figures_by_file
    except Exception:
        return set(), set()
    # The registry's hair LENGTH, which cast_identity never compares (it keys
    # on colour): 390 of 2925 ORV "lead portraits" describe long hair while
    # the registry says short, and the long-haired one became the after half.
    lead_len = next((hair_length(m.get("visual_description"))
                     for m in ((c.get("cast") or c.get("members") or []))
                     if _norm_name(m.get("id")) == _norm_name(lead)
                     or _norm_name(m.get("canonical_name")) == _norm_name(lead)),
                    None)
    subjects = {os.path.basename(str(p.get("scene_file") or "")):
                (p.get("subjects") or []) for p in (u.get("panels") or [])}
    # hair COLOUR too, via cast_identity's own classes: suggestions offered
    # "short light blonde hair" and "short grey hair" as ORV's dark-haired lead
    from cast_identity import _hair_colors, _tokens
    lead_desc = next((str(m.get("visual_description") or "")
                      for m in ((c.get("cast") or c.get("members") or []))
                      if _norm_name(m.get("id")) == _norm_name(lead)
                      or _norm_name(m.get("canonical_name")) == _norm_name(lead)),
                     "")
    lead_hc = _hair_colors(_tokens(lead_desc))

    def clashes(s: str) -> bool:
        hc = _hair_colors(_tokens(s))
        return (bool(lead_len) and hair_length(s) not in (None, lead_len)) or \
            (bool(lead_hc) and bool(hc) and not (hc & lead_hc))
    out: set = set()
    for f, figs in (resolve_figures_by_file(u, c) or {}).items():
        if any(clashes(s) for s in subjects.get(os.path.basename(str(f)), [])):
            continue
        names = [str(x.get("name") or "") for x in (figs or []) if x.get("name")]
        # Count EVERY figure, not just the identified ones. Counting only named
        # figures let a five-figure battle panel score as 2 -- the three
        # 'unknown' entries are still people in the frame, and one of them is
        # what the image model copied.
        # Match on a NORMALISED name: the cast carries the id ("our_protagonist")
        # while resolve_figures_by_file returns the display name ("our
        # protagonist"), so a literal comparison never matched and this filter
        # silently produced nothing, falling back a tier on every chapter.
        if _norm_name(lead) in {_norm_name(n) for n in names} \
                and len(names) <= max_figures:
            out.add(os.path.basename(str(f)))
    # the evidence must be about someone who COULD be the lead: subjects are
    # not tied to figures, and Ep196's "a woman with short dark hair" beside
    # Dokja counted as his short hair
    lead_g = next((gender_word(m.get("visual_description"))
                   for m in ((c.get("cast") or c.get("members") or []))
                   if _norm_name(m.get("id")) == _norm_name(lead)
                   or _norm_name(m.get("canonical_name")) == _norm_name(lead)),
                  None)
    look = {f for f in out if lead_len and any(
        hair_length(s) == lead_len
        # "a person with short dark hair" is not evidence either: when the
        # registry states a gender, the description must state the same one
        and (gender_word(s) == lead_g if lead_g else True)
        for s in subjects.get(f, []))}
    return out, look


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
        portraits, look_files = _lead_panels(d)
        for p in data.get("panels") or []:
            q = dict(p)
            q["_ep_index"] = i        # rides back on the returned climax panel
            q["scene_file"] = os.path.basename(str(p.get("scene_file") or ""))
            q["_is_portrait"] = q["scene_file"] in portraits
            q["_is_look"] = q["scene_file"] in look_files
            panels.append(q)
    if not panels:
        return None
    # Three tiers, strongest first. "Shows a person" alone was not enough: the
    # panel it chose held EIGHT subjects and the model copied a side character
    # out of the crowd. A reference has one job -- show the image model who the
    # lead IS -- so prefer panels where the lead is present and it is not a
    # crowd. Each tier falls back so a chapter without one still works.
    # Registry look first: a lead panel that SAYS the registry's hair length
    # beats a more dramatic one that doesn't. ORV's climax chapter (Ep306)
    # draws Dokja's hair longer than the registry, and every regenerate
    # painted him long-haired from those refs.
    look = [p for p in panels if p.get("_is_look")]
    portrait = [p for p in panels if p.get("_is_portrait")]
    with_people = [p for p in panels if panel_shows_a_character(p)]
    climax = _tp.select_climax_panel(look or portrait or with_people or panels)
    if not climax:
        return None
    sf = climax.get("scene_file")
    if not sf:
        return climax["_ep_index"], []
    # ONE reference is weak constraint: against a genre prompt the model fills
    # the rest from its priors and drifts toward whatever famous series that
    # description matches, importing costumes and props (a giant sword) that
    # appear nowhere in this story. Send several panels of the SAME lead from
    # the same chapter so appearance is pinned by evidence, not by one frame.
    ep_i = climax["_ep_index"]
    extra = [p["scene_file"] for p in sorted(
                 panels, key=lambda p: 0 if p.get("_is_look") else 1)
             if p.get("_ep_index") == ep_i and p.get("_is_portrait")
             and p.get("scene_file") and p["scene_file"] != sf]
    return ep_i, [sf] + extra[:2]


# '[' and ']' drawn on a system window are read by OCR as 'I' and 'J' glued to
# the first and last word ("INO ONE MAY ENTER ... COMPLETE.J"). BOTH ends must
# agree, so a line that really starts with "INSIDE" is left alone.
_MISREAD_OPEN_RE = re.compile(r"^I(?=[A-Z]{2})")
_MISREAD_CLOSE_RE = re.compile(r"(?<=[A-Za-z.!?])J$")
_CARD_CHARS_RE = re.compile(r"[A-Za-z0-9 .,:;!?'%+\-]*")
_CARD_MIN_WORDS, _CARD_MAX_WORDS = 2, 12


def printable_card_line(raw: Any) -> str:
    """A system window's OCR, fit to PRINT on a thumbnail -- or "".

    The card QUOTES the story, but raw OCR is not printable. Measured on the
    Mini (2026-09-21) over 8 series' teaser windows: ORV has 66 system panels
    and 16 printable lines; the rest is window markup, mirrored text read as
    Cyrillic, crop debris ("E0x") and several windows run into one blob. For
    words a viewer reads at a glance, losing a line beats printing garbage, so
    everything doubtful is refused. Reuses the narration pipeline's own card
    cleaners rather than a second opinion on what a card word is.
    """
    from narration_consistency import is_unvoiceable_line
    import gemini_narrative_pass as _gnp          # heavy: lazy, like the teaser
    t = str(raw or "").strip()
    if _MISREAD_OPEN_RE.search(t) and _MISREAD_CLOSE_RE.search(t):
        t = _MISREAD_CLOSE_RE.sub("", _MISREAD_OPEN_RE.sub("", t))
    t = _gnp.clean_card_text(t)
    words = t.split()
    # ONE sentence: text after a full stop is a second window, a bracket read
    # as a digit ("AVAILABLE.1") or a character's speech glued on; a run of
    # terminal marks ("...!!") is speech, not a system line
    if re.search(r"[.!?]\s*[A-Za-z0-9]|[.!?]{2,}", t):
        return ""
    if (is_unvoiceable_line(t) or not _CARD_CHARS_RE.fullmatch(t)
            or not _CARD_MIN_WORDS <= len(words) <= _CARD_MAX_WORDS):
        return ""
    for tok in re.findall(r"[A-Za-z0-9']+", t):
        letters = re.sub(r"[^A-Za-z]", "", tok.split("'")[0])
        if any(c.isdigit() for c in tok) and letters:
            return ""                              # "E0x": a crop, not a word
        if len(letters) >= 3 and not _gnp._is_card_word(letters):
            return ""
    return t


def _attach_printed(panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Give each montage panel its PRINTED words. They are OCR in the chapter's
    manifest.vision.json; the understood manifest carries none for a system
    window (dialogue is empty on all six of ORV's teaser system panels)."""
    cache: Dict[str, Dict[str, str]] = {}
    for p in panels:
        path = str(p.get("scene_file") or "")
        ep = os.path.dirname(os.path.dirname(path))
        if ep not in cache:
            try:
                with open(os.path.join(ep, "manifest.vision.json"),
                          encoding="utf-8") as f:
                    items = json.load(f).get("items") or []
            except (OSError, ValueError):
                items = []
            cache[ep] = {os.path.basename(str(i.get("scene_file") or "")):
                         str(i.get("ocr_clean") or "") for i in items}
        p["printed"] = cache[ep].get(os.path.basename(path), "")
    return panels


def _panel_key(path: Any):
    """(chapter dir, basename): absolute prefixes differ between the run that
    planned a teaser and the run reading it."""
    path = str(path)
    return (os.path.basename(os.path.dirname(os.path.dirname(path))),
            os.path.basename(path))


def _teaser_manifest_panels(eps: List[str], manifest_path: str):
    """The PLANNED teaser's panels in its own order, each carrying its
    understood fields and its narration line. None when it can't be used."""
    import teaser_planner as _tp
    try:
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        return None
    sources = man.get("panel_sources") or {}
    lines = {str(n.get("scene_file")): str(n.get("line") or "")
             for n in (man.get("panel_narration") or []) if isinstance(n, dict)}
    chapters = {_panel_key(src)[0] for src in sources.values()}
    used = [e for e in eps
            if os.path.basename(os.path.normpath(e)) in chapters]
    understood = {_panel_key(p["scene_file"]): p
                  for p in _tp.load_bundle_panels(used)}
    out = []
    for ns in man.get("scene_files") or []:
        src = sources.get(ns)
        # a panel that is gone from disk cannot be a thumbnail reference
        if not src or not os.path.exists(src):
            return None
        panel = understood.get(_panel_key(src))
        if panel is None:
            return None
        out.append({**panel, "line": lines.get(ns, "")})
    return out or None


def teaser_montage(eps: List[str], manifest_path: str = "", *, scan: int,
                   min_panels: int, max_panels: int, tail_frac: float = 0.0):
    """The teaser's montage (climax LAST) and where it came from.

    The thumbnail sells what the teaser sells, so it reads the same window with
    the same selector. With no planned teaser the montage is computed on the
    fly: pure Python, no model, no state touched. Returns (None, "digest") when
    the window holds too few eligible panels, so the caller can fall back to
    the whole series and SAY that it did.
    """
    if manifest_path:
        planned = _teaser_manifest_panels(eps, manifest_path)
        if planned:
            return _attach_printed(planned), "teaser:manifest"
    import teaser_planner as _tp
    panels = _tp.load_bundle_panels(eps, max_scan_chapters=scan)
    montage = _tp.select_montage(panels, max_panels=max_panels,
                                 min_panels=min_panels,
                                 payoff_tail_frac=tail_frac)
    if not montage:
        return None, "digest"
    return _attach_printed(montage), "montage:computed"


def teaser_hook_block(montage: List[Dict[str, Any]], reason: str = "") -> str:
    """The teaser's montage as the brief's THE HOOK: what each panel shows and
    the line the teaser speaks over it, climax LAST. The thumbnail sells what
    the cold open sells, so the model reads the same moments first."""
    if not montage:
        return ""
    rows = []
    for i, p in enumerate(montage):
        what = str(p.get("description") or p.get("action") or "").strip()
        said = (str(p.get("dialogue") or "").strip()
                or printable_card_line(p.get("printed")))
        line = str(p.get("line") or "").strip()
        rows.append("%d. %s%s%s%s" % (
            i + 1, what, ' | printed: "%s"' % said if said else "",
            " | teaser says: %s" % line if line else "",
            "  <-- CLIMAX: what the lead becomes" if i == len(montage) - 1 else ""))
    head = ("THE HOOK -- the moments this series' teaser is built from, in "
            "order. The thumbnail and title must sell THIS promise.")
    return "\n".join([head] + ([("Why it hooks: " + reason)] if reason else [])
                     + rows)


def _teaser_reason(manifest_path: str) -> str:
    try:
        with open(manifest_path, encoding="utf-8") as f:
            return str(json.load(f).get("reason") or "").strip()
    except (OSError, ValueError):
        return ""


def window_card_lines(w_eps: List[str], montage: List[Dict[str, Any]], *,
                      banned: str = "", max_lines: int = 3) -> List[str]:
    """The card's words from the teaser WINDOW: the montage's own lines first
    (they are what the teaser shows), then the rest of the window's system
    panels in reading order. ORV, measured: 6 of 10 montage panels are system
    windows yet yield one line, and it names the series; the window holds 16."""
    import teaser_planner as _tp
    lines = system_card_lines(montage, max_lines=len(montage) or 1, banned=banned)
    rest = [p for p in _tp.load_bundle_panels(w_eps)
            if str(p.get("panel_kind") or "") == "system"]
    for text in system_card_lines(_attach_printed(rest),
                                  max_lines=len(rest) or 1, banned=banned):
        if text not in lines:
            lines.append(text)
    return lines[:max_lines]


# System panels a montage needs before the teaser counts as being ABOUT system
# windows. Measured on the Mini, 2026-09-21, per 10-panel montage: ORV 6, the
# tower series 2, Nano Machine 1, Infinite Evolution 1 (a TV news broadcast),
# Death Knight 0. One panel is incidental; the gap sits between 1 and 2.
# ponytail: a count, not a share -- re-measure if max_hook_panels changes.
_MIN_SYSTEM_PANELS = 2


def rank_designs(montage: List[Dict[str, Any]], banned: str = "",
                 card_lines: Optional[List[str]] = None):
    """The two hook designs to BUILD, ranked from the teaser's montage by code.

    The model never picks the design (the owner picks between built options).
    A system window is only offered when the montage holds a system panel whose
    printed text is usable -- the card QUOTES the story, it never invents. A
    transformation cue on the climax (the LAST panel) puts the headline ahead
    of the plain nametag. Returns (names, reason); reason carries the raw
    signal values so a concept can explain its own design.
    """
    import teaser_planner as _tp
    # A window is offered when the TEASER is about system windows (its montage
    # holds one) AND there is something to print in it. *card_lines* are the
    # SAME lines the build will get (the caller's window_card_lines), or the
    # ranking promises an empty window; without them, the montage's own.
    n_sys = sum(1 for p in montage
                if str(p.get("panel_kind") or "") == "system")
    if card_lines is None:
        card_lines = system_card_lines(montage, max_lines=len(montage) or 1,
                                       banned=banned)
    sys_text = card_lines if n_sys >= _MIN_SYSTEM_PANELS else []
    hits = int(_tp.score_panel(montage[-1])["transform_hits"]) if montage else 0
    reason = {"system_panels": n_sys,
              "system_share": round(n_sys / (len(montage) or 1), 3),
              "card_lines": len(card_lines),
              "climax_transform_hits": hits}
    rest = (["nametag_headline", "nametag"] if hits
            else ["nametag", "nametag_headline"])
    return ((["system_window"] if sys_text else []) + rest)[:2], reason


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
    where a protagonist is most likely to be shown at their weakest. A panel
    showing the LEAD wins over one that merely shows a person: "a person" is
    how a side character became the protagonist before.
    """
    fallback = person = ""
    for ci in range(0, max(1, min(climax_ci, len(beats_objs)))):
        # the "before" half must SHOW the character too -- picking purely on
        # intensity handed the image model a panel with no person in it, and
        # the fidelity instruction had nothing to copy (see
        # panel_shows_a_character).
        shows: Dict[str, bool] = {}
        lead = protagonist_portrait_files(ep_dirs[ci]) if ci < len(ep_dirs) else set()
        if ci < len(ep_dirs):
            try:
                u = json.load(open(os.path.join(
                    ep_dirs[ci], "manifest.panels.understood.json")))
                shows = {os.path.basename(str(p.get("scene_file") or "")):
                         panel_shows_a_character(p)
                         for p in (u.get("panels") or [])}
            except Exception:
                shows = {}
        for fn, inten in _kept_panels(beats_objs[ci]):
            if inten not in ("calm", "tense"):
                continue
            path = (os.path.join(ep_dirs[ci], "scenes", fn)
                    if ci < len(ep_dirs) else fn)
            if fn in lead:
                return path
            if shows.get(fn):
                person = person or path
            elif not shows:
                # no understanding for this chapter: no evidence either way.
                # A panel KNOWN to show nobody is never a fallback.
                fallback = fallback or path
    return person or fallback


def refs_for_style(style: str, beats_list: List[Dict[str, Any]],
                   ep_dirs: List[str], *, climax_ci: int,
                   climax_refs: List[str]) -> List[str]:
    """The reference panels a style's composition actually needs.

    before_after paints two moments, so its BEFORE half needs a panel from
    before the climax (an absolute path, usually another chapter); climax refs
    alone painted both halves from one moment. Every other style is one scene
    and uses the climax refs. The ONE place this is decided: the two-stage path
    (the production default) used to skip it and series_9 shipped a
    before_after built from three climax-chapter panels.
    """
    refs = list(climax_refs or [])
    # climax in the first chapter: there is no "before" to search, and
    # select_before_ref would search the climax chapter itself
    if style == "before_after" and climax_ci >= 1:
        before = select_before_ref(beats_list, ep_dirs, climax_ci=climax_ci)
        if before:
            refs = [before] + [r for r in refs if r != before]
    return refs


def choose_refs(style: str, beats_list: List[Dict[str, Any]],
                ep_dirs: List[str], *, climax_ci: int, auto_refs: List[str],
                picked: Optional[List[str]] = None) -> List[str]:
    """The refs a run paints from: the owner's picks, as-is, for EVERY layout
    (owner: one set of clear MC shots; the prompt makes before and after).
    Nothing picked: the automatic choice."""
    if picked:
        return list(picked)
    return refs_for_style(style, beats_list, ep_dirs, climax_ci=climax_ci,
                          climax_refs=list(auto_refs or []))


_CLOSE_UP_RE = re.compile(r"close-up|closeup|portrait|\\bface\\b", re.IGNORECASE)


def ref_candidates(ep_dirs: List[str], beats_list: Any = None, *,
                   n: int = 8, prefer_first: int = 0) -> Dict[str, Any]:
    """Reference panels to SUGGEST to the owner, who ticks 1-3 for both options.

    Owner, 2026-09-17: the first suggestions were "wierd", not clear images of
    the MC -- they were ranked by DRAMA (crowds, effects, speech bubbles), and
    split into lead/before sets the owner did not want.

    A tile is a CLEAR SOLO SHOT of the lead: a lead panel (identity + registry
    look guards in _lead_panels) with exactly one subject, no dialogue, text
    coverage <= 3%, aspect 0.5-2.0 and at least 600px wide. Ranked by registry
    look, then close-up/face, then size; one per chapter. Fewer than *n* is
    returned rather than padding with unclear panels. Measured on ORV: 10 such
    panels among the look-matched ones, so all lead panels are considered.
    ponytail: identity for every chapter (~1-2 min on 309), so it runs as a job.

    *prefer_first* is the teaser's window: ONLY the first N chapters are
    considered. The claim and climax come from that window, so the LOOK does
    too. It first topped the list up from later chapters; measured on ORV
    (2026-09-21) the window holds 2 clean tiles and the top-up refilled the page
    with Episodes 100-262 -- other characters, the tiles the owner had just
    rejected. Two right tiles beat two right and six wrong. 0 = every chapter.
    """
    cands: List[Dict[str, Any]] = []
    window = list(ep_dirs or [])
    if prefer_first > 0:
        window = window[:prefer_first]
    for i, d in enumerate(window):
        try:
            u = json.load(open(os.path.join(d, "manifest.panels.understood.json")))
        except Exception:
            continue
        try:
            v = json.load(open(os.path.join(d, "manifest.vision.json")))
        except Exception:
            v = {}
        vis = {os.path.basename(str(it.get("scene_file") or "")): it
               for it in (v.get("items") or [])}
        portraits, look = _lead_panels(d)
        for p in u.get("panels") or []:
            fn = os.path.basename(str(p.get("scene_file") or ""))
            it = vis.get(fn) or {}
            w, h = int(it.get("width") or 0), int(it.get("height") or 0)
            if (fn not in portraits or len(p.get("subjects") or []) != 1
                    or str(p.get("dialogue") or "").strip()
                    or float(it.get("text_coverage") or 0) > 0.03
                    or not h or not 0.5 <= w / h <= 2.0 or w < 600):
                continue
            cands.append({
                "path": os.path.abspath(os.path.join(d, "scenes", fn)),
                "chapter": i, "label": os.path.basename(d.rstrip("/")),
                "file": fn, "subjects": p.get("subjects") or [],
                "_rank": (0 if fn in look else 1,
                          0 if _CLOSE_UP_RE.search(str(p.get("description") or "")) else 1,
                          -(w * h))})
    cands.sort(key=lambda c: c["_rank"])
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for c in cands:
        if c["chapter"] in seen:
            continue
        seen.add(c["chapter"])
        out.append({k: v for k, v in c.items() if k != "_rank"})
        if len(out) >= n:
            break
    out.sort(key=lambda c: c["chapter"])
    return {"refs": out}


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
        "title": normalize_title(llm.get("title")),
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
    c["refs"] = refs_for_style(c.get("style", ""), beats_list, ep_dirs or [],
                               climax_ci=climax_ci, climax_refs=refs)
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
    ap.add_argument("--single-shot", action="store_true",
                    help="the OLD one-call path: write copy straight from the "
                         "raw narration digest. Kept for comparison; the "
                         "default now understands the story first (two calls).")
    ap.add_argument("--thumbnail-concept", default="",
                    help="the PICKED series thumbnail's concept.json: the "
                         "title reuses its status words (bundle mode)")
    ap.add_argument("--ref-candidates", action="store_true",
                    help="write SUGGESTED reference panels (json) to --out "
                         "and stop: no model call, nothing paid")
    ap.add_argument("--refs", default="",
                    help="comma-separated ABSOLUTE ref paths the owner picked "
                         "(overrides the automatic choice)")
    ap.add_argument("--design", default="", choices=[""] + sorted(HOOK_DESIGNS),
                    help="the hook DESIGN (label layer) to build; ranked "
                         "from the teaser by --rank-designs, never by the model")
    ap.add_argument("--write-claim", default="", metavar="PATH",
                    help="understand the series ONCE from its teaser window and "
                         "save the claim every card, the painted scene and the "
                         "titles read (2 local model calls, nothing paid)")
    ap.add_argument("--tone", default=DEFAULT_TONE, choices=sorted(CLAIM_TONES),
                    help="which moment of the hook the claim goes for "
                         "(thumbnail_styles.CLAIM_TONES); absurd by default")
    ap.add_argument("--claim", default="", metavar="PATH",
                    help="build the --design card from a saved claim: no model "
                         "call, so every card shares labels, scene and refs")
    ap.add_argument("--rank-designs", action="store_true",
                    help="write the two designs the teaser ranks first "
                         "(json) to --out and stop: no model call, nothing paid")
    ap.add_argument("--teaser-manifest", default="",
                    help="a PLANNED/APPROVED teaser's manifest.teaser.json; "
                         "without one the montage is computed on the fly")
    ap.add_argument("--teaser-scan-chapters", type=int, default=0,
                    help="the teaser's window: thumbnail claim, climax and "
                         "refs come from the first N chapters. 0 = off")
    ap.add_argument("--teaser-min-panels", type=int, default=4)
    ap.add_argument("--teaser-max-panels", type=int, default=10)
    ap.add_argument("--teaser-payoff-tail-frac", type=float, default=0.0)
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
        if args.ref_candidates:
            if not args.out:
                ap.error("--ref-candidates needs --out")
            cands = ref_candidates(eps, beats_list,
                                   prefer_first=args.teaser_scan_chapters)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(cands, f, ensure_ascii=False, indent=2)
            print("[ok] wrote=%s refs=%d" % (args.out, len(cands["refs"])))
            return 0
        if args.claim:
            if not (args.design and args.out):
                ap.error("--claim needs --design and --out")
            try:
                with open(args.claim, encoding="utf-8") as f:
                    concept = concept_from_claim(json.load(f), args.design)
            except (OSError, ValueError) as e:
                print("[err] cannot build %s from the claim: %s" % (args.design, e))
                return 2
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(concept, f, ensure_ascii=False, indent=2)
            print("[ok] wrote=%s design=%s hook=%r (from the claim, no model call)"
                  % (args.out, args.design, concept["hook"]))
            return 0
        # The thumbnail sells what the teaser sells: claim, climax and refs
        # come from the TEASER'S WINDOW. (ORV: teaser in chapters 3-9, the
        # thumbnail's climax 115 chapters later, in Episode 124.) The window
        # is a PREFIX, so climax_ci still indexes the caller's full list.
        montage, claim_source = None, "digest"
        if args.teaser_scan_chapters > 0:
            montage, claim_source = teaser_montage(
                eps, args.teaser_manifest, scan=args.teaser_scan_chapters,
                min_panels=args.teaser_min_panels,
                max_panels=args.teaser_max_panels,
                tail_frac=args.teaser_payoff_tail_frac)
        win = args.teaser_scan_chapters if montage else len(eps)
        w_eps, w_beats = eps[:win], beats_list[:win]
        # EVERY printable window line is offered (the owner picks the one the
        # window prints); the first three stay the default card
        card_options = (window_card_lines(w_eps, montage, banned=args.series_title,
                                          max_lines=100) if montage else [])
        card = card_options[:3]
        if args.rank_designs:
            if not montage or not args.out:
                print("[err] --rank-designs needs --out and a teaser montage "
                      "(--teaser-scan-chapters); window gave: %s" % claim_source)
                return 2
            names, why = rank_designs(montage, banned=args.series_title,
                                      card_lines=card)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                # the card LINES, not a count: the owner reviews the words a
                # system window would print BEFORE any image is paid for
                json.dump({"designs": names, "reason": why, "card": card,
                           "claim_source": claim_source}, f, indent=2,
                          ensure_ascii=False)
            print("[ok] wrote=%s designs=%s source=%s"
                  % (args.out, names, claim_source))
            for line in card:
                print("[..] card line: %s" % line)
            return 0
        if args.write_claim and not montage:
            print("[err] --write-claim needs a teaser montage "
                  "(--teaser-scan-chapters); window gave: %s" % claim_source)
            return 2
        print("[..] claim source: %s (%d of %d chapters)"
              % (claim_source, len(w_eps), len(eps)))
        durations = [_plan_duration(e) for e in eps]
        # the climax scan is exhaustive (cheap, pure Python over every
        # chapter); only the LLM DIGEST is bounded — see bundle_digest. Same
        # scorer build_bundle_concept uses, so style/digest/refs all agree on
        # which chapter is the peak.
        _sc = select_bundle_climax_scored(w_eps)
        climax_ci = (_sc[0] if _sc
                     else (select_bundle_climax(w_beats)[0] if w_beats else 0))
        style = args.style or select_style(
            w_beats[climax_ci] if w_beats else {}, genre=args.genre)
        digest = bundle_digest(w_beats, max_chapters=args.digest_chapters,
                               climax_index=climax_ci)
        if len(w_beats) > args.digest_chapters:
            print(f"[..] digest: sampled {args.digest_chapters} of "
                  f"{len(w_beats)} chapters (climax #{climax_ci + 1} kept) "
                  f"— {len(digest):,} chars")
        def _lead_refs(layout: str) -> List[str]:
            auto_refs = _sc[1] if _sc else select_bundle_climax(w_beats)[1]
            if montage:
                # WHO to draw: the same clean-solo-shot rule the suggestion tiles
                # use, over the teaser's window (absolute paths). The climax
                # chapter's "lead" panels were, on ORV, the back of a head, a
                # half profile and a close-up of someone else.
                clean = [r["path"] for r in ref_candidates(w_eps, n=3)["refs"]]
                if clean:
                    auto_refs = clean
                print("[..] lead refs: %s" % ("clean solo shots from the window"
                                              if clean else "climax-chapter "
                                              "panels (no clean solo shot found)"))
            return choose_refs(layout, w_beats, w_eps, climax_ci=climax_ci,
                               auto_refs=auto_refs,
                               picked=[r for r in args.refs.split(",") if r.strip()])

        if args.write_claim:
            # ONE understanding per series. Every card is then built from this
            # file with no further model call (concept_from_claim).
            hook_block = teaser_hook_block(montage, _teaser_reason(args.teaser_manifest))
            brief = _gemma(build_brief_prompt(digest, args.series_title,
                                              hook_block=hook_block),
                           args.ollama_model)
            print("[..] brief: %s" % str(brief.get("premise") or "")[:110])
            pkg = _gemma(build_claim_prompt(brief, args.series_title,
                                            tone=args.tone),
                         args.ollama_model)
            corpus = beats_text_corpus(w_beats[climax_ci] if w_beats else {})
            grounded = lambda raw: [x for x in (t.replace("|", " -> ")
                                                for t in _clean_labels(raw))
                                    if hook_is_grounded(x, corpus)]
            names, why = rank_designs(montage, banned=args.series_title,
                                      card_lines=card)
            synopsis = str(pkg.get("description") or "").strip()
            hashtags = pkg.get("hashtags") or ["#manhwa", "#manga", "#manhwarecap"]
            claim = {
                "title": normalize_title(pkg.get("title")),
                "style": SCENE_STYLES[0], "brief": brief,
                "scene": re.sub(r"\s+", " ", str(pkg.get("scene") or "")).strip(),
                "labels": grounded(pkg.get("labels")),
                "headlines": grounded(pkg.get("headlines")),
                "card": card, "card_options": card_options, "tone": args.tone,
                "designs": names, "design_reason": why,
                "claim_source": claim_source,
                "teaser_panels": [str(p.get("scene_file") or "") for p in montage],
                "climax_chapter_index": climax_ci,
                "refs": _lead_refs(SCENE_STYLES[0]),
                "badge": "%d CHAPTERS" % len(beats_list),
                "synopsis": synopsis, "hashtags": hashtags,
                "description": build_description(synopsis, hashtags),
                "pinned_comment": pinned_comment(args.series_title,
                                                 args.official_link)}
            os.makedirs(os.path.dirname(os.path.abspath(args.write_claim)),
                        exist_ok=True)
            with open(args.write_claim, "w", encoding="utf-8") as f:
                json.dump(claim, f, ensure_ascii=False, indent=2)
            print("[ok] wrote=%s designs=%s labels=%d scene=%r"
                  % (args.write_claim, names, len(claim["labels"]),
                     claim["scene"][:90]))
            return 0

        if args.single_shot:
            llm = _gemma(build_concept_prompt(digest, args.series_title, style),
                         args.ollama_model)
            concept = build_bundle_concept(
                beats_list, llm, durations=durations,
                series_title=args.series_title, genre=args.genre,
                official_link=args.official_link, ep_dirs=eps, style=style)
        else:
            # STAGE 1 — understand the series, STAGE 2 — write from that.
            brief = _gemma(build_brief_prompt(
                digest, args.series_title,
                hook_block=teaser_hook_block(
                    montage or [], _teaser_reason(args.teaser_manifest))),
                args.ollama_model)
            print("[..] brief: %s" % str(brief.get("premise") or "")[:110])
            thumb_labels = ""
            if args.thumbnail_concept and os.path.exists(args.thumbnail_concept):
                tc = json.load(open(args.thumbnail_concept))
                thumb_labels = thumb_label_words(tc)
            pkg = _gemma(
                build_package_prompt(brief, args.series_title,
                                     [args.style] if args.style else None,
                                     thumb_labels=thumb_labels,
                                     design=args.design),
                args.ollama_model)
            style_beats = w_beats[climax_ci] if w_beats else {}
            concept = assemble_package(
                style_beats, brief, pkg, series_title=args.series_title,
                official_link=args.official_link,
                styles=[args.style] if args.style else None,
                design=args.design,
                card_lines=(card if args.design == "system_window" else None))
            if args.design == "system_window" and not concept.get("card"):
                # an empty left third is not a system window: refuse before
                # paying for the image
                print("[err] system_window needs a printable system line in the "
                      "teaser window; none survived (OCR junk or the banned title)")
                return 2
            concept["climax_chapter_index"] = climax_ci
            concept["claim_source"] = claim_source
            if montage:
                concept["teaser_panels"] = [str(p.get("scene_file") or "")
                                            for p in montage]
                concept["design_reason"] = rank_designs(
                    montage, banned=args.series_title, card_lines=card)[1]
            concept["refs"] = _lead_refs(concept["style"])
            if concept["style"] == "before_after" and not (
                    concept["refs"] and os.path.isabs(concept["refs"][0])):
                # both halves would be painted from the climax: refuse before
                # paying for the image (and say which chapter was the climax)
                print("[err] before_after needs a panel of the lead from BEFORE "
                      "the climax (chapter #%d of %d); none found"
                      % (climax_ci + 1, len(beats_list)))
                return 2
            concept["parts"] = parts_timestamps(durations)
            if beats_list:
                concept["badge"] = "%d CHAPTERS" % len(beats_list)
            concept["description"] = (concept["description"] + "\n\n"
                                      + "\n".join(concept["parts"]))
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

    if len(concept.get("title") or "") > 100:
        # YouTube rejects titles over 100 characters. Uploads are manual, so
        # say it loudly rather than cut the claim mid-sentence.
        print("[warn] title is %d chars, YouTube allows 100: %r"
              % (len(concept["title"]), concept["title"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(concept, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote={out} style={concept['style']} "
          f"hook={concept['hook']!r} title={concept['title']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
