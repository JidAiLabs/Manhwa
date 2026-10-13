#!/usr/bin/env python3
"""
scene_chrome.py — deterministic detection of chapter CHROME scenes.

Chrome = pages that belong to the publication, not the story: publisher and
studio logo pages, series cover/title pages, chapter-number cards, app UI
(view counters), author/translator credits. Chrome must never be grouped,
narrated, or shown — the ORV/IE openings narrated "presented by Redice
Studio" and "VIEWS: 1" because nothing modeled this concept.

Used by scene_group_builder (exclude before any LLM sees them) and
render_prep (belt-and-suspenders cut gate for already-processed chapters).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# UI counters / app chrome — "VIEWS: 1", "LIKES 203", subscriber counts.
# The value class includes I/l/| /O: Vision routinely OCRs digits as letters
# ("VIEWS: I" is the real ORV panel that slipped the first filter pass).
_COUNTER_RE = re.compile(r"\b(views?|likes?|subscribers?|comments?)\s*[:.]?\s*[\dIlO|,.]+", re.I)

# Publication credits and platform names.
_CREDITS_RE = re.compile(
    r"\b(studio|publisher|published\s+by|presented\s+by|author|art\s+by|story\s+by|"
    r"adaptation|translat\w*|proofread\w*|typeset\w*|scanlat\w*|copyright|"
    r"all\s+rights|webtoon|naver|kakao|redice)\b|©",
    re.I,
)

# Site plugs and scanlation-team credits (aggregator stamps on covers).
# Domains and team-credit tags are chrome no matter how wordy the page
# (the IE cover OCRs ~58 words around ELFTOON.COM). OCR often splits the
# domain ("ELFTOON .com" on the IE end card) — tolerate spaces around the
# dot. Bare plug PHRASES are chrome only on short banners: story dialogue
# legitimately says "read this" (the ORV novel-app panel).
_SITE_HARD_RE = re.compile(
    r"\b\w[\w-]*\s?\.\s?(com|net|org|io|to|gg)\b|\b(ed|tl|pr|qc|clrd|rd)\s*:",
    re.I,
)
_SITE_PLUG_RE = re.compile(
    r"please\s+read|read\s+(this|free)|thanks\s+for\s+reading|"
    r"join\s+our\s+discord|our\s+(web\s?site|discord)|discord\s+server",
    re.I,
)
_SITE_PLUG_MAX_WORDS = 12

_MARKER_RE = re.compile(
    r"^\s*(?:chapter|episode|ep\.?|prologue|final(?:e)?|season\s*\d+)?[\s:#-]*\d{0,4}\s*$",
    re.I,
)

_WORD_RE = re.compile(r"[a-z0-9]+")


_KR_CREDITS_RE = re.compile(r"그림|각색|원작|작가|연출")


def _norm_words(s: str) -> list:
    return _WORD_RE.findall((s or "").lower())


# function words a title shares with any sentence — never evidence of the logo
# ('of' alone was a story panel's whole OCR in TT ch138; 'THE 00' in DK ch2)
_TITLE_STOP = frozenset({"the", "of", "a", "an", "and", "in", "from", "to", "s",
                         "is", "for", "on", "my", "i"})


def is_licensed_title_scene(item: dict, series_title: str) -> bool:
    """True when the panel's OCR IS the licensed series title (the chapter logo /
    title card): at least 60% of the title's distinctive words are present and at
    most ONE word is not a title word (one OCR-garbled letter run — 'TEADVANCED',
    'JUTORIAL' — is how a real logo reads). Measured 2026-10-05 over 12 series:
    every Tutorial Tower logo (65/65, three OCR garbles) matches; no in-world
    dialogue or card does ('NANO MACHINE? YES, MASTER.', 'THE TOWER COLLAPSES...').
    ponytail: a bubble that says ONLY a two-word title ('NANO MACHINE') would match
    — zero in the corpus; add a punctuation/bubble check if one ever appears."""
    distinct = {w for w in _norm_words(series_title) if w not in _TITLE_STOP and len(w) >= 3}
    words = _norm_words(str(item.get("ocr_clean") or ""))
    if not distinct or not words:
        return False
    title_words = set(_norm_words(series_title))
    covered = len(set(words) & distinct) / len(distinct)
    return covered >= 0.6 and sum(1 for w in words if w not in title_words) <= 1


def needs_image_stats(ocr: str) -> bool:
    """Callers should compute midtone_frac for these OCR signatures: empty
    OCR (stylized number cards), a site hit that may be a mere watermark, or a
    'To Be Continued' that may be stamped over real art."""
    ocr = (ocr or "").strip()
    return (not ocr or bool(_SITE_HARD_RE.search(ocr))
            or bool(_END_CARD_RE.search(ocr)))


def item_midtone(item: Dict[str, Any]) -> Optional[float]:
    """The share of mid-grey pixels (60 < L < 200) of the item's scene image —
    real art has midtones, a flat card has none — computed only for the OCR
    signatures that need it (needs_image_stats); None otherwise or on any error."""
    if (not needs_image_stats(str(item.get("ocr_clean") or ""))
            or not item.get("scene_path")):
        return None
    try:
        from PIL import Image
        import numpy as np
        im = np.asarray(Image.open(item["scene_path"]).convert("L"))
        return float(((im > 60) & (im < 200)).mean())
    except Exception:
        return None


def furniture_files(vision_items) -> set:
    """scene_files that ARE publication furniture (furniture_reason) — OR-ed into
    every exclusion AFTER the understanding's rescues, like the licensed title."""
    return {it.get("scene_file") for it in vision_items or []
            if it.get("scene_file")
            and furniture_reason(it, midtone_frac=item_midtone(it))}


# ---------------------------------------------------------------------------
# PUBLICATION STAMPS (owner 2026-10-13). Aggregator/scanlator text printed ON the
# page: a watermark over story art ("KIM SHI HYUN! ASURASCANS.COM"; FTA stamps
# "YOU CAN READ THE CHAPTER ON: EN-THUNDERSCANS.COM" across its art), a release
# plug, a recruitment card, a credits roster, a "To Be Continued" end card. The
# narrator voiced them and named a hero "Asura" after the watermark. ONE
# vocabulary for every consumer: vision_extract scrubs stamps out of ocr_clean
# at the source (the stamps go to item["watermarks"]); furniture_reason decides
# whether the PANEL is the message (is_chrome_scene, panel_understand);
# cast_builder never backs a name with a stamp; prep_qa + qa_rules_sweep catch a
# voiced one (voiced_furniture). Measured on the 2026-10-13 census (91,346
# panels): real OCR shapes and in-world look-alikes are pinned in
# tests/test_publication_stamps.py.
# ---------------------------------------------------------------------------
_END_CARD_RE = re.compile(r"\bto\s+be\s+continue[ds]?\b\s*(?:\.{2,}|…)?", re.I)

# "ASURASCANS" as OCR reads it: ASURAS CANS, ASURA S CANS, ASURASCAMS, ASURASGANS,
# ASURASCEANS, ASURAS CAUS, ASURASCAINS, ASURASCAL… (census 2026-10-13) — the
# second half is (c|g|s|t|n|d|3) + a vowel run + (n|m|u|i|v|l); "Asura screams" /
# "Asura grits" (narration verbs) are not it.
_ASURA_SCANS = r"asura\s?s?\s?[cgsdtn3]\s?[ea]?a[nmuivl]{1,2}s?"

# (kind, pattern) in application order: the specific aggregator spellings before
# the generic domain (else "MANGA YY.ORG" leaves "MANGA"), the generic domain
# before the generic "...scans" name (else "EN-THUNDERSCANS.COM" leaves "EN-").
_STAMPS = [
    ("recruitment", re.compile(
        r"\bwe'?\s?re\s+(?:now\s+)?recruiting\b[^!.]*[!.]?"
        r"|\bjoin\s+our\s+discord(?:\s+to\s+apply)?\s*-?", re.I)),
    ("release_plug", re.compile(
        r"\bread\s+at\b[\s.…]*(?:\S+\s+)?(?:fo)?r\s+the\s+fastest\s+releases?\b\.?"
        r"|\b(?:fo)?r\s+the\s+fastest\s+releases?\b\.?"
        r"|\bplease\s+read\s+this\s+chapter\s+on\b"
        r"|\bread\s+on\s+our\s+website\b!?|\bon\s+our\s+ad-?free\s+website\b:?"
        r"|\bsupport\s+the\s+team\s+behind\s+the\s+translation\b"
        r"|\bthis\s+chapter\s+is\s+provided\s+by\b\s*:?(?:\s+\w+)?"
        r"|\b(?:we\s+have\s+)?more\s+chapters\s+of\s+this\s+series(?:\s+are)?\b", re.I)),
    # a plug STAMPED across story art (FTA: "YOU CAN READ THE CHAPTER ON: …" every few
    # panels; Clan: "READ ONLY AT MADARASCANS.COM" on 60+ story panels) — scrubbed,
    # never proof the panel is an ad
    ("read_plug", re.compile(
        r"(?:\byou\s+)?(?:\b(?:c?an|n)\s+)?\bread\s+the\s+chapter\s+on[il]?\b\s*:?"
        r"|\bread\s+only\s+at\b"
        r"|\bread\s+(?:it\s+)?on\s+(?=\(?[\w-]*scans\b)", re.I)),
    ("end_card", _END_CARD_RE),
    ("credit", re.compile(
        r"\bredice\s+(?:studio|tudio)\b"
        r"|\bpublished\s*by\b(?:\s+\S+){0,2}?\s+riverse\b(?:\s+inc\b\.?)?"
        r"|\bedited\s+by\s+\w+", re.I)),
    ("aggregator", re.compile(
        r"\b(?:" + _ASURA_SCANS + r"|asura\s?(?:comic|toon)s?|elft[o0d][o0]n"
        r"|manga\s?yy|[a-z]*read)\b(?:\s?\.?\s?(?:com|net|org|gg)\b)(?:\s?/\s?[\w-]+)*"
        r"|\b(?:" + _ASURA_SCANS + r"|asura\s?(?:comic|toon)s?|elft[o0d][o0]n"
        r"|manga\s?yy)\b"
        r"|(?:https?\s*:\s*[/j]+\s*)?\bdisco[rd]{1,2}\s?\.\s?gg\b(?:\s?/\s?[\w-]+)*"
        r"|\basura\s?[.,•]?\s?(?:g\s?g|[6i]\s?g{1,2}|co(?:mic)?)\b"
        r"|\basura\s?\.\s?g[o0]\b", re.I)),          # 'ASURA. GO': OCR of ASURA.GG
    # a site address; never an e-mail ("TLS123@NAVER.COM" is in-world mail in ORV),
    # and no space after the dot: "...LEAVE. COM" is a sentence end beside a split
    # watermark fragment, and the word is dialogue (census 2026-10-13: THEM. COM,
    # BANK. COM, PARTICIPANT. ORG)
    ("domain", re.compile(
        r"(?:https?\s*:\s*[/j]+\s*)?(?<![@\w.])[a-z0-9][\w-]*\s?\.(?:com|net|org|gg)\b"
        r"(?:\s?/\s?[\w-]+)*", re.I)),
    ("aggregator", re.compile(
        r"\b[a-z]{3,}scans\b(?:\s?\.?\s?(?:com|net|org|gg)\b)?|\b(?!web)[a-z]{5,}toon\b", re.I)),
]
# words that are publication furniture only NEXT TO a real stamp: "ARCHITECT OF
# DISCORD" and a cult's "JOIN US!" are in-world (ORV, census)
_WEAK_STAMPS = re.compile(
    r"\bjoin\s+us!|\bwebsite\b|\bdiscord\b|\bscans\b|\bbetter\s+quality\b|\bmore\s+content\b"
    r"|\bto\s+\w{3,5}\s+the\s+remaining\s+chapters\b",
    re.I)
_SEP = r"[\s•›‹>→@~|·:\-()]*"   # the bullets/arrows the scanlators print before a stamp
_HOLE = "\x00"


def scrub_stamps(text: str) -> tuple:
    """(text without publication stamps, [{"kind", "text"}, ...]). Story text,
    in-world system text and in-world names ('VENERABLE ASURA') are untouched."""
    out = text or ""
    stamps = []

    def _cut(kind, pat):
        nonlocal out

        def _rep(m):
            stamps.append({"kind": kind, "text": m.group(0).strip()})
            return _HOLE
        out = pat.sub(_rep, out)

    for kind, pat in _STAMPS:
        _cut(kind, pat)
    if stamps:
        _cut("weak", _WEAK_STAMPS)
        out = re.sub(_SEP + _HOLE + "(?:" + _SEP + _HOLE + ")*" + _SEP, " ", out)
        out = re.sub(r"\s+", " ", out).strip()
    return out, stamps


_LETTER_WORD_RE = re.compile(r"[^\W\d_]{2,}")
_ROLE_WORDS_RE = re.compile(
    r"\b(translator|proofreader|cleaner|redrawer|typesetter|quality\s+checker|"
    r"raw\s+provider|editor)s?\b", re.I)
_ROLE_TAGS_RE = re.compile(r"\b(TL|PR|TS|QC|RD|CL|CD|ED)\b")   # case-sensitive roster tags
_KR_ROLES_RE = re.compile(r"그림|각색|원작|작가|연출|채색|편집|콘티|보정|식자")


def furniture_reason(item: Dict[str, Any], *,
                     midtone_frac: Optional[float] = None) -> Optional[str]:
    """Why this panel IS publication furniture ('credits' | 'recruitment' |
    'release_plug' | 'end_card'), or None. A stamp on a story panel never makes it
    furniture — the panel must BE the message. Measured 2026-10-13 on
    1,969 stamped panels (tests/test_publication_stamps.py holds the shapes):
      - a credits roster / recruitment notice is always its own page;
      - the Asura "READ AT … FOR THE FASTEST RELEASES" card leaves NO word behind,
        while the same plug under a story panel leaves its dialogue ("GO, CHILD!!") —
        image flatness can't tell them apart (the card's glow measures 0.18-0.25,
        stamped art 0.14+), the words left can;
      - a stamp on wordless art ("READ ONLY AT MADARASCANS.COM", the Redice credit
        line over ORV's last panel) stays story, even on a dark panel (Clan ch11
        p000057, a door creaking open, midtone 0.05 — checked by eye 2026-10-13).
    Known limit: a crop that merges the last story panel with an end card (Wimp ch36
    p000116, TT ch36 p000090 — 2 of 70 flips) goes as a whole.
    Reads item["watermarks"] when vision_extract recorded them, else scrubs
    ocr_clean on the fly (manifests made before 2026-10-13)."""
    ocr = str(item.get("ocr_clean") or "")
    stamps = item.get("watermarks")
    if stamps is None:
        clean, stamps = scrub_stamps(ocr)
    else:
        clean = ocr
    full = " ".join([ocr] + [str(s.get("text") or "") for s in stamps])
    if (len(set(_KR_ROLES_RE.findall(full))) >= 2
            or len({m.lower() for m in _ROLE_WORDS_RE.findall(full)}) >= 2
            or len(set(_ROLE_TAGS_RE.findall(full))) >= 3):
        return "credits"
    kinds = {s.get("kind") for s in stamps}
    left = len(_LETTER_WORD_RE.findall(clean))
    if "recruitment" in kinds and left <= 3:
        return "recruitment"
    if "release_plug" in kinds and left == 0:
        return "release_plug"
    if ("end_card" in kinds and left <= 2
            and not (midtone_frac is not None and midtone_frac >= 0.15)):
        return "end_card"
    return None


def watermark_names(names, vision_items) -> set:
    """The proper names in *names* whose words the chapter prints ONLY inside
    publication stamps — never in a non-chrome panel's story text. Boundless
    Necromancer's hero was narrated as "Asura": its only "Asura" is the ASURA.GG
    watermark (2026-10-13). Descriptive handles ('our protagonist') are skipped."""
    page, stamped = [], []
    for it in vision_items or []:
        ocr = str(it.get("ocr_clean") or "")
        marks = it.get("watermarks")
        if marks is None:
            ocr, marks = scrub_stamps(ocr)
        stamped += [str(m.get("text") or "") for m in marks]
        if str(it.get("panel_kind") or "") != "chrome":
            page.append(ocr)
    page_l, stamp_l = " ".join(page).lower(), " ".join(stamped).lower()
    out = set()
    for name in names or []:
        toks = [t for t in re.split(r"[^a-z0-9]+", str(name or "").lower()) if len(t) >= 3]
        if (not toks or not str(name)[:1].isupper()
                or str(name).lower().startswith(("the ", "our ", "a ", "an "))):
            continue
        backed = all(re.search(r"\b" + re.escape(t) + r"\b", page_l) for t in toks)
        if not backed and any(t in stamp_l for t in toks):
            out.add(str(name))
    return out


_END_FILLER_RE = re.compile(
    r"^\W*(?:the\s+(?:story|scene)\s+(?:will\s+)?continues?|to\s+be\s+continued)\W*$", re.I)
_RECRUIT_LINE_RE = re.compile(
    r"\brecruit\w*\b[^.!?]{0,40}?\b(?:translators?|typesetters?|proofreaders?|"
    r"cleaners?|redrawers?)\b", re.I)


def voiced_furniture(line: str) -> Optional[str]:
    """Why a NARRATION line voices publication furniture, or None: a stamp in it
    (a site, a plug, a credit), an end-card filler ('The story will continue.'),
    or a recruitment notice. In-world look-alikes ('the Architect of Discord',
    'read at a faster speed', 'Venerable Asura') pass."""
    if not line:
        return None
    if _END_FILLER_RE.match(line):
        return "end_card"
    _, stamps = scrub_stamps(line)
    if stamps:
        return f"{stamps[0]['kind']}: {stamps[0]['text']}"
    if _RECRUIT_LINE_RE.search(line):
        return "recruitment"
    return None


def is_chrome_scene(
    item: Dict[str, Any],
    *,
    series_title: Optional[str] = None,
    midtone_frac: Optional[float] = None,
) -> bool:
    """True when a vision item's scene is publication chrome, not story.

    Rules (any hit = chrome):
      1. UI counters dominate the text (VIEWS/LIKES + numbers, including
         OCR digit-confusions like 'VIEWS: I').
      2. Credits/publisher/platform vocabulary present.
      3. The OCR is nothing but a chapter/episode/number marker.
      4. The OCR is dominated by the series title (cover/title page).
      5. OCR-BLIND chrome (stylized number cards): empty OCR + a near-binary
         pixel profile (*midtone_frac*, supplied by callers with image
         access) — real art always has midtones.

    The multimodal UNDERSTANDING is authoritative when present: if the vision item
    carries a panel_kind (stamped onto the vision manifest by panel_understand),
    a 'story'/'caption'/'system' verdict is NEVER chrome (whatever the OCR says)
    and 'chrome' is chrome. This is the SINGLE source of truth — story_group,
    render_prep and prep_qa all reach chrome detection through here, so none of
    them re-derive it from OCR and disagree (e.g. the beast panel whose only OCR
    was '1'). 'system' = in-world game/UI cards (quest/status/notification/alarm/
    system-message) — always PLOT, never decorative chrome.
    """
    # Rule 0 — HARD RULE, above the understanding: the licensed series title/logo is
    # never shown and never voiced. The model read Tutorial Tower's chapter logo as a
    # 'system' card (ch151) / 'story' (ch167) and the writer voiced it; the model
    # cannot know what is licensed, so this is decided before its verdict (2026-10-05).
    if series_title and is_licensed_title_scene(item, series_title):
        return True
    # Rule 0b — also above the understanding: a panel that IS publication furniture
    # (an ad, a recruitment card, a credits roster, a To-Be-Continued card). The
    # model called the Asura ad 'system' and the end card 'story' (TT ch177, voiced
    # 2026-10-02); bad7838f lifted only the title rule above the label (2026-10-13).
    if furniture_reason(item, midtone_frac=midtone_frac):
        return True
    kind = str(item.get("panel_kind") or "").strip().lower()
    if kind in ("story", "caption", "system"):
        return False
    if kind == "chrome":
        return True
    ocr = str(item.get("ocr_clean") or "").strip()
    if not ocr:
        # pure art — chrome only when image stats prove a binary card
        return midtone_frac is not None and midtone_frac < 0.08

    if _COUNTER_RE.search(ocr):
        # counters are chrome when they ARE the content, not one mention in
        # a long dialogue line
        counter_hits = len(_COUNTER_RE.findall(ocr))
        words = _norm_words(ocr)
        if counter_hits * 2 >= max(1, len(words)) / 2 or len(words) <= 6:
            return True

    if _CREDITS_RE.search(ocr):
        return True

    # Korean staff credits on title/end cards (그림=art, 각색=adaptation,
    # 원작=original, 작가=author, 연출=direction) — the Latin credit regexes
    # are blind to hangul, which let the 나노마신 title card through. Two
    # distinct roles on one card = credits, certainly chrome.
    if len(set(_KR_CREDITS_RE.findall(ocr))) >= 2:
        return True

    hard_hits = sum(1 for _ in _SITE_HARD_RE.finditer(ocr))
    plug = bool(_SITE_PLUG_RE.search(ocr))
    if hard_hits >= 2:
        return True  # domains + team-credit tags pile up on real covers
    if hard_hits == 1:
        if plug:
            return True  # domain + "thanks for reading"/"join our" = end card
        # ONE domain amid real dialogue is an aggregator watermark stamped ON
        # story art (IE p000039) — chrome only when the panel is otherwise
        # text-sparse AND image stats don't prove real art
        if (len(_norm_words(ocr)) <= _SITE_PLUG_MAX_WORDS
                and not (midtone_frac is not None and midtone_frac >= 0.15)):
            return True

    if plug and len(_norm_words(ocr)) <= _SITE_PLUG_MAX_WORDS:
        return True

    if _MARKER_RE.match(ocr):
        return True

    if series_title:
        title_words = set(_norm_words(series_title))
        words = _norm_words(ocr)
        if title_words and words:
            in_title = sum(1 for w in words if w in title_words)
            if in_title / len(words) >= 0.6 and len(words) <= len(title_words) + 3:
                return True
            # vertical/stylized covers OCR as garbage tokens around the one
            # distinctive title word ("SR OMNISCIENT CE IA ED NEO TR")
            big = [w for w in title_words if len(w) >= 8]
            if big and len(words) <= 10 and any(w in words for w in big):
                return True

    return False
