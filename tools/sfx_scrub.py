"""Sound-effect / onomatopoeia detection + scrub.

Webtoon bubbles often carry pure SFX/screams (EUAACK!!, ACK!!!, KEUK, "HUH...
HUH?!") and garbled OCR. Voicing these verbatim is absurd. This module decides
whether a quoted span is SFX (no real content word) and removes such spans from
narration. Shared by the verbatim narration scrub (script_expander) and the
prep_qa `sfx_voiced` verifier flag.

A two-tier guard avoids over-firing on real short interjections: a quote is SFX
only when it contains NO real content word (a token that isn't onomatopoeia and
has >=3 alpha chars), or it is pure scream punctuation.
"""
from __future__ import annotations
import re

# common webtoon SFX/scream tokens (lowercased, punctuation stripped)
_SFX_WORDS = {
    "euaack", "ack", "acck", "keuk", "ugh", "hng", "ngh", "gah", "argh", "grr",
    "hoh", "huh", "hehe", "haha", "kya", "aah", "ahh", "ooh", "eek", "tch", "tsk",
    "hmph", "sob", "gasp", "thud", "boom", "bang", "clang", "crash", "whoosh",
    "gulp", "kaboom", "fwoosh", "swish", "thwack", "krak", "pow", "zap", "nin",
}
_VOWELS = "aeiou"


def _is_sfx_token(tok: str) -> bool:
    w = re.sub(r"[^a-z]", "", tok.lower())
    if not w:
        return False
    if w in _SFX_WORDS:
        return True
    if re.search(r"(.)\1\1", w):                 # 3+ same letter in a row (aaack)
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", w):  # 4+ consonant run (keuk)
        return True
    if re.search(r"[aeiou]{3,}", w):             # 3+ vowel run (euaa)
        return True
    if not any(c in _VOWELS for c in w) and len(w) <= 4:  # no vowel, short (grr)
        return True
    return False


def is_sfx_quote(q: str) -> bool:
    """True when a quoted span carries no real spoken content (pure SFX/garble)."""
    toks = re.findall(r"[A-Za-z']+", q)
    if not toks:
        return bool(re.search(r"[!?]{2,}", q) or q.strip())  # pure punctuation/garble
    real = [t for t in toks
            if not _is_sfx_token(t) and len(re.sub(r"[^a-z]", "", t.lower())) >= 3]
    return len(real) == 0


_QUOTE_RE = re.compile(r'(["“‘’”])(.+?)(["“‘’”])')


def sfx_quotes(text: str) -> list:
    """The SFX quoted spans found in *text* (for the verifier flag)."""
    return [m.group(2) for m in _QUOTE_RE.finditer(text) if is_sfx_quote(m.group(2))]


def scrub_sfx_quotes(text: str) -> str:
    """Remove pure-SFX quoted spans + obvious dangling lead-ins, conservatively.
    Real spoken quotes are left untouched. (The proper fix is to re-narrate the
    beat from panels; this keeps a clean line for the verbatim path meanwhile.)"""
    # remove an optional lead-in (of/like/saying/colon/comma) TOGETHER with the
    # SFX quote, so "cries of \"EUAACK\" as he fell" -> "cries as he fell".
    full = re.compile(r'(?:\s*\b(?:of|like|saying)\b|\s*[:,])?\s*'
                      r'(["“‘’”])(.+?)(["“‘’”])')
    out = full.sub(lambda m: "" if is_sfx_quote(m.group(2)) else m.group(0), text)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*\.", ".", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\.\s*\.", ".", out)
    return out


if __name__ == "__main__":   # smoke check
    assert is_sfx_quote("EUAACK...!! ACK!!! ACCK!!!")
    assert is_sfx_quote("HUH... HUH?!")
    assert is_sfx_quote("Keuk...!")
    assert not is_sfx_quote("Kill him!")
    assert not is_sfx_quote("How dare they dishonor my mother")
    s = scrub_sfx_quotes('He let out desperate cries of "EUAACK...!! ACK!!!" as he fell.')
    assert "EUAACK" not in s and "ACK" not in s, s
    assert "Kill him" in scrub_sfx_quotes('The order rang out: "Kill him!"')
    print("sfx_scrub smoke OK:", repr(s))
