"""The chapter's own pronoun decides gender — the drawing cannot.

ORV Ep107: the Beast Lord is drawn with long flowing hair and described as
"a figure ... with visible red blood stains on THEIR torso", so
cast_identity._gender derived nothing, the identity gender veto went inert,
and the narration called her "he" for a whole chapter — while the caption on
p000002 reads "but SHE was up against flames of hell".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import cast_builder as cb                                    # noqa: E402
import gemini_narrative_pass as gnp                          # noqa: E402
from cast_identity import cast_profiles                      # noqa: E402

NEUTRAL = ("A figure with long, flowing dark hair and tattered clothing "
           "with visible red blood stains on their torso.")


def test_story_gender_is_stamped_onto_the_cast_and_beats_a_lock():
    cast = {"cast": [
        {"canonical_name": "Beast Lord", "aliases": []},
        {"canonical_name": "our protagonist", "aliases": ["Kim Dokja"]},
        {"canonical_name": "Owner Locked", "aliases": [], "gender": "male"},
        {"canonical_name": "Unmentioned", "aliases": []},
    ]}
    n = cb.stamp_story_gender(cast, [
        {"name": "Beast Lord", "gender": "female"},
        {"name": "Kim Dokja", "gender": "male"},          # matched by ALIAS
        {"name": "Owner Locked", "gender": "female"},     # lock wins
        {"name": "Nobody", "gender": "female"},
    ])
    got = {m["canonical_name"]: m.get("gender") for m in cast["cast"]}
    assert n == 2
    assert got == {"Beast Lord": "female", "our protagonist": "male",
                   "Owner Locked": "male", "Unmentioned": None}


def test_a_neutral_description_derives_nothing_but_the_field_carries():
    bare = cast_profiles({"cast": [
        {"canonical_name": "Beast Lord", "visual_description": NEUTRAL}]})[0]
    assert bare["gender"] is None                 # this is how she slipped past
    stamped = cast_profiles({"cast": [
        {"canonical_name": "Beast Lord", "visual_description": NEUTRAL,
         "gender": "female"}]})[0]
    assert stamped["gender"] == "female"
    junk = cast_profiles({"cast": [
        {"canonical_name": "Beast Lord", "visual_description": NEUTRAL,
         "gender": "unknown"}]})[0]
    assert junk["gender"] is None                 # only the enum is trusted


def test_the_writer_is_told_the_gender_it_must_not_guess(tmp_path):
    import json
    cp = tmp_path / "manifest.cast.json"
    cp.write_text(json.dumps({"cast": [
        {"id": "beast_lord", "canonical_name": "Beast Lord",
         "role": "antagonist", "visual_description": NEUTRAL,
         "gender": "female"},
        {"id": "stranger", "canonical_name": "Stranger", "role": "extra",
         "visual_description": "a tall figure"},
    ]}))
    lines = {ln.split("(")[0].strip(" -"): ln
             for ln in gnp._build_cast_block(str(cp)).splitlines()
             if ln.startswith("  - ")}
    assert "[female]" in lines["Beast Lord"]
    # an unmarked member is left unmarked -- the writer must never be handed a
    # guessed gender, which is the whole failure this fixes
    assert "[" not in lines["Stranger"].split(":", 1)[0]


# ---- the page's own pronouns (deterministic, no model) ----------------------

def test_gender_is_read_off_the_same_ocr_line():
    """ORV Ep107 p000002 names her AND uses the pronoun in one line, so this
    needs no coreference. Asking the story pass instead resolved 13% of the
    corpus and missed her; this resolves 26% and gets her."""
    cast = {"cast": [{"canonical_name": "Beast Lord", "aliases": []}]}
    n = cb.gender_from_ocr_pronouns(cast, [
        {"ocr_clean": "A BEAST LORD USUALLY DOESN'T DIE FROM A WOUND LIKE "
                      "THAT, BUT SHE WAS UP AGAINST FLAMES OF HELL."}])
    assert n == 1 and cast["cast"][0]["gender"] == "female"


def test_a_line_naming_two_members_casts_no_vote():
    """The pronoun has more than one candidate, so it proves nothing."""
    cast = {"cast": [{"canonical_name": "Beast Lord", "aliases": []},
                     {"canonical_name": "Captain", "aliases": []}]}
    assert cb.gender_from_ocr_pronouns(cast, [
        {"ocr_clean": "THE CAPTAIN AND THE BEAST LORD SPEAK; SHE NODS."}]) == 0
    assert all("gender" not in m for m in cast["cast"])


def test_an_unambiguous_gendered_noun_in_the_name_vetoes_the_vote():
    """The three measured corpus errors were all this shape -- 'Lady Hwa' ->
    male, 'King of Beauty' -> female, 'unnamed mother' -> male."""
    cast = {"cast": [{"canonical_name": "Lady Hwa", "aliases": []}]}
    assert cb.gender_from_ocr_pronouns(
        cast, [{"ocr_clean": "LADY HWA, HE IS ALREADY GONE."}]) == 0
    assert "gender" not in cast["cast"][0]


def test_lord_is_not_a_vetoing_noun():
    """'lord' is deliberately absent from the veto list: the Beast Lord is a
    woman, so including it would break the one case this exists for."""
    cast = {"cast": [{"canonical_name": "Beast Lord", "aliases": []}]}
    cb.gender_from_ocr_pronouns(cast, [{"ocr_clean": "THE BEAST LORD? SHE FELL."}])
    assert cast["cast"][0]["gender"] == "female"


def test_a_disagreeing_page_and_an_owner_lock_both_win():
    cast = {"cast": [{"canonical_name": "Yuseung", "aliases": []},
                     {"canonical_name": "Locked", "aliases": [],
                      "gender": "male"}]}
    n = cb.gender_from_ocr_pronouns(cast, [
        {"ocr_clean": "YUSEUNG RAISES HIS BLADE."},     # male
        {"ocr_clean": "YUSEUNG TURNS, AND SHE RUNS."},  # female -> disagreement
        {"ocr_clean": "LOCKED, SHE SAID."}])            # lock must survive
    assert n == 0
    assert "gender" not in cast["cast"][0]
    assert cast["cast"][1]["gender"] == "male"
