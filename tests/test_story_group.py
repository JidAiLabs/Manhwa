"""story_group (Pass 2): group by understanding into a CONSECUTIVE, fully-covering
partition. The repair logic is the coverage invariant — every panel lands in
exactly one shot, in order, no matter how the model mis-orders/omits."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "story_group",
    Path(__file__).resolve().parent.parent / "tools" / "story_group.py")
sg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sg)  # type: ignore[union-attr]

ORDER = ["p0", "p1", "p2", "p3", "p4"]


def _files(shots):
    return [s["scene_files"] for s in shots]


def test_basic_grouping_assigns_contiguous_shot_ids():
    shots = sg.repair_to_shots(ORDER[:3], [
        {"scene_files": ["p0", "p1"], "segment": "present", "arc_label": "intro"},
        {"scene_files": ["p2"], "segment": "present", "arc_label": "next"}])
    assert _files(shots) == [["p0", "p1"], ["p2"]]
    assert [s["shot_id"] for s in shots] == [1, 2]


def test_coverage_invariant_unassigned_panel_is_never_dropped():
    # model forgot p2 entirely -> it continues the current beat, still shown
    shots = sg.repair_to_shots(ORDER[:3], [
        {"scene_files": ["p0", "p1"]}])
    flat = [f for s in shots for f in s["scene_files"]]
    assert flat == ["p0", "p1", "p2"]                 # all 3 covered, in order


def test_model_misordering_becomes_consecutive_runs():
    # model groups p0+p2 together (non-consecutive); reading order is preserved
    shots = sg.repair_to_shots(ORDER[:3], [
        {"scene_files": ["p0", "p2"], "arc_label": "A"},
        {"scene_files": ["p1"], "arc_label": "B"}])
    assert _files(shots) == [["p0"], ["p1"], ["p2"]]  # consecutive runs only


def test_max_beat_len_splits_long_runs():
    shots = sg.repair_to_shots(ORDER, [
        {"scene_files": ORDER, "segment": "present", "arc_label": "battle"}],
        max_beat_len=2)
    assert _files(shots) == [["p0", "p1"], ["p2", "p3"], ["p4"]]
    assert all(s["arc_label"] == "battle" for s in shots)   # tag preserved


def test_default_grouping_has_no_magic_panel_cap():
    # The reference-channel contract does not target a fixed group count. If the
    # model says a long consecutive run is one context span, default repair keeps
    # it as one span; splitting requires an explicit max_beat_len override.
    order = [f"p{i}" for i in range(7)]
    shots = sg.repair_to_shots(order, [
        {"scene_files": order, "segment": "present", "arc_label": "single idea"}])
    assert _files(shots) == [order]


def test_default_max_beat_len_caps_group_size():
    """REGRESSION (panel-collapse): the CLI now defaults to a finite cap so a huge
    consecutive run is split into gemma-sized beats (a 29/35-panel group overflows
    gemma4:26b and parse-fails). The cap limits PANELS PER BEAT only — it never
    caps a panel's narration length."""
    assert sg.DEFAULT_MAX_BEAT_LEN == 6
    order = [f"p{i}" for i in range(40)]
    shots = sg.repair_to_shots(order, [
        {"scene_files": order, "segment": "present", "arc_label": "one long battle"}],
        max_beat_len=sg.DEFAULT_MAX_BEAT_LEN)
    # every emitted beat stays within the cap; full coverage preserved
    assert all(len(s["scene_files"]) <= sg.DEFAULT_MAX_BEAT_LEN for s in shots)
    assert [f for s in shots for f in s["scene_files"]] == order


def test_flashback_segment_is_carried():
    shots = sg.repair_to_shots(ORDER[:3], [
        {"scene_files": ["p0"], "segment": "present"},
        {"scene_files": ["p1", "p2"], "segment": "flashback", "arc_label": "ten years ago"}])
    assert shots[1]["segment"] == "flashback" and shots[1]["arc_label"] == "ten years ago"
    assert shots[0]["segment"] == "present"
    # invalid segment normalizes to present
    assert sg.repair_to_shots(["x"], [{"scene_files": ["x"], "segment": "weird"}])[0]["segment"] == "present"


def test_group_panels_full_pipeline_with_stub_covers_everything():
    panels = [{"scene_file": f} for f in ORDER]
    captured = {}

    def stub(payload):
        captured["payload"] = payload
        return {"chapter": {"logline": "A lonely reader's novel becomes real.",
                            "premise": "He alone knows how the world ends."},
                "beats": [{"scene_files": ["p0", "p1"], "segment": "present"},
                          {"scene_files": ["p2", "p3", "p4"], "segment": "flashback"}]}

    shots, chapter = sg.group_panels(panels, stub, max_beat_len=4)
    flat = [f for s in shots for f in s["scene_files"]]
    assert flat == ORDER                              # full coverage
    assert captured["payload"]["panels"][0]["n"] == 0  # numbered, ordered input
    assert chapter["logline"].startswith("A lonely reader")   # spine captured
    assert sg.group_panels([], stub) == ([], {})


def test_group_schema_requires_nonoptional_story_spine_fields():
    assert set(sg.GROUP_SCHEMA["required"]) == {"chapter", "beats"}
    chapter = sg.GROUP_SCHEMA["properties"]["chapter"]
    assert set(chapter["required"]) == {"logline", "premise"}
    assert chapter["properties"]["logline"]["minLength"] >= 1
    assert chapter["properties"]["premise"]["minLength"] >= 1


def test_chapter_spine_complete_rejects_blank_fields():
    assert sg._chapter_spine_complete(
        {"logline": "A prince is hunted.", "premise": "His bloodline is fatal."})
    assert not sg._chapter_spine_complete({"logline": "", "premise": "x"})
    assert not sg._chapter_spine_complete({})


def test_grouping_context_cannot_starve_structured_output():
    assert sg._normalized_group_num_ctx(None) == 16384
    assert sg._normalized_group_num_ctx(8192) == 12288
    assert sg._normalized_group_num_ctx(16384) == 16384


def test_nonstory_files_drops_chrome_empty_and_parse_failures():
    panels = [
        {"scene_file": "p0", "panel_kind": "story"},
        {"scene_file": "p1", "panel_kind": "chrome"},       # logo / end-card
        {"scene_file": "p2", "panel_kind": "empty"},        # blank / empty bubble
        {"scene_file": "p3", "panel_kind": "story", "error": "parse_failed"},  # unparsed
        {"scene_file": "p4"},                                # missing kind -> kept
    ]
    dropped = sg.nonstory_files(panels)
    assert dropped == {"p1", "p2", "p3"}                 # chrome + empty + error
    assert "p0" not in dropped and "p4" not in dropped   # real story stays
    # captions are NOT dropped — they're kept (their words ride the narration)
    assert sg.nonstory_files([{"scene_file": "c", "panel_kind": "caption"}]) == set()


def test_effect_only_drops_pure_effect_panels_but_keeps_real_scenes():
    # REAL gemma records (live ORV/Nano/IE understood.json) — the calibration oracle.
    # Only a panel that names NOTHING concrete AND has an effect cue is dropped;
    # gemma's subjects=[] is unreliable, so real character/combat panels (which are
    # FULL of effect words: flash, embers, sparks) must survive on their nouns.
    panels = [
        # THE ORV SLIVER — story-kind, no subject, no dialogue, names only
        # shapes/fragments/streaks. 'background' must NOT match 'ground'. -> DROP
        {"scene_file": "p000008.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "",
         "description": "The panel shows bright, glowing red shapes against a "
                        "solid black background, resembling fragments or light streaks."},
        # real combat — names 'man'/'blade' though subjects=[] -> KEEP
        {"scene_file": "p000024.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "",
         "description": "A dark-haired man in tactical gear swings a blade, creating "
                        "a bright flash of light amidst flying debris and glowing embers."},
        # real close-up — 'face'/'eye'/'hair' though subjects=[] -> KEEP
        {"scene_file": "p000034.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "",
         "description": "A close-up shot of an anime-style character's face, showing "
                        "their eye and part of their blue-tinted hair against a bright, "
                        "glowing background."},
        # real scene — 'arm'/'hand'/'foliage' -> KEEP
        {"scene_file": "p000003.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "",
         "description": "A close-up shot shows a pale arm or limb being gripped or "
                        "struck by a dark, shadowy clawed hand amidst dark foliage."},
        # SFX transition but names 'structures'/'machinery' -> KEEP (conservative)
        {"scene_file": "p000011.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "",
         "description": "Large, stylized sound effect text overlays a blurred, "
                        "fast-moving scene of metallic structures or machinery streaking past."},
        # has a listed subject -> never evaluated -> KEEP
        {"scene_file": "p000007.jpg", "panel_kind": "story",
         "subjects": ["debris", "sparks"], "dialogue": "",
         "description": "Red SFX text over an abstract scene of debris and sparks."},
        # system/age card: subjects=[] but carries dialogue -> KEEP
        {"scene_file": "p000card.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "LIN ZICHEN - AGE: 5 MONTHS",
         "description": "Character introduction cards and a system notification window."},
        # a caption (not story-kind) is never touched by the effect filter
        {"scene_file": "c.jpg", "panel_kind": "caption", "subjects": [],
         "dialogue": "", "description": "glowing streaks of abstract light"},
    ]
    dropped = sg.effect_only_files(panels)
    assert dropped == {"p000008.jpg"}                      # ONLY the sliver
    for keep in ("p000024.jpg", "p000034.jpg", "p000003.jpg", "p000011.jpg",
                 "p000007.jpg", "p000card.jpg", "c.jpg"):
        assert keep not in dropped


def test_effect_only_drops_story_panel_with_empty_description():
    # story-kind, no subject, no dialogue, no description at all -> nothing real -> DROP
    assert sg.effect_only_files(
        [{"scene_file": "x.jpg", "panel_kind": "story", "subjects": [],
          "dialogue": "", "description": ""}]) == {"x.jpg"}
    # but an atmospheric establishing shot with NO effect words is KEPT even with
    # an unknown noun (the effect-cue requirement is the second safety net)
    assert sg.effect_only_files(
        [{"scene_file": "y.jpg", "panel_kind": "story", "subjects": [],
          "dialogue": "", "description": "A quiet panorama at dawn."}]) == set()


def test_effect_only_keeps_establishing_atmosphere_panels():
    # Aftermath/atmosphere establishing shots carry effect cues (glow, smoke,
    # haze) but ARE real scenes — they must survive on the broadened noun set,
    # while a pure flash/spark panel that names nothing concrete still drops.
    panels = [
        {"scene_file": "field.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "",
         "description": "Distant silhouettes stand on a smoke-covered battlefield "
                        "under a dim, glowing light."},
        {"scene_file": "ruin.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "",
         "description": "The view pans over wreckage and ruins, smoke rising, "
                        "faint light glowing through the haze."},
        # pure SFX flash — names NO concrete noun, has effect cues -> still DROP
        {"scene_file": "flash.jpg", "panel_kind": "story", "subjects": [],
         "dialogue": "",
         "description": "A bright flash and scattered sparks with motion lines "
                        "and streaking energy beams."},
    ]
    dropped = sg.effect_only_files(panels)
    assert dropped == {"flash.jpg"}
    assert "field.jpg" not in dropped and "ruin.jpg" not in dropped


def test_caption_solo_beat_folds_into_previous_same_segment_beat():
    panels = [{"scene_file": "p0", "panel_kind": "story"},
              {"scene_file": "c1", "panel_kind": "caption"},
              {"scene_file": "p2", "panel_kind": "story"}]
    assert sg.caption_files(panels) == {"c1"}
    shots = [
        {"shot_id": 1, "scene_files": ["p0"], "segment": "present", "arc_label": "a"},
        {"shot_id": 2, "scene_files": ["c1"], "segment": "present", "arc_label": "cap"},
        {"shot_id": 3, "scene_files": ["p2"], "segment": "present", "arc_label": "b"}]
    merged = sg.merge_caption_solos(shots, {"c1"})
    assert [s["scene_files"] for s in merged] == [["p0", "c1"], ["p2"]]
    assert [s["shot_id"] for s in merged] == [1, 2]      # renumbered contiguous


def test_system_panel_is_kept_and_never_folded_as_caption():
    # A panel panel_understand promoted to 'system' (a trained system_box
    # detection — Nano ch1 p000114) must be kept + SHOWN: it is not a caption, so
    # caption_files excludes it and merge_caption_solos never folds it away. The
    # adjacent caption instead folds INTO the system beat (its words ride that
    # shown art) — the system card is never dropped.
    panels = [{"scene_file": "sys", "panel_kind": "system"},
              {"scene_file": "c1", "panel_kind": "caption"}]
    caps = sg.caption_files(panels)
    assert caps == {"c1"} and "sys" not in caps         # system is NOT a caption
    shots = [
        {"shot_id": 1, "scene_files": ["sys"], "segment": "present", "arc_label": "s"},
        {"shot_id": 2, "scene_files": ["c1"], "segment": "present", "arc_label": "c"}]
    merged = sg.merge_caption_solos(shots, caps)
    assert [s["scene_files"] for s in merged] == [["sys", "c1"]]   # sys shown, kept


def test_title_card_rescues_system_card_mislabeled_chrome():
    # an in-world SYSTEM card the LLM mislabeled 'chrome' MUST be rescued, else it
    # silently drops from the video (it carries system vocab + is a flat card).
    sys_card = {"scene_file": "p001.jpg", "ocr_clean": "SYSTEM ACTIVATION",
                "panel_kind": "chrome", "flat_frac": 0.85, "text_coverage": 0.05}
    assert "p001.jpg" in sg.title_card_files([sys_card])


def test_title_card_rescue_does_not_re_include_chapter_or_credits_chrome():
    # the rescue must NOT bring back chapter-number / credits cards (no system
    # vocab) — that would re-introduce title cards into the video.
    chap = {"scene_file": "p002.jpg", "ocr_clean": "CHAPTER ELEVEN",
            "panel_kind": "chrome", "flat_frac": 0.85, "text_coverage": 0.05}
    credits = {"scene_file": "p003.jpg", "ocr_clean": "AUTOR HAN JOONG ARTISTA",
               "panel_kind": "chrome", "flat_frac": 0.85, "text_coverage": 0.05}
    out = sg.title_card_files([chap, credits])
    assert "p002.jpg" not in out and "p003.jpg" not in out


def test_title_card_keeps_correctly_stamped_system_card():
    s = {"scene_file": "p004.jpg", "ocr_clean": "STATUS WINDOW",
         "panel_kind": "system", "flat_frac": 0.85, "text_coverage": 0.05}
    assert "p004.jpg" in sg.title_card_files([s])


def test_title_card_files_protects_story_system_cards_not_chrome():
    # flat_frac pre-set so the detector skips image I/O; reuses prep_qa._is_title_card
    items = [
        {"scene_file": "c", "ocr_clean": "CENTRAL TOWER.", "flat_frac": 0.9,
         "text_coverage": 0.05, "text_only": False},                       # story org card
        {"scene_file": "a", "ocr_clean": "LIN ZICHEN - AGE: 5 MONTHS", "flat_frac": 0.9,
         "text_coverage": 0.05, "text_only": False},                       # story time card
        {"scene_file": "d", "ocr_clean": "thanks for reading join our discord and subscribe now please",
         "flat_frac": 0.9, "text_coverage": 0.05, "text_only": False},     # promo chrome
    ]
    cards = sg.title_card_files(items)
    assert "c" in cards and "a" in cards     # in-world system cards protected
    assert "d" not in cards                  # long promo is chrome, not protected


def test_leading_caption_folds_forward_into_next_same_segment_beat():
    # A caption that INTRODUCES the moment after it (no previous same-segment beat
    # to fold back into) must ride the NEXT beat's art + narration, never stand
    # alone as a shown bubble. (story_group SYSTEM prompt: never strand an intro
    # caption before the moment it sets up.)
    shots = [
        {"shot_id": 1, "scene_files": ["c0"], "segment": "present", "arc_label": "intro cap"},
        {"shot_id": 2, "scene_files": ["p1", "p2"], "segment": "present", "arc_label": "scene"}]
    merged = sg.merge_caption_solos(shots, {"c0"})
    assert [s["scene_files"] for s in merged] == [["c0", "p1", "p2"]]
    assert [s["shot_id"] for s in merged] == [1]


def test_caption_only_kept_out_of_standalone_shots_but_text_rides_neighbor():
    # caption/empty/chrome are excluded as STANDALONE shots: empty/chrome never
    # enter `story`; a caption folds into the adjacent ART beat so the bubble is
    # never shown while its words stay in that beat for the narrator.
    panels = [{"scene_file": "p0", "panel_kind": "story"},
              {"scene_file": "cap", "panel_kind": "caption"},
              {"scene_file": "emp", "panel_kind": "empty"},
              {"scene_file": "p3", "panel_kind": "story"}]
    nonstory = sg.nonstory_files(panels)             # empty excluded from `story`
    assert "emp" in nonstory and "cap" not in nonstory
    shots = [
        {"shot_id": 1, "scene_files": ["p0"], "segment": "present", "arc_label": "a"},
        {"shot_id": 2, "scene_files": ["cap"], "segment": "present", "arc_label": "c"},
        {"shot_id": 3, "scene_files": ["p3"], "segment": "present", "arc_label": "b"}]
    merged = sg.merge_caption_solos(shots, sg.caption_files(panels))
    # no shot is caption-only: the caption rides p0's beat; no standalone bubble
    cap = sg.caption_files(panels)
    assert not any(s["scene_files"] and all(f in cap for f in s["scene_files"])
                   for s in merged)
    assert ["cap"] not in [s["scene_files"] for s in merged]


def test_caption_closer_with_no_same_segment_neighbour_stays():
    shots = [
        {"shot_id": 1, "scene_files": ["p0"], "segment": "flashback", "arc_label": "x"},
        {"shot_id": 2, "scene_files": ["c9"], "segment": "present", "arc_label": "end"}]
    merged = sg.merge_caption_solos(shots, {"c9"})
    assert [s["scene_files"] for s in merged] == [["p0"], ["c9"]]   # segment differs


def test_annotate_intensity_takes_the_peak_per_beat():
    panels = [{"scene_file": "a", "intensity": "calm"},
              {"scene_file": "b", "intensity": "explosive"},
              {"scene_file": "c", "intensity": "tense"}]
    shots = [{"shot_id": 1, "scene_files": ["a", "b"]},   # peak of calm+explosive
             {"shot_id": 2, "scene_files": ["c"]},
             {"shot_id": 3, "scene_files": ["z"]}]         # unknown panel -> calm
    sg.annotate_intensity(shots, panels)
    assert shots[0]["intensity"] == "explosive"   # one explosive panel sets pace
    assert shots[1]["intensity"] == "tense"
    assert shots[2]["intensity"] == "calm"


def test_system_panel_is_never_excluded():
    # In-world quest/status/notification cards are PLOT — the three filter helpers
    # must never drop them. keep_by_understanding (in run()) is covered at the
    # scene_chrome chokepoint: is_chrome_scene returns False for panel_kind="system",
    # so ocr_chrome can never include a system panel (it always stays in the keep-set).
    panels = [
        {"scene_file": "p01.jpg", "panel_kind": "story",  "description": "a man stands", "subjects": ["man"]},
        {"scene_file": "p02.jpg", "panel_kind": "system", "description": "QUEST DIRECTIONS window",
         "dialogue": "QUEST DIRECTIONS. NUMBER OF PLAYERS TO KILL: 1.", "subjects": []},
    ]
    assert "p02.jpg" not in sg.nonstory_files(panels)
    assert "p02.jpg" not in sg.effect_only_files(panels)
    assert "p02.jpg" not in sg.caption_files(panels)


def test_oversized_beat_splits_at_cap():
    scene_order = [f"p{i}.jpg" for i in range(12)]
    model_beats = [{"scene_files": scene_order}]   # one 12-panel beat (canonical shape)
    shots = sg.repair_to_shots(scene_order, model_beats, max_beat_len=8)
    assert len(shots) >= 2
    assert all(len(s["scene_files"]) <= 8 for s in shots)
    assert sum(len(s["scene_files"]) for s in shots) == 12   # every panel covered once


def test_default_cap_is_tighter():
    assert sg.DEFAULT_MAX_BEAT_LEN == 6
