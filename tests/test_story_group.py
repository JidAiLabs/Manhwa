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


def test_gist_truncates_at_word_boundary_with_ellipsis():
    assert sg._gist("short text", 160) == "short text"          # under limit -> untouched
    long = "one two three four five six seven eight nine ten " * 5   # 250+ chars
    g = sg._gist(long, 20)
    assert len(g) <= 21 and g.endswith("…") and not g[:-1].endswith(" ")
    assert long.startswith(g[:-1].rstrip())                      # cut at a real word boundary
    # no space anywhere in the window -> hard cut, still ellipsis-marked
    assert sg._gist("x" * 300, 20) == "x" * 20 + "…"
    assert sg._gist("", 160) == "" and sg._gist(None, 160) == ""


def test_build_grouping_payload_shrinks_description_to_a_gist_under_budget():
    """REGRESSION (context/output budget): Omniscient Reader Episode 1's ~96
    panels of rich pu_v2 descriptions built a grouping payload large enough to
    starve the model's context/output budget — generation truncated mid-JSON at
    ~4-5KB of output on every retry (three captured raw responses, all
    Unterminated string / Expecting value). Grouping only needs a gist plus
    subjects/panel_kind to tell panels apart, not full prose; shrinking
    `description` to ~160 chars measurably shrinks the whole-chapter payload."""
    import json
    panels = [{
        "scene_file": f"p{i:03d}.jpg",
        "description": ("A very long and detailed panel description of the "
                        "action taking place here. " * 10)[:400],
        "panel_kind": "story",
        "subjects": ["a man"],
        "intensity": "calm",
    } for i in range(100)]
    built = sg.build_grouping_payload(panels)
    ellipsis_len = len("…")
    for p in built["panels"]:
        assert len(p["description"]) <= 160 + ellipsis_len
        assert p["panel_kind"] == "story" and p["subjects"] == ["a man"]   # kept intact
        assert p["intensity"] == "calm"
    # MEASURED: this synthetic 100-panel/400-char-description chapter serializes
    # to 42202 bytes under the old [:300]-slice payload vs 33202 bytes here —
    # keep comfortably under a budget that separates the two.
    total_bytes = len(json.dumps(built).encode("utf-8"))
    assert total_bytes < 38_000, f"grouping payload too large: {total_bytes} bytes"


def test_shrink_payload_re_gists_an_already_built_payload_harder():
    built = sg.build_grouping_payload([{
        "scene_file": "p0.jpg",
        "description": "one two three four five six seven eight nine ten " * 5,
        "panel_kind": "story", "subjects": ["a man"], "intensity": "calm"}])
    shrunk = sg._shrink_payload(built, 80)
    assert len(shrunk["panels"][0]["description"]) <= 80 + len("…")
    assert shrunk["panels"][0]["panel_kind"] == "story"          # other fields untouched
    assert built["panels"][0]["description"] != shrunk["panels"][0]["description"]


def test_unsafe_spine_dumps_raw_response_before_raising(tmp_path, monkeypatch):
    """Diagnosability (eyes wave): when spine validation rejects the grouping
    ("unsafe chapter story spine"), the raw model response is atomically
    dumped beside the manifests and the raise message names the file — the
    ORV ep1 grouped_failed left NOTHING to diagnose."""
    import json
    import sys as _sys

    import pytest

    understood = {"panels": [{
        "scene_file": "p0.jpg", "description": "A man walks the ridge line.",
        "action": "he walks on", "subjects": ["a man"],
        "panel_kind": "story", "intensity": "calm"}]}
    vision = {"items": [{"scene_file": "p0.jpg"}]}
    up = tmp_path / "manifest.panels.understood.json"
    up.write_text(json.dumps(understood))
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    out = tmp_path / "manifest.groups.json"

    def fake_call(**kw):
        parsed = {"chapter": {"logline": "A ridge walk.", "premise": ""},
                  "beats": [{"from_index": 0, "to_index": 0}]}
        return parsed, "RAW_MODEL_TEXT", {}

    monkeypatch.setattr(sg, "_call_model_with_backoff", fake_call)
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")   # main() mutates it
    monkeypatch.setattr(_sys, "argv", [
        "story_group.py", "--understood", str(up),
        "--vision-manifest", str(vp), "--out", str(out)])
    with pytest.raises(SystemExit) as ei:
        sg.main()
    msg = str(ei.value)
    assert "unsafe chapter story spine" in msg
    dumps = sorted(tmp_path.glob(".story_group_raw-*.json"))
    assert len(dumps) == 1, "raw capture file must exist beside the manifests"
    data = json.loads(dumps[0].read_text())
    assert data["raw_response"] == "RAW_MODEL_TEXT"
    assert data["parsed_obj"]["chapter"]["premise"] == ""
    assert data["model"] and data["ts"]
    assert dumps[0].name in msg               # the raise names the capture


def test_parse_failure_error_message_names_truncation_not_blank_spine(tmp_path, monkeypatch):
    """BETTER ERROR (context/output budget): when the grouping response never
    parses into a dict at ALL (raw non-empty), the spine-guard message must say
    so -- "logline/premise is blank" is misleading when there was no chapter
    object to inspect in the first place. This is exactly what the three ORV
    ep1/2 raw captures showed: Unterminated string / Expecting value, parsed
    None, and the old message blamed a spine the parser never reached. The raw
    capture-to-disk behavior (test above) must still fire."""
    import json
    import sys as _sys

    import pytest

    understood = {"panels": [{
        "scene_file": "p0.jpg", "description": "A man walks the ridge line.",
        "action": "he walks on", "subjects": ["a man"],
        "panel_kind": "story", "intensity": "calm"}]}
    vision = {"items": [{"scene_file": "p0.jpg"}]}
    up = tmp_path / "manifest.panels.understood.json"
    up.write_text(json.dumps(understood))
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    out = tmp_path / "manifest.groups.json"

    raw_truncated = '{"chapter": {"logline": "A ridge walk", "premise": "He walks'  # unterminated

    def fake_call(**kw):
        return None, raw_truncated, {}          # parse failure on every attempt

    monkeypatch.setattr(sg, "_call_model_with_backoff", fake_call)
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")
    monkeypatch.setattr(_sys, "argv", [
        "story_group.py", "--understood", str(up),
        "--vision-manifest", str(vp), "--out", str(out)])
    with pytest.raises(SystemExit) as ei:
        sg.main()
    msg = str(ei.value)
    assert "failed to parse" in msg
    assert "logline/premise is blank" not in msg   # the old, misleading message
    assert f"raw length={len(raw_truncated)}" in msg
    dumps = sorted(tmp_path.glob(".story_group_raw-*.json"))
    assert len(dumps) == 1, "raw capture file must exist beside the manifests"
    assert dumps[0].name in msg
    data = json.loads(dumps[0].read_text())
    assert data["raw_response"] == raw_truncated and data["parsed_obj"] is None


def test_empty_response_error_message_is_truthful_not_blank_spine(tmp_path,
                                                                   monkeypatch):
    """TRUTHFULNESS EDGE (2026-07-07 review): when the grouping response
    parses to NOTHING and raw comes back EMPTY too (no text at all, not even
    a truncated fragment), the spine-guard message must say exactly that --
    "chapter logline/premise is blank" implies a chapter object existed with
    blank fields, which is misleading when there was no response whatsoever.
    Distinct from the truncated-mid-generation case above (needs non-empty
    raw); the raw capture-to-disk behavior must still fire."""
    import json
    import sys as _sys

    import pytest

    understood = {"panels": [{
        "scene_file": "p0.jpg", "description": "A man walks the ridge line.",
        "action": "he walks on", "subjects": ["a man"],
        "panel_kind": "story", "intensity": "calm"}]}
    vision = {"items": [{"scene_file": "p0.jpg"}]}
    up = tmp_path / "manifest.panels.understood.json"
    up.write_text(json.dumps(understood))
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    out = tmp_path / "manifest.groups.json"

    def fake_call(**kw):
        return None, "", {}                       # truly empty response

    monkeypatch.setattr(sg, "_call_model_with_backoff", fake_call)
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")
    monkeypatch.setattr(_sys, "argv", [
        "story_group.py", "--understood", str(up),
        "--vision-manifest", str(vp), "--out", str(out)])
    with pytest.raises(SystemExit) as ei:
        sg.main()
    msg = str(ei.value)
    assert "model returned an empty response" in msg
    assert "logline/premise is blank" not in msg
    assert "failed to parse" not in msg
    dumps = sorted(tmp_path.glob(".story_group_raw-*.json"))
    assert len(dumps) == 1, "raw capture file must exist beside the manifests"
    data = json.loads(dumps[0].read_text())
    assert data["raw_response"] == "" and data["parsed_obj"] is None


def test_shrink_retry_succeeds_after_one_harder_shrink(tmp_path, monkeypatch):
    """DEFENSIVE RETRY (context/output budget): a parse failure (not a bad
    spine) gets exactly ONE extra attempt with descriptions hard-capped to 80
    chars before giving up -- a smaller prompt frees context/output headroom
    for the model to finish valid JSON. Must engage the shrink immediately, not
    burn an attempt on the generic REPAIR text first (that targets a valid-but-
    incomplete spine, a different failure mode)."""
    import json
    import sys as _sys

    understood = {"panels": [{
        "scene_file": "p0.jpg",
        "description": "a very long panel description " * 20,   # > 160 chars
        "action": "he walks on", "subjects": ["a man"],
        "panel_kind": "story", "intensity": "calm"}]}
    vision = {"items": [{"scene_file": "p0.jpg"}]}
    up = tmp_path / "manifest.panels.understood.json"
    up.write_text(json.dumps(understood))
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    out = tmp_path / "manifest.groups.json"

    calls = []

    def fake_call(**kw):
        payload = kw["user_payload"]
        calls.append(payload)
        desc = payload["panels"][0]["description"]
        # threshold sits well clear of both sides: a normal 160-gist lands near
        # 160 chars, a hard-shrunk 80-gist near 80 (+1 for a possible ellipsis)
        if len(desc) > 120:                       # first attempt: normal 160-gist
            return None, "TRUNCATED_MID_STR_HERE", {}
        return ({"chapter": {"logline": "A ridge walk.", "premise": "He walks alone."},
                  "beats": [{"from_index": 0, "to_index": 0}]}, "OK_RAW", {})

    monkeypatch.setattr(sg, "_call_model_with_backoff", fake_call)
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")
    monkeypatch.setattr(_sys, "argv", [
        "story_group.py", "--understood", str(up),
        "--vision-manifest", str(vp), "--out", str(out)])
    assert sg.main() == 0
    assert len(calls) == 2, "exactly one retry: normal gist, then the hard shrink"
    assert len(calls[0]["panels"][0]["description"]) > 80     # attempt 1: unshrunk gist
    assert len(calls[1]["panels"][0]["description"]) <= 80 + len("…")   # attempt 2: shrunk
    written = json.loads(out.read_text())
    assert written["shots"] and written["shots"][0]["scene_files"] == ["p0.jpg"]


# ---------------------------------------------------------------------------
# expand_index_ranges (root cause fix): GROUP_SCHEMA beats are now
# {from_index, to_index} index ranges over the payload's numbered panel list,
# not scene_files filename echoes -- gemma4:26b died mid-enumeration typing
# out ~30 beats' worth of filenames on a ~96-panel chapter. These tests cover
# the pure expansion/validation function and its wiring into call_fn's
# REPAIR re-ask loop and the final safety-net gate.
# ---------------------------------------------------------------------------

def test_expand_index_ranges_happy_path_matches_scene_files_equivalent():
    """Range expansion happy path (synthetic 10-panel story, 3 beats): the
    ranges must expand to the EXACT SAME shots as the equivalent hand-written
    scene_files beats, once both are fed through repair_to_shots -- proving
    expand_index_ranges is a lossless, byte-shape-identical stand-in for the
    old scene_files echo."""
    order = [f"p{i}" for i in range(10)]
    range_beats = [
        {"from_index": 0, "to_index": 2, "segment": "present", "arc_label": "intro"},
        {"from_index": 3, "to_index": 5, "segment": "present", "arc_label": "rising"},
        {"from_index": 6, "to_index": 9, "segment": "flashback", "arc_label": "backstory"},
    ]
    expanded, issue = sg.expand_index_ranges(range_beats, order)
    assert issue == ""
    assert [b["scene_files"] for b in expanded] == [order[0:3], order[3:6], order[6:10]]

    scene_files_beats = [
        {"scene_files": order[0:3], "segment": "present", "arc_label": "intro"},
        {"scene_files": order[3:6], "segment": "present", "arc_label": "rising"},
        {"scene_files": order[6:10], "segment": "flashback", "arc_label": "backstory"},
    ]
    assert (sg.repair_to_shots(order, expanded)
            == sg.repair_to_shots(order, scene_files_beats))


def test_expand_index_ranges_allows_gaps_like_scene_files_always_did():
    # A gap between ranges is the model's prerogative, not a defect -- exactly
    # like an omitted filename in the old scene_files lists. repair_to_shots's
    # coverage invariant folds the skipped panel into the beat before it, so
    # nothing is ever lost.
    order = [f"p{i}" for i in range(5)]
    expanded, issue = sg.expand_index_ranges(
        [{"from_index": 0, "to_index": 1, "segment": "present", "arc_label": "a"},
         {"from_index": 3, "to_index": 4, "segment": "present", "arc_label": "b"}], order)
    assert issue == ""
    shots = sg.repair_to_shots(order, expanded)
    assert [f for s in shots for f in s["scene_files"]] == order   # p2 not lost


def test_expand_index_ranges_rejects_overlapping_ranges():
    order = [f"p{i}" for i in range(6)]
    expanded, issue = sg.expand_index_ranges(
        [{"from_index": 0, "to_index": 3}, {"from_index": 2, "to_index": 5}], order)
    assert expanded == [] and "overlap" in issue


def test_expand_index_ranges_rejects_to_index_out_of_bounds():
    order = [f"p{i}" for i in range(4)]                    # valid indices: 0..3
    expanded, issue = sg.expand_index_ranges([{"from_index": 0, "to_index": 4}], order)
    assert expanded == [] and "out of bounds" in issue


def test_expand_index_ranges_rejects_from_greater_than_to():
    order = [f"p{i}" for i in range(4)]
    expanded, issue = sg.expand_index_ranges([{"from_index": 2, "to_index": 0}], order)
    assert expanded == [] and "from_index" in issue and "to_index" in issue


def test_expand_index_ranges_rejects_non_integer_or_boolean_index():
    order = [f"p{i}" for i in range(4)]
    assert sg.expand_index_ranges([{"from_index": "0", "to_index": 1}], order)[1]
    assert sg.expand_index_ranges([{"from_index": -1, "to_index": 1}], order)[1]
    # bool is an int subclass in Python -- must still be rejected as non-integer
    assert sg.expand_index_ranges([{"from_index": True, "to_index": 1}], order)[1]


def test_expand_index_ranges_rejects_missing_or_empty_beats():
    order = ["p0"]
    assert sg.expand_index_ranges(None, order)[1]
    assert sg.expand_index_ranges([], order)[1]
    assert sg.expand_index_ranges("not a list", order)[1]


def test_realistic_96_panel_range_response_is_well_under_output_budget():
    """OUTPUT BUDGET (the root cause): the OLD scene_files-echo schema forced
    the model to spell out every scene filename per beat -- on Omniscient
    Reader Episode 1's ~96 panels that was ~30 beats' worth of filename
    arrays, and gemma4:26b died mid-enumeration on every production attempt
    (raw captures cut off mid scene_files-array; one attempt came back
    near-empty). An index-range beat is two small integers plus a couple of
    short strings -- build a REAL-SHAPED response for a chapter of the same
    size (96 panels, 24 beats -- the same ballpark as the observed ~30) and
    confirm it both parses cleanly through expand_index_ranges AND that its
    JSON is comfortably small (the pinned budget claim)."""
    import json

    n_panels = 96
    scene_order = [f"p{i:06d}.jpg" for i in range(n_panels)]
    span = 4                                   # 96 / 4 = 24 beats, clean division
    labels = ("open", "clash", "reveal", "turn", "aftermath", "escape",
              "reunion", "standoff")
    beats = [{
        "from_index": i, "to_index": i + span - 1,
        "segment": "flashback" if bi % 8 == 0 else "present",
        "arc_label": labels[bi % len(labels)],
    } for bi, i in enumerate(range(0, n_panels, span))]
    response = {
        "chapter": {"logline": "A reader is pulled into his own novel.",
                    "premise": "He alone knows how it ends and must survive."},
        "beats": beats,
    }

    # vendored parse: round-trip through JSON exactly as the real pipeline
    # would (model text -> json.loads -> expand_index_ranges).
    raw = json.dumps(response, separators=(",", ":"))
    parsed = json.loads(raw)
    expanded, issue = sg.expand_index_ranges(parsed["beats"], scene_order)
    assert issue == ""
    assert len(expanded) == len(beats) == 24
    assert sum(len(b["scene_files"]) for b in expanded) == n_panels   # full coverage

    # THE BUDGET CLAIM, pinned: comfortably under 2000 chars for a realistic
    # ~96-panel chapter's beats -- vs. the old scheme, which could not even
    # finish generating on the same chapter (~4-5KB truncated mid-string).
    assert len(raw) < 2000, f"range response too large: {len(raw)} chars"


def test_range_repair_retry_succeeds_after_range_specific_complaint(tmp_path, monkeypatch):
    """REPAIR re-ask (malformed ranges): a beat with an out-of-bounds range
    must trigger the SAME REPAIR-and-reask mechanism as an unsafe chapter
    spine, but with a complaint that specifically names the RANGE problem
    (not the generic spine wording) -- the second attempt then succeeds."""
    import json
    import sys as _sys

    understood = {"panels": [
        {"scene_file": "p0.jpg", "description": "a", "panel_kind": "story", "subjects": ["x"]},
        {"scene_file": "p1.jpg", "description": "b", "panel_kind": "story", "subjects": ["x"]},
    ]}
    vision = {"items": [{"scene_file": "p0.jpg"}, {"scene_file": "p1.jpg"}]}
    up = tmp_path / "manifest.panels.understood.json"
    up.write_text(json.dumps(understood))
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    out = tmp_path / "manifest.groups.json"

    calls = []

    def fake_call(**kw):
        calls.append(kw["system_instruction"])
        good_chapter = {"logline": "A short chapter.", "premise": "Two panels happen."}
        if len(calls) == 1:
            # to_index=5 is out of bounds for a 2-panel chapter (max valid index 1)
            return ({"chapter": good_chapter,
                      "beats": [{"from_index": 0, "to_index": 5}]}, "BAD_RANGE_RAW", {})
        return ({"chapter": good_chapter,
                  "beats": [{"from_index": 0, "to_index": 1, "segment": "present",
                             "arc_label": "both"}]}, "GOOD_RANGE_RAW", {})

    monkeypatch.setattr(sg, "_call_model_with_backoff", fake_call)
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")
    monkeypatch.setattr(_sys, "argv", [
        "story_group.py", "--understood", str(up),
        "--vision-manifest", str(vp), "--out", str(out)])
    assert sg.main() == 0
    assert len(calls) == 2
    assert "invalid index ranges" in calls[1]                     # range-specific complaint
    assert "chapter story spine was invalid" not in calls[1]      # not the spine wording
    written = json.loads(out.read_text())
    assert written["shots"][0]["scene_files"] == ["p0.jpg", "p1.jpg"]


def test_unsafe_range_dumps_raw_response_before_raising(tmp_path, monkeypatch):
    """Malformed ranges that survive the REPAIR re-ask must fail exactly like
    an unsafe chapter spine: raw response dumped beside the manifests, and a
    truthful message naming the range problem -- instead of silently
    collapsing every panel into one giant shot and writing THAT out (which is
    what repair_to_shots would do if fed beats with no "scene_files" key)."""
    import json
    import sys as _sys

    import pytest

    understood = {"panels": [
        {"scene_file": "p0.jpg", "description": "a", "panel_kind": "story", "subjects": ["x"]},
        {"scene_file": "p1.jpg", "description": "b", "panel_kind": "story", "subjects": ["x"]},
    ]}
    vision = {"items": [{"scene_file": "p0.jpg"}, {"scene_file": "p1.jpg"}]}
    up = tmp_path / "manifest.panels.understood.json"
    up.write_text(json.dumps(understood))
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    out = tmp_path / "manifest.groups.json"

    def fake_call(**kw):
        # persistently overlapping ranges -- never resolved, even after REPAIR
        return ({"chapter": {"logline": "A short chapter.", "premise": "Stuff happens here."},
                  "beats": [{"from_index": 0, "to_index": 1},
                            {"from_index": 1, "to_index": 1}]}, "OVERLAP_RAW", {})

    monkeypatch.setattr(sg, "_call_model_with_backoff", fake_call)
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")
    monkeypatch.setattr(_sys, "argv", [
        "story_group.py", "--understood", str(up),
        "--vision-manifest", str(vp), "--out", str(out)])
    with pytest.raises(SystemExit) as ei:
        sg.main()
    msg = str(ei.value)
    assert "beat index ranges" in msg
    assert "overlap" in msg
    dumps = sorted(tmp_path.glob(".story_group_raw-*.json"))
    assert len(dumps) == 1, "raw capture file must exist beside the manifests"
    assert dumps[0].name in msg
    data = json.loads(dumps[0].read_text())
    assert data["raw_response"] == "OVERLAP_RAW"
