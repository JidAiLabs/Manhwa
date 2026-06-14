"""publish_concept: coherent title/hook/style/description/pinned assembly."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "publish_concept",
    Path(__file__).resolve().parent.parent / "tools" / "publish_concept.py")
pc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pc)  # type: ignore[union-attr]


def test_pinned_comment_is_only_place_with_real_name():
    p = pc.pinned_comment("Infinite Evolution From Zero", "https://x.com/book/1")
    assert "Infinite Evolution From Zero" in p and "official" in p
    assert pc.pinned_comment("") .startswith("Manhwa:")


def test_pick_hook_matches_style():
    assert pc.pick_hook(["GENIUS", "LEVEL 9999", "HE WINS"], "stat_callout") == "LEVEL 9999"
    assert pc.pick_hook(["WEAK|GOD", "GENIUS"], "before_after") == "WEAK|GOD"
    assert pc.pick_hook(["GENIUS", "SSS"], "power_reveal") == "GENIUS"
    assert pc.pick_hook([], "power_reveal") == ""


def test_description_has_synopsis_tags_boilerplate_but_no_real_name():
    d = pc.build_description("A nobody awakens a hidden class! 🔥",
                            ["#manhwa", "necromancer"])
    assert "hidden class" in d and "#manhwa" in d and "#necromancer" in d
    assert "Patreon" in d and "Tags:" in d


def test_assemble_concept_is_coherent_and_copyright_safe():
    beats = {"beats": [{"group_id": 1, "what_happens": "he checks his status "
                        "window; level and rank S skill appear"}]}
    llm = {"title": "When a Nobody Awakens the Rarest Class!",
           "hooks": ["GENIUS", "RANK SSS", "HE WINS"],
           "synopsis": "A mocked boy awakens a hidden class. 🔥",
           "hashtags": ["#manhwa", "#system"]}
    c = pc.assemble_concept(beats, llm, series_title="Solo Necromancer",
                            official_link="http://x")
    assert c["style"] == "stat_callout"            # from the UI/level signal
    assert c["hook"] == "RANK SSS"                  # stat hook for stat style
    assert "Solo Necromancer" not in c["title"]
    assert "Solo Necromancer" not in c["description"]
    assert "Solo Necromancer" in c["pinned_comment"]   # only here
    assert c["style_overlay"]["label_pos"]            # overlay wired


# ---- bundle (per-video) level --------------------------------------------

def test_parts_timestamps_start_at_zero_and_accumulate():
    p = pc.parts_timestamps([3925.0, 3923.0, 3800.0])
    assert p[0].startswith("0:00 ")                 # YouTube rule: first = 0:00
    assert p[1].startswith("1:05:25 ")              # 3925s -> 1:05:25
    assert "Part 3" in p[2]


def test_select_bundle_climax_picks_highest_intensity_chapter():
    ch1 = {"beats": [{"group_id": 1, "scene_selection": [{"intensity": "calm",
            "scene_file": "a.jpg"}]}]}
    ch2 = {"beats": [{"group_id": 1, "scene_selection": [{"intensity": "explosive",
            "scene_file": "boom.jpg"}]}]}
    ci, refs = pc.select_bundle_climax([ch1, ch2])
    assert ci == 1 and refs == ["boom.jpg"]         # climax is in chapter 2


def test_bundle_digest_spans_chapters():
    chs = [{"beats": [{"group_id": 1, "hook": f"hook {i}"}]} for i in range(3)]
    d = pc.bundle_digest(chs)
    assert "[Chapter 1]" in d and "[Chapter 3]" in d


def test_build_bundle_concept_arc_title_climax_refs_and_parts():
    weak = {"beats": [{"group_id": 1, "what_happens": "a mocked weakling",
            "scene_selection": [{"intensity": "calm", "scene_file": "w.jpg"}]}]}
    payoff = {"beats": [{"group_id": 1, "what_happens": "he breaks every record",
              "scene_selection": [{"intensity": "explosive", "scene_file": "win.jpg"}]}]}
    llm = {"title": "From Mocked Weakling to Record Breaker!",
           "hooks": ["GENIUS"], "synopsis": "Setup to payoff. 🔥",
           "hashtags": ["#manhwa"]}
    c = pc.build_bundle_concept([weak, payoff], llm, durations=[3600.0, 3600.0],
                                series_title="Hidden Series")
    assert c["climax_chapter_index"] == 1 and c["refs"] == ["win.jpg"]
    assert c["parts"][0].startswith("0:00") and "1:00:00" in c["parts"][1]
    assert "0:00" in c["description"]               # parts appended to desc
    assert "Hidden Series" not in c["description"]  # still copyright-safe
