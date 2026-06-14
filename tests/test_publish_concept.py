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
