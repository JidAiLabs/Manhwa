# tests/test_narration_punchup_persona.py
import json

import tools.narration_punchup as np


def test_cinematic_rules_persona_is_default_not_seasoning():
    rules = np.CINEMATIC_RULES.lower()
    # the OLD gate (persona off on dramatic beats) must be gone:
    assert "occasional seasoning" not in rules
    assert "never the default" not in rules
    assert "purely cinematic" not in rules
    # the NEW contract (voice always on; gravity only drops jokes) must be present:
    assert "always on" in rules
    assert "drop the jokes" in rules or "drops the jokes" in rules
    # grounding guardrails retained:
    assert "weather" in rules and "caption" in rules


def test_build_prompt_injects_niche_register():
    lines = [{"group_id": 1, "narration": "x"}]   # build_prompt's cinematic branch needs this schema
    with_niche = np.build_prompt(lines, ["Hero"], "cinematic",
                                 niche="C", niche_secondary="A")
    assert "Dark-Action/Revenge" in with_niche      # primary C register text
    assert "SECONDARY" in with_niche                 # secondary A flavor
    base_only = np.build_prompt(lines, ["Hero"], "cinematic")
    # the C register text proves injection; its ABSENCE proves base-only.
    # (Do NOT assert on "NICHE TEMPERATURE" — CINEMATIC_RULES itself mentions that phrase.)
    assert "Dark-Action/Revenge" not in base_only


def test_load_niche_reads_episode_manifest(tmp_path):
    (tmp_path / "manifest.series.json").write_text(
        json.dumps({"niche_primary": "C", "niche_secondary": "A"}))
    assert np._load_niche(str(tmp_path), "", "") == ("C", "A")
    # explicit args win over the manifest:
    assert np._load_niche(str(tmp_path), "D", "") == ("D", "")
    # missing manifest -> empty (base voice), never crash:
    assert np._load_niche(str(tmp_path / "nope"), "", "") == ("", "")
