# tests/test_narration_punchup_persona.py
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
