# tests/test_intensity_calibration.py
import tools.script_expander as se
import tools.panel_understand as pu


def test_intense_panel_no_longer_auto_escalates_to_tense():
    # rank 2 = "intense": a single intense panel must NOT force the beat's mood up
    assert se._escalate_tag_for_intensity("serious", 2) == "serious"
    # rank 3 = "explosive": a genuine peak still escalates
    assert se._escalate_tag_for_intensity("serious", 3) == "excited"
    # non-escalatable tags untouched at any rank
    assert se._escalate_tag_for_intensity("whisper", 3) == "whisper"


def test_panel_understand_prompt_reserves_high_intensity_for_peaks():
    # the SYSTEM prompt must instruct reserving intense/explosive for real peaks
    prompt = pu._build_system_prompt() if hasattr(pu, "_build_system_prompt") else pu.SYSTEM
    low = prompt.lower()
    assert "reserve" in low and "intense" in low and "peak" in low
