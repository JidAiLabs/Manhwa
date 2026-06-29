import importlib.util
from pathlib import Path
_S = importlib.util.spec_from_file_location(
    "sfx_scrub", Path(__file__).resolve().parent.parent / "tools" / "sfx_scrub.py")
sx = importlib.util.module_from_spec(_S); _S.loader.exec_module(sx)


def test_is_sfx_quote():
    for q in ("EUAACK...!! ACK!!! ACCK!!!", "HUH... HUH?!", "Keuk...!", "Hoh?", "GRR"):
        assert sx.is_sfx_quote(q), q
    for q in ("Kill him!", "How dare they dishonor my mother", "Serves you all right"):
        assert not sx.is_sfx_quote(q), q


def test_scrub_removes_sfx_keeps_dialogue():
    s = sx.scrub_sfx_quotes('He let out desperate cries of "EUAACK...!! ACK!!!" as he fell.')
    assert "EUAACK" not in s and "ACK" not in s
    assert s == "He let out desperate cries as he fell."
    assert "Kill him" in sx.scrub_sfx_quotes('The order rang out: "Kill him!"')


def test_is_fragment_quote_detects_incomplete_stubs():
    # incomplete trailing-ellipsis / leading-ellipsis / dangling-dash fragments
    for q in ("Ancestor...?", "...serves you all right.", "And then...",
              "Wait, what—", "But—"):
        assert sx.is_fragment_quote(q), q
    # complete punchy lines (even short) are NOT fragments and stay quotable
    for q in ("Kill him!", "Serves you all right.", "How dare you betray us",
              "I will end this now."):
        assert not sx.is_fragment_quote(q), q


def test_scrub_drops_fragment_quotes_keeps_punchy_quote():
    # the p95/p96 'Ancestor...?' incomplete fragment must not be voiced/quoted
    s = sx.scrub_sfx_quotes('He whispers, "Ancestor...?" and steps back.')
    assert "Ancestor" not in s
    assert s == "He whispers and steps back."
    # a real, complete, punchy quote survives the scrub
    assert "Serves them right" in sx.scrub_sfx_quotes(
        'He sneers, "Serves them right!"')


def test_droppable_quotes_reports_sfx_and_fragments():
    text = 'He yells "EUAACK!!" then mutters "Ancestor...?" before "Kill him!".'
    bad = sx.droppable_quotes(text)
    assert any("EUAACK" in b for b in bad)
    assert any("Ancestor" in b for b in bad)
    assert not any("Kill him" in b for b in bad)
