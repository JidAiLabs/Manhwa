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
