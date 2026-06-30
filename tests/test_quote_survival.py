# tests/test_quote_survival.py
from tools.sfx_scrub import is_droppable_quote, scrub_sfx_quotes


def test_iconic_short_quotes_survive():
    for q in ("Kill him!", "Ancestor!", "Damn you.", "I can't move."):
        assert not is_droppable_quote(q), q
    # garble / trailing-off still dropped:
    assert is_droppable_quote("EUAACK...!! ACK!!!")
    assert is_droppable_quote("Ancestor...?")
    # a kept quote stays in the line:
    assert "Kill him" in scrub_sfx_quotes('The order rang out: "Kill him!"')
