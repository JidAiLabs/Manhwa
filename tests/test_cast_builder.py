

def test_the_json_repair_retry_calls_a_chat_function_that_exists():
    """The self-repair retry was dead code for months.

    Its own comment says it exists because "an unterminated string cost a 12-min
    job restart" — but it called `client.chat(...)` inside the OLLAMA branch,
    where no `client` was ever bound on the ollama path.
    So a truncated model response raised UnboundLocalError instead of retrying,
    masking the real JSONDecodeError and failing the chapter (ORV Episode 108).
    Even bound it would have been wrong: that client exposed
    models.generate_content(), never .chat().
    """
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "tools" / "cast_builder.py"
    assert "client.chat(" not in src.read_text(), (
        "cast_builder calls client.chat() — no such client exists and "
        "has no .chat(); the ollama path must use _ollama_chat()")


def test_series_registry_locks_names_and_looks_verbatim():
    """ORV Ep128: cast_builder re-guessed Dokja's look per chapter and once
    gave him Michio's (glasses). The owner's series registry is authoritative:
    matched members take its name/look/not verbatim, a second gemma member
    for the same entry folds in, unlisted characters stay as guessed."""
    from tools.cast_builder import apply_series_cast
    guessed = {"cast": [
        {"id": "protagonist", "canonical_name": "our protagonist",
         "is_protagonist": True, "aliases": ["Dokja Kim"], "role": "protagonist",
         "visual_description": "dark messy hair and glasses"},
        {"id": "kim", "canonical_name": "Kim Dokja", "is_protagonist": False,
         "aliases": [], "role": "minor", "visual_description": "a man"},
        {"id": "michio", "canonical_name": "Michio", "is_protagonist": False,
         "aliases": ["Shoji"], "role": "ally", "visual_description": "glasses"},
        {"id": "izumi", "canonical_name": "Izumi", "is_protagonist": False,
         "aliases": [], "role": "minor", "visual_description": "a tall woman"},
    ]}
    registry = {"cast": [
        {"canonical_name": "our protagonist", "is_protagonist": True,
         "aliases": ["Dokja Kim", "Kim Dokja"], "not": ["glasses"],
         "visual_description": "dark hair, long white coat over a black shirt"},
        {"canonical_name": "Michio Shoji", "aliases": ["Michio"],
         "visual_description": "dark hair and glasses, black button-down shirt"},
        {"canonical_name": "Yoo Joonghyuk", "aliases": [],
         "visual_description": "absent this chapter"},
    ]}
    out, n = apply_series_cast(guessed, registry)
    names = [m["canonical_name"] for m in out["cast"]]
    assert n == 2 and names == ["our protagonist", "Michio Shoji", "Izumi"]
    dokja = out["cast"][0]
    assert dokja["visual_description"] == "dark hair, long white coat over a black shirt"
    assert dokja["not"] == ["glasses"] and dokja["is_protagonist"] is True
    assert "Kim Dokja" in dokja["aliases"] and "Dokja Kim" in dokja["aliases"]
    assert out["cast"][1]["aliases"] == ["Michio", "Shoji"]
    assert out["cast"][2]["visual_description"] == "a tall woman"   # untouched
    assert "Yoo Joonghyuk" not in names                             # not injected
