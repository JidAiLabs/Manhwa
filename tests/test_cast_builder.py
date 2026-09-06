

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
