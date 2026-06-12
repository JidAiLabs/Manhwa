"""One choke point for local-LLM chat calls: every ollama request gets a
hard timeout so a dead or stale server fails fast and the caller's retry
logic takes over. Lesson (claw-mini, 2026-06-12): a long-lived ollama 0.24
service silently dropped a gemma4 request and the timeout-less client hung
the whole gpu lane for 30 minutes on a socket read.
"""
from __future__ import annotations

# cold 17 GB model load is ~2-3 min worst case; warm generation well under 2
TIMEOUT_SEC = 600


def chat(**kw):
    import ollama
    if hasattr(ollama, "Client"):
        return ollama.Client(timeout=TIMEOUT_SEC).chat(**kw)
    return ollama.chat(**kw)   # test fakes are bare modules with .chat only
