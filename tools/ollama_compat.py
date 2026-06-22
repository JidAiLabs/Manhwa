"""One choke point for local-LLM chat calls, with TWO layers of protection:

1. an httpx client timeout (a dead server fails the socket read), and
2. a HARD wall-clock watchdog thread.

Why both: the httpx timeout is a *read* timeout — it does NOT fire when ollama
accepts the request and then stalls mid-generation. Measured (claw-mini,
2026-06-23): a gemma4 beats call at a 16k context kept invalidating its SWA
cache and re-processing the full prompt; it hung ~32 minutes despite the 600s
client timeout, with the gpu lane dead the whole time. The wall-clock watchdog
abandons the call after HARD_TIMEOUT_SEC no matter what, so the caller's
retry / fail-soft logic always gets control instead of blocking forever.
"""
from __future__ import annotations

import os
import queue
import threading

# inner httpx client timeout (socket-level)
TIMEOUT_SEC = 600
# hard wall-clock ceiling for ONE call. Covers a cold 17 GB model load (~2-3 min)
# plus a warm generation; a healthy call at <=8k ctx is well under a minute, so
# this only trips on a genuine stall. Env-tunable for slow hosts.
HARD_TIMEOUT_SEC = float(os.environ.get("STUDIO_OLLAMA_HARD_TIMEOUT", "420"))


def _raw_chat(**kw):
    import ollama
    if hasattr(ollama, "Client"):
        return ollama.Client(timeout=TIMEOUT_SEC).chat(**kw)
    return ollama.chat(**kw)   # test fakes are bare modules with .chat only


def chat(**kw):
    """Run one ollama chat under a hard wall-clock watchdog.

    Raises TimeoutError if the call exceeds HARD_TIMEOUT_SEC (the abandoned
    worker is a daemon thread; the inner httpx timeout eventually reaps it).
    Any backend exception is re-raised to the caller unchanged.
    """
    result_q: "queue.Queue" = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_q.put(("ok", _raw_chat(**kw)))
        except BaseException as exc:  # noqa: BLE001 - relay to the caller
            result_q.put(("err", exc))

    th = threading.Thread(target=_target, name="ollama-chat", daemon=True)
    th.start()
    th.join(HARD_TIMEOUT_SEC)
    if th.is_alive():
        raise TimeoutError(
            f"ollama chat exceeded {HARD_TIMEOUT_SEC:.0f}s hard timeout "
            f"(model={kw.get('model')!r}) — abandoned so the caller can retry")
    state, val = result_q.get_nowait()
    if state == "err":
        raise val
    return val
