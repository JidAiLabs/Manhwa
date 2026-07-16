"""Ollama-protocol shim over mlx-vlm — gemma-4 on Apple MLX, zero caller changes.

Speaks POST /api/chat exactly like an Ollama daemon, backed by an mlx-vlm
vision model. The pipeline's gemma tools all route through
tools/ollama_compat.chat -> ollama.Client(), which honors OLLAMA_HOST — so the
whole A/B is an env flip:

    .mlx_vlm_venv/bin/python tools/mlx_ollama_shim.py          # serve :11500
    OLLAMA_HOST=http://127.0.0.1:11500 <any pipeline command>  # route to MLX

Ollama semantics honored (2026-07-16 hardening — each was a production bug):
  - options.num_predict: absent/<=0 = generate until done (512 default had
    truncated story_group's grouping JSON).
  - options.temperature: absent = 0.8, the ollama daemon default (callers
    that rely on the default must not silently become greedy).
  - options.top_p / top_k / repeat_penalty: forwarded when present.
  - format (JSON Schema dict or "json"): ollama grammar-CONSTRAINS decoding;
    MLX cannot, so we emulate — schema appended to the prompt, fence/prose
    tolerant JSON extraction, canonical re-serialization (message.content is
    ALWAYS valid JSON on success), and ONE self-repair retry at a bumped
    temperature. The retry is load-bearing: MLX at temp 0 is DETERMINISTIC,
    so without it a malformed roll repeats forever (ollama's nondeterminism
    was silently retry-rouletting this class).
  - think: forwarded to the chat template (gemma-4 uses enable_thinking);
    <think> blocks are stripped from output defensively either way.
  - options.num_ctx: IGNORED — MLX has no fixed KV window; long prompts
    simply work. (On ollama this truncates; callers still pass it for that
    backend's sake.)

The requested model name is logged but IGNORED — this shim serves ONE model
(env MLX_SHIM_MODEL, default gemma-4-26b-a4b-it-4bit). Runs in .mlx_vlm_venv
(own venv per local backend, house rule).

Parallel serving (2026-07-17, owner speed ladder #1): MLX_SHIM_INSTANCES
(default 2) worker PROCESSES each load their own model copy (~14GB unified
memory apiece; separate processes sidestep MLX thread-safety entirely) and
serve on PORT+1..PORT+N; the main process becomes a least-busy async proxy on
PORT. The writer lane stays sequential by story design (each beat's prompt
carries the previous narration) — the win is understanding / judges /
accept-better (440+ independent calls per chapter). MLX_SHIM_INSTANCES=1
keeps the classic single-process behavior exactly.

ponytail: within one worker, generation still serializes under one lock.
ponytail: format emulation is prompt+parse, not a real decoding grammar;
an outlines/grammar integration is the upgrade path if malformed-JSON
retries ever show up in the logs at scale.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODEL_ID = os.environ.get(
    "MLX_SHIM_MODEL", "mlx-community/gemma-4-26b-a4b-it-4bit")
PORT = int(os.environ.get("MLX_SHIM_PORT", "11500"))
INSTANCES = max(1, int(os.environ.get("MLX_SHIM_INSTANCES", "2")))
ROLE = os.environ.get("MLX_SHIM_ROLE", "")   # "" = entrypoint, "worker" = child
# ollama daemon defaults (Modelfile-less): keep callers' implicit contracts.
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 8192

app = FastAPI()
_lock = threading.Lock()
_model = _processor = _config = None

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _load():
    global _model, _processor, _config
    if _model is None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        print(f"[shim] loading {MODEL_ID} ...", flush=True)
        t0 = time.time()
        _model, _processor = load(MODEL_ID)
        _config = load_config(MODEL_ID)
        print(f"[shim] loaded in {time.time() - t0:.0f}s", flush=True)
    return _model, _processor, _config


def _images(msgs):
    from PIL import Image
    out = []
    for m in msgs:
        for im in m.get("images") or []:
            if isinstance(im, str) and os.path.exists(im):
                out.append(Image.open(im).convert("RGB"))
            else:  # ollama wire format: base64 string
                out.append(Image.open(
                    io.BytesIO(base64.b64decode(im))).convert("RGB"))
    return out


def _extract_json_value(text):
    """Fence/prose/think-tolerant JSON extraction — object OR array."""
    if not text:
        return None
    t = _THINK_RE.sub("", str(text))
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1)
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for o, c in (("[", "]"), ("{", "}")):
        s, e = t.find(o), t.rfind(c)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except Exception:
                continue
    return None


class _SchemaLogitsProcessor:
    """Grammar-constrained decoding (owner speed ladder #4): at every step,
    mask the logits to the token ids the JSON-schema automaton allows —
    invalid JSON becomes IMPOSSIBLE instead of retried. mlx-vlm passes
    (generated_tokens_so_far, logits) per step; we advance the guide with the
    newly sampled tokens and add a -inf mask for everything off-grammar.
    *guide* is duck-typed (outlines_core.Guide in prod, a fake in tests)."""

    def __init__(self, guide, eos_ids):
        self._guide = guide
        self._eos = sorted({int(e) for e in eos_ids if e is not None})
        self._consumed = 0
        self._dead = False

    def __call__(self, tokens, logits):
        import numpy as np
        try:
            import mlx.core as mx
        except ImportError:      # unit tests off the Apple stack: numpy ducks
            mx = np
        n = int(tokens.shape[-1]) if getattr(tokens, "size", 0) else 0
        if n > self._consumed and not self._dead:
            for t in tokens[self._consumed:].tolist():
                if self._guide.is_finished():
                    break
                try:
                    self._guide.advance(int(t))
                except Exception:            # left the automaton: stand down
                    self._dead = True
                    break
        self._consumed = n
        if self._dead:
            return logits
        vocab = int(logits.shape[-1])
        mask = np.full(vocab, -np.inf, dtype=np.float32)
        if self._guide.is_finished():
            ok = [e for e in self._eos if e < vocab]
            if not ok:
                return logits
            mask[ok] = 0.0
        else:
            allowed = [t for t in self._guide.get_tokens() if t < vocab]
            if not allowed:                  # dead end: never strand decoding
                return logits
            mask[allowed] = 0.0
        return logits + mx.array(mask)


_GRAMMAR = {"vocab": None, "index_cache": {}}


def _schema_processor(fmt):
    """Build a grammar processor for a JSON-Schema format dict; None when the
    grammar engine is unavailable or the schema is unsupported (the caller
    falls back to prompt-emulation + retry — grammar is an accelerator,
    never a dependency)."""
    if not isinstance(fmt, dict):
        return None
    try:
        from outlines_core import Guide, Index, Vocabulary
        from outlines_core.json_schema import build_regex_from_schema
        key = json.dumps(fmt, sort_keys=True)
        if key not in _GRAMMAR["index_cache"]:
            if _GRAMMAR["vocab"] is None:
                _GRAMMAR["vocab"] = Vocabulary.from_pretrained(MODEL_ID)
            regex = build_regex_from_schema(key)
            _GRAMMAR["index_cache"][key] = Index(regex, _GRAMMAR["vocab"])
        _model, processor, _config = _load()
        tok = getattr(processor, "tokenizer", processor)
        eos = {getattr(tok, "eos_token_id", None)}
        try:
            eot = tok.convert_tokens_to_ids("<end_of_turn>")
            if isinstance(eot, int) and eot >= 0:
                eos.add(eot)
        except Exception:                                   # noqa: BLE001
            pass
        return _SchemaLogitsProcessor(Guide(_GRAMMAR["index_cache"][key]),
                                      eos)
    except Exception as exc:                                # noqa: BLE001
        print(f"[shim] grammar unavailable ({type(exc).__name__}: {exc}) — "
              "falling back to prompt emulation", flush=True)
        return None


def _format_block(fmt) -> str:
    if isinstance(fmt, dict):
        return ("\n\nRespond with ONLY valid JSON that matches this JSON "
                "Schema EXACTLY (correct key spelling, all required keys, no "
                "prose, no code fences):\n" + json.dumps(fmt))
    if isinstance(fmt, str) and fmt.strip().lower() == "json":
        return "\n\nRespond with ONLY valid JSON. No prose, no code fences."
    return ""


def _gen_once(prompt: str, imgs, opts, think, temperature: float,
              grammar=None):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, config = _load()
    tmpl_kwargs = {}
    if think is not None:
        tmpl_kwargs["enable_thinking"] = bool(think)
    formatted = apply_chat_template(
        processor, config, prompt, num_images=len(imgs), **tmpl_kwargs)
    mt = int(opts.get("num_predict") or 0)
    if mt <= 0:
        mt = DEFAULT_MAX_TOKENS
    gen_kwargs = {"max_tokens": mt, "temperature": temperature,
                  "verbose": False}
    for src, dst in (("top_p", "top_p"), ("top_k", "top_k"),
                     ("repeat_penalty", "repetition_penalty")):
        if opts.get(src) is not None:
            gen_kwargs[dst] = opts[src]
    if grammar is not None:
        gen_kwargs["logits_processors"] = [grammar]
    t0 = time.time()
    res = generate(model, processor, formatted, imgs if imgs else None,
                   **gen_kwargs)
    text = _THINK_RE.sub("", str(getattr(res, "text", res)))
    ptok = int(getattr(res, "prompt_tokens", 0) or 0)
    gtok = int(getattr(res, "generation_tokens", 0) or 0)
    return text, ptok, gtok, time.time() - t0


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    msgs = body.get("messages") or []
    opts = body.get("options") or {}
    fmt = body.get("format")
    think = body.get("think")
    prompt = "\n\n".join(str(m.get("content") or "") for m in msgs
                         if m.get("content")) + _format_block(fmt)
    imgs = _images(msgs)
    temp = opts.get("temperature")
    temperature = DEFAULT_TEMPERATURE if temp is None else float(temp)

    with _lock:
        try:
            grammar = _schema_processor(fmt) if fmt else None
            text, ptok, gtok, dur = _gen_once(prompt, imgs, opts, think,
                                              temperature, grammar=grammar)
            retried = False
            if fmt:
                value = _extract_json_value(text)
                if value is None:
                    # deterministic-failure escape: same prompt at temp 0
                    # regenerates the same malformed JSON forever — ONE
                    # bumped-temperature retry with the failure named.
                    retried = True
                    retry_prompt = (
                        prompt + "\n\nYour previous response was not valid "
                        "JSON (it was malformed or wrapped in prose/fences). "
                        "Output ONLY the corrected, complete JSON — nothing "
                        "else.")
                    # a FORMATTING fix does not need the montage re-encoded:
                    # multi-image writer calls pay 60-80s per roll and the
                    # retry tax measured 79 of 200 minutes (job 48+49). A
                    # single-image call (understanding) keeps its image —
                    # dropping it would invite hallucinated re-description.
                    retry_imgs = imgs if len(imgs) <= 1 else []
                    # guides are stateful: the retry needs a FRESH one
                    text2, p2, g2, d2 = _gen_once(
                        retry_prompt, retry_imgs, opts, think,
                        max(0.4, temperature + 0.3),
                        grammar=_schema_processor(fmt) if fmt else None)
                    ptok += p2
                    gtok += g2
                    dur += d2
                    value = _extract_json_value(text2)
                    if value is None:
                        text = text2          # caller's own fallback engages
                if value is not None:
                    text = json.dumps(value, ensure_ascii=False)
        except Exception as exc:  # relay as an ollama-style error
            print(f"[shim] ERROR {type(exc).__name__}: {exc}", flush=True)
            return JSONResponse({"error": str(exc)}, status_code=500)

    print(f"[shim] model={body.get('model')} imgs={len(imgs)} "
          f"fmt={'schema' if isinstance(fmt, dict) else (fmt or '-')} "
          f"temp={temperature} ptok={ptok} gtok={gtok} "
          f"{'RETRIED ' if retried else ''}{dur:.1f}s", flush=True)
    return {
        "model": body.get("model") or MODEL_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": str(text)},
        "done": True,
        "done_reason": "stop",
        "total_duration": int(dur * 1e9),
        "prompt_eval_count": int(ptok),
        "eval_count": int(gtok),
    }


@app.get("/api/tags")  # `ollama list` parity so health checks work
async def tags():
    return {"models": [{"name": MODEL_ID, "model": MODEL_ID}]}


# --- least-busy proxy (MLX_SHIM_INSTANCES > 1) --------------------------------

proxy_app = FastAPI()
_backends: list = []          # [{"port": int, "inflight": int}]
_backends_lock = threading.Lock()


def _least_busy(backends):
    """Index of the backend with the fewest in-flight requests (stable on
    ties). Pure — the unit under test."""
    best = 0
    for i, b in enumerate(backends):
        if b["inflight"] < backends[best]["inflight"]:
            best = i
    return best


@proxy_app.post("/api/chat")
async def proxy_chat(req: Request):
    import httpx
    body = await req.json()
    last_err = "no backends"
    # model load takes ~10-60s per worker: retry across backends generously
    for _attempt in range(60):
        with _backends_lock:
            b = _backends[_least_busy(_backends)]
            b["inflight"] += 1
        try:
            async with httpx.AsyncClient(timeout=900.0) as client:
                r = await client.post(
                    f"http://127.0.0.1:{b['port']}/api/chat", json=body)
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as exc:                            # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            await __import__("asyncio").sleep(2.0)
        finally:
            with _backends_lock:
                b["inflight"] -= 1
    return JSONResponse({"error": f"all workers unreachable ({last_err})"},
                        status_code=502)


@proxy_app.get("/api/tags")
async def proxy_tags():
    return {"models": [{"name": MODEL_ID, "model": MODEL_ID}]}


def _run_pool() -> None:
    """Spawn INSTANCES worker processes (own model copy each) and serve the
    least-busy proxy on PORT. Workers die with the parent (SIGTERM handler +
    a ppid watchdog in the worker), so launchd manages ONE process."""
    import uvicorn
    procs: list = []
    env = dict(os.environ, MLX_SHIM_ROLE="worker")
    for i in range(INSTANCES):
        port = PORT + 1 + i
        _backends.append({"port": port, "inflight": 0})
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(env, MLX_SHIM_PORT=str(port))))
        print(f"[shim] worker {i + 1}/{INSTANCES} spawning on :{port} "
              f"(pid {procs[-1].pid})", flush=True)

    def _reap(_sig=None, _frm=None):
        for p in procs:
            try:
                p.terminate()
            except Exception:                               # noqa: BLE001
                pass
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _reap)
    signal.signal(signal.SIGINT, _reap)
    try:
        uvicorn.run(proxy_app, host="127.0.0.1", port=PORT,
                    log_level="warning")
    finally:
        _reap()


def _watch_parent() -> None:
    """Worker-side: exit when orphaned (parent proxy died) so no model copy
    outlives the daemon."""
    def _loop():
        while True:
            if os.getppid() == 1:
                os._exit(0)
            time.sleep(15)
    threading.Thread(target=_loop, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    if ROLE == "worker" or INSTANCES == 1:
        if ROLE == "worker":
            _watch_parent()
        _load()  # fail fast + warm before accepting traffic
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
    else:
        _run_pool()
