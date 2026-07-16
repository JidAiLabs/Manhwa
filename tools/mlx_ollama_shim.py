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

ponytail: generation is serialized under one lock (MLX single stream);
batched/parallel decode is the upgrade path if judge throughput matters.
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
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODEL_ID = os.environ.get(
    "MLX_SHIM_MODEL", "mlx-community/gemma-4-26b-a4b-it-4bit")
PORT = int(os.environ.get("MLX_SHIM_PORT", "11500"))
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


def _format_block(fmt) -> str:
    if isinstance(fmt, dict):
        return ("\n\nRespond with ONLY valid JSON that matches this JSON "
                "Schema EXACTLY (correct key spelling, all required keys, no "
                "prose, no code fences):\n" + json.dumps(fmt))
    if isinstance(fmt, str) and fmt.strip().lower() == "json":
        return "\n\nRespond with ONLY valid JSON. No prose, no code fences."
    return ""


def _gen_once(prompt: str, imgs, opts, think, temperature: float):
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
            text, ptok, gtok, dur = _gen_once(prompt, imgs, opts, think,
                                              temperature)
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
                    text2, p2, g2, d2 = _gen_once(
                        retry_prompt, imgs, opts, think,
                        max(0.4, temperature + 0.3))
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


if __name__ == "__main__":
    import uvicorn
    _load()  # fail fast + warm before accepting traffic
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
