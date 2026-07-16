"""Ollama-protocol shim over mlx-vlm — gemma-4 on Apple MLX, zero caller changes.

Speaks POST /api/chat exactly like an Ollama daemon, backed by an mlx-vlm
vision model. The pipeline's 9 gemma tools all route through
tools/ollama_compat.chat -> ollama.Client(), which honors OLLAMA_HOST — so the
whole A/B is an env flip:

    .mlx_vlm_venv/bin/python tools/mlx_ollama_shim.py          # serve :11500
    OLLAMA_HOST=http://127.0.0.1:11500 <any pipeline command>  # route to MLX

The requested model name is logged but IGNORED — this shim serves ONE model
(env MLX_SHIM_MODEL, default gemma-4-26b-a4b-it-4bit). Runs in .mlx_vlm_venv
(own venv per local backend, house rule).

ponytail: generation is serialized under one lock (MLX single stream);
batched/parallel decode is the upgrade path if judge throughput matters.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODEL_ID = os.environ.get(
    "MLX_SHIM_MODEL", "mlx-community/gemma-4-26b-a4b-it-4bit")
PORT = int(os.environ.get("MLX_SHIM_PORT", "11500"))

app = FastAPI()
_lock = threading.Lock()
_model = _processor = _config = None


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


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    msgs = body.get("messages") or []
    opts = body.get("options") or {}
    prompt = "\n\n".join(str(m.get("content") or "") for m in msgs
                         if m.get("content"))
    imgs = _images(msgs)

    def _run():
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor, config = _load()
        formatted = apply_chat_template(
            processor, config, prompt, num_images=len(imgs))
        t0 = time.time()
        # ollama semantics: num_predict absent / <=0 = generate until done.
        # 512 truncated story_group's grouping JSON mid-generation (job 39) —
        # default LARGE, never small.
        mt = int(opts.get("num_predict") or 0)
        if mt <= 0:
            mt = 8192
        res = generate(model, processor, formatted, imgs if imgs else None,
                       max_tokens=mt,
                       temperature=float(opts.get("temperature") or 0.0),
                       verbose=False)
        text = getattr(res, "text", res)
        ptok = getattr(res, "prompt_tokens", 0) or 0
        gtok = getattr(res, "generation_tokens", 0) or 0
        return text, ptok, gtok, time.time() - t0

    with _lock:
        try:
            text, ptok, gtok, dur = _run()
        except Exception as exc:  # relay as an ollama-style error
            print(f"[shim] ERROR {type(exc).__name__}: {exc}", flush=True)
            return JSONResponse({"error": str(exc)}, status_code=500)

    print(f"[shim] model={body.get('model')} imgs={len(imgs)} "
          f"ptok={ptok} gtok={gtok} {dur:.1f}s", flush=True)
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
